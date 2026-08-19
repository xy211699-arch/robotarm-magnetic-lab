"""Dynamic-capsule runtime bridge for the TASK-007 analytical virtual magnet."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.managers import ManagerTermBase

from robotarm_magnetic_lab.magnetics import FiniteMagnetSystem, config_sha256, load_config

from ..controllers.virtual_magnet import (
    ActionResult,
    Lifecycle,
    VirtualMagnetElevenActionController,
    load_profile,
    profile_sha256,
)
from ..controllers.virtual_magnet.types import ControllerState


def _quat_xyzw_to_matrix(quaternion) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return Rotation.from_quat(quaternion).as_matrix()


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm <= 1.0e-12:
        return vector
    return vector * limit / norm


def _finite_array(value, shape) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("non-finite runtime state")
    return result


class VirtualMagnetBridge(ManagerTermBase):
    """Own the pure controller and apply only model-produced capsule wrench."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("TASK-007 bridge supports exactly one environment")
        params = dict(getattr(cfg, "params", {}) or {})
        asset_name = str(params.get("asset_name", "capsule"))
        contact_sensor_name = str(params.get("contact_sensor_name", "capsule_contact"))
        self.profile_path = str(params.get("profile_path", ""))
        self.debug_xform = bool(params.get("debug_xform", True))
        self.env = env
        self.capsule = env.scene[asset_name]
        self.contact = env.scene[contact_sensor_name]
        self.profile = load_profile(self.profile_path or None)
        self.magnetic_config = load_config()
        self.model = FiniteMagnetSystem(self.magnetic_config)
        self.cylinder_cfg = self.magnetic_config["magnets"]["target_cylinder"]
        self.sim_cfg = self.magnetic_config["simulation"]
        self.capsule_cfg = self.magnetic_config["external_magnet"]["capsule"]
        self._capsule_magnet_position = np.zeros(3, dtype=np.float64)
        self._capsule_magnet_rotation = np.eye(3, dtype=np.float64)
        self._filtered_wrench = np.zeros(6, dtype=np.float64)
        self._elapsed_s = 0.0
        self._physics_substeps = 0
        self._last_sidewall_contact_substep = -1_000_000
        self._last_result: ActionResult | None = None
        self._last_lifecycle = Lifecycle.READY
        self._terminal_serial = 0
        self._request_serial = 0

        state = self._read_state()
        nominal_position = state.capsule_magnet_position + state.capsule_magnet_rotation @ np.asarray(
            self.profile.nominal_position_capsule_m, dtype=np.float64
        )
        nominal_rotation = state.capsule_magnet_rotation @ Rotation.from_quat(
            self.profile.nominal_quaternion_capsule_xyzw
        ).as_matrix()
        self.controller = VirtualMagnetElevenActionController(
            self.profile,
            self._model_wrench,
            initial_magnet_position=nominal_position,
            initial_magnet_quaternion_xyzw=Rotation.from_matrix(nominal_rotation).as_quat(),
        )
        self._idle_magnet_position = nominal_position.copy()
        self._idle_magnet_quaternion = Rotation.from_matrix(nominal_rotation).as_quat()
        self.audit = self._empty_audit()
        env._virtual_magnet_bridge = self
        env._virtual_magnet_audit = self.audit
        if self.debug_xform:
            self._create_debug_xform()
        print(
            "VIRTUAL_MAGNET_BRIDGE_READY "
            f"model_sha256={config_sha256()} profile_sha256={profile_sha256(self.profile_path or None)} "
            "capsule_dynamic=true arm_ball_actuation=false",
            flush=True,
        )

    def _empty_audit(self) -> dict:
        return {
            "physics_substeps": 0,
            "action_substeps": 0,
            "feedback_updates": 0,
            "request_serial": 0,
            "terminal_serial": 0,
            "lifecycle": Lifecycle.READY.value,
            "result": None,
            "action_id": None,
            "desired_wrench": np.zeros(6),
            "model_raw_wrench": np.zeros(6),
            "model_filtered_wrench": np.zeros(6),
            "applied_wrench": np.zeros(6),
            "virtual_magnet_position": np.zeros(3),
            "virtual_magnet_quaternion_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
            "virtual_magnet_relative_position": np.zeros(3),
            "solver_saturated": False,
            "inverse_condition_number": 0.0,
            "constrained": False,
            "low_effect": False,
            "optical_axis_error_deg": 0.0,
            "tangent_drift_m": 0.0,
            "move_signed_displacement_m": 0.0,
            "linear_speed_m_s": 0.0,
            "angular_speed_rad_s": 0.0,
            "camera_contact": False,
            "sidewall_contact": False,
            "contact_force_n": 0.0,
            "capsule_dynamic": True,
        }

    def _model_wrench(self, main_position: np.ndarray, main_rotation: np.ndarray) -> np.ndarray:
        force, torque = self.model.force_torque_si(
            main_position,
            main_rotation,
            self._capsule_magnet_position,
            self._capsule_magnet_rotation,
        )
        return np.concatenate((force, torque)).astype(np.float64)

    def _read_state(self) -> ControllerState:
        position = _finite_array(self.capsule.data.root_pos_w.torch[0].detach().cpu().numpy(), 3)
        quaternion = _finite_array(self.capsule.data.root_quat_w.torch[0].detach().cpu().numpy(), 4)
        rotation = _quat_xyzw_to_matrix(quaternion)
        linear_velocity = _finite_array(self.capsule.data.root_lin_vel_w.torch[0].detach().cpu().numpy(), 3)
        angular_velocity = _finite_array(self.capsule.data.root_ang_vel_w.torch[0].detach().cpu().numpy(), 3)
        magnet_offset = np.array(
            [0.0, 0.0, float(self.cylinder_cfg["center_offset_axis_m"])], dtype=np.float64
        )
        self._capsule_magnet_position = position + rotation @ magnet_offset
        self._capsule_magnet_rotation = rotation

        # The DS01 optical axis is capsule local -Z. The capsule long axis is
        # unsigned for MOVE eligibility, so local +Z is sufficient there.
        optical_axis = -rotation[:, 2]
        long_axis = rotation[:, 2]
        camera_up = rotation[:, 1]
        camera_right = rotation[:, 0]
        inward_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        contact_force = _finite_array(self.contact.data.net_forces_w.torch[0, 0].detach().cpu().numpy(), 3)
        contact = float(np.linalg.norm(contact_force)) > 1.0e-4
        tilt = math.degrees(math.acos(float(np.clip(abs(np.dot(long_axis, inward_normal)), 0.0, 1.0))))
        sidewall_contact = contact and tilt >= self.profile.move_tilt_min_deg
        camera_contact = contact and not sidewall_contact
        if sidewall_contact:
            self._last_sidewall_contact_substep = self.controller.total_substeps if hasattr(self, "controller") else 0
        contact_point = position - inward_normal * float(np.dot(position, inward_normal))
        return ControllerState(
            capsule_position=position,
            capsule_rotation=rotation,
            capsule_magnet_position=self._capsule_magnet_position.copy(),
            capsule_magnet_rotation=rotation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            optical_axis=optical_axis,
            camera_up=camera_up,
            camera_right=camera_right,
            long_axis=long_axis,
            inward_normal=inward_normal,
            contact_point=contact_point,
            camera_contact=camera_contact,
            sidewall_contact=sidewall_contact,
            last_sidewall_contact_substep=self._last_sidewall_contact_substep,
        )

    def submit(self, action_id: int) -> bool:
        state = self._read_state()
        if self.controller.lifecycle == Lifecycle.TERMINAL:
            self.controller.acknowledge()
        accepted = self.controller.submit(action_id, state)
        self._request_serial += 1
        self.audit["request_serial"] = self._request_serial
        if accepted:
            self.audit["action_id"] = int(action_id)
            self.audit["result"] = None
            self.audit["lifecycle"] = Lifecycle.EXECUTING.value
        return accepted

    def __call__(
        self,
        env,
        env_ids,
        asset_name: str = "capsule",
        contact_sensor_name: str = "capsule_contact",
        profile_path: str = "",
        debug_xform: bool = True,
    ) -> None:
        # Parameters are consumed during ManagerTermBase construction. They
        # remain explicit here because EventManager validates callable
        # signatures before accepting configured params.
        return

    def reset(self, env_ids=None) -> None:
        self.controller.reset()
        self._elapsed_s = 0.0
        self._physics_substeps = 0
        self._last_sidewall_contact_substep = -1_000_000
        self._filtered_wrench[:] = 0.0
        self._last_result = None
        self._last_lifecycle = Lifecycle.READY
        state = self._read_state()
        self._idle_magnet_position = state.capsule_magnet_position + state.capsule_magnet_rotation @ np.asarray(
            self.profile.nominal_position_capsule_m, dtype=np.float64
        )
        self._idle_magnet_quaternion = Rotation.from_matrix(
            state.capsule_magnet_rotation
            @ Rotation.from_quat(self.profile.nominal_quaternion_capsule_xyzw).as_matrix()
        ).as_quat()
        self.audit.clear()
        self.audit.update(self._empty_audit())
        self.capsule.permanent_wrench_composer.reset(env_ids=env_ids)

    def physics_step(self) -> None:
        state = self._read_state()
        command = self.controller.step(state)
        self._elapsed_s += float(self.env.physics_dt)
        self._physics_substeps += 1

        raw_wrench = self._model_wrench(
            self._idle_magnet_position,
            Rotation.from_quat(self._idle_magnet_quaternion).as_matrix(),
        )
        desired_wrench = np.zeros(6, dtype=np.float64)
        magnet_position = self._idle_magnet_position
        magnet_quaternion = self._idle_magnet_quaternion
        if command is not None:
            raw_wrench = command.model_wrench.copy()
            desired_wrench = command.desired_wrench.copy()
            magnet_position = command.virtual_magnet_position.copy()
            magnet_quaternion = command.virtual_magnet_quaternion_xyzw.copy()
            self._idle_magnet_position = magnet_position.copy()
            self._idle_magnet_quaternion = magnet_quaternion.copy()

        force = _clip_norm(raw_wrench[:3], self.profile.max_desired_force_n)
        stabilization = (
            self.controller.telemetry.action_id is not None
            and (
                self.controller.telemetry.action_id == 0
                or self.controller.telemetry.substep >= self.profile.motion_substeps
            )
        )
        torque_limit = (
            self.profile.stabilization_max_desired_torque_nm
            if stabilization
            else self.profile.max_desired_torque_nm
        )
        torque = _clip_norm(raw_wrench[3:], torque_limit)
        ramp = min(self._elapsed_s / max(self.profile.coupling_ramp_s, 1.0e-9), 1.0)
        force = force * ramp - float(self.capsule_cfg.get("linear_drag_n_per_m_s", 0.0)) * state.linear_velocity
        torque = torque * ramp - float(self.capsule_cfg.get("angular_drag_nm_per_rad_s", 0.0)) * state.angular_velocity
        force = _clip_norm(force, self.profile.max_desired_force_n)
        torque = _clip_norm(torque, torque_limit)
        target = np.concatenate((force, torque))
        tau = max(self.profile.wrench_filter_time_constant_s, 0.0)
        alpha = 1.0 if tau <= 0.0 else 1.0 - math.exp(-float(self.env.physics_dt) / tau)
        self._filtered_wrench += alpha * (target - self._filtered_wrench)

        force_tensor = torch.as_tensor(self._filtered_wrench[:3], device=self.env.device, dtype=torch.float32).reshape(1, 1, 3)
        torque_tensor = torch.as_tensor(self._filtered_wrench[3:], device=self.env.device, dtype=torch.float32).reshape(1, 1, 3)
        position_tensor = torch.as_tensor(state.capsule_magnet_position, device=self.env.device, dtype=torch.float32).reshape(1, 1, 3)
        self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force_tensor,
            torques=torque_tensor,
            positions=position_tensor,
            body_ids=[0],
            env_ids=torch.tensor([0], device=self.env.device),
            is_global=True,
        )
        # Development invariant: the only applied controller wrench is the
        # filtered output of the repository-local finite-magnet model.
        applied = np.concatenate((force_tensor.cpu().numpy().reshape(3), torque_tensor.cpu().numpy().reshape(3)))
        if not np.allclose(applied, self._filtered_wrench, rtol=1.0e-6, atol=1.0e-8):
            raise AssertionError("applied capsule wrench differs from filtered finite-model wrench")

        telemetry = self.controller.telemetry
        if telemetry.lifecycle == Lifecycle.TERMINAL and self._last_lifecycle != Lifecycle.TERMINAL:
            self._terminal_serial += 1
        self._last_lifecycle = telemetry.lifecycle
        self._last_result = telemetry.result
        contact_force = float(np.linalg.norm(self.contact.data.net_forces_w.torch[0, 0].detach().cpu().numpy()))
        self.audit.update(
            physics_substeps=self._physics_substeps,
            action_substeps=telemetry.substep,
            feedback_updates=telemetry.feedback_updates,
            terminal_serial=self._terminal_serial,
            lifecycle=telemetry.lifecycle.value,
            result=None if telemetry.result is None else telemetry.result.value,
            desired_wrench=desired_wrench,
            model_raw_wrench=raw_wrench,
            model_filtered_wrench=self._filtered_wrench.copy(),
            applied_wrench=applied,
            virtual_magnet_position=np.asarray(magnet_position).copy(),
            virtual_magnet_quaternion_xyzw=np.asarray(magnet_quaternion).copy(),
            virtual_magnet_relative_position=np.asarray(magnet_position) - state.capsule_magnet_position,
            solver_saturated=telemetry.solver_saturated,
            inverse_condition_number=telemetry.inverse_condition_number,
            constrained=telemetry.constrained,
            low_effect=telemetry.low_effect,
            optical_axis_error_deg=telemetry.optical_axis_error_deg,
            tangent_drift_m=telemetry.tangent_drift_m,
            move_signed_displacement_m=telemetry.move_signed_displacement_m,
            linear_speed_m_s=telemetry.linear_speed_m_s,
            angular_speed_rad_s=telemetry.angular_speed_rad_s,
            camera_contact=state.camera_contact,
            sidewall_contact=state.sidewall_contact,
            contact_force_n=contact_force,
        )
        if command is not None:
            self._update_debug_xform(magnet_position, magnet_quaternion)

    def _create_debug_xform(self) -> None:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        self._debug_path = "/World/envs/env_0/Scene/MagneticDemo/virtual_external_magnet"
        UsdGeom.Xform.Define(stage, self._debug_path)

    def _update_debug_xform(self, position, quaternion_xyzw) -> None:
        if not hasattr(self, "_debug_path"):
            return
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(self._debug_path))
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
        matrix[:3, 3] = position
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))


def virtual_magnet_public_observation(env) -> torch.Tensor:
    """Public non-privileged action status; capsule truth stays internal."""
    bridge = getattr(env, "_virtual_magnet_bridge", None)
    result_code = 0.0
    lifecycle_code = 0.0
    action_id = -1.0
    if bridge is not None:
        lifecycle_code = {
            Lifecycle.READY: 0.0,
            Lifecycle.EXECUTING: 1.0,
            Lifecycle.TERMINAL: 2.0,
            Lifecycle.FAULT: 3.0,
        }[bridge.controller.lifecycle]
        result_code = {
            None: 0.0,
            ActionResult.COMPLETED: 1.0,
            ActionResult.REJECTED: 2.0,
            ActionResult.FAULT: 3.0,
        }[bridge.controller.telemetry.result]
        if bridge.controller.telemetry.action_id is not None:
            action_id = float(bridge.controller.telemetry.action_id)
    return torch.tensor([[action_id, lifecycle_code, result_code]], device=env.device, dtype=torch.float32)
