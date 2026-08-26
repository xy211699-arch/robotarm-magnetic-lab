"""Isaac Lab action term for the 10 Hz parameterized-force controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw

from ..controllers.dynamic_force_macro import camera_sphere_centers_local, equivalent_com_wrench
from ..controllers.parameterized_force import (
    ParameterizedForceConfig,
    ParameterizedForceMode,
    parameterized_endpoint_forces,
    validate_parameterized_command,
)


@dataclass(frozen=True)
class ParameterizedForceTelemetry:
    control_cycle: int
    physics_step_in_cycle: int
    mode: int
    mode_name: str
    alpha: float
    force_ratio: float
    target_total_force_n: float
    submitted_force_world: tuple[float, float, float]
    submitted_torque_world: tuple[float, float, float]
    direction_world: tuple[float, float, float]
    com_world: tuple[float, float, float]
    camera_center_world: tuple[float, float, float]
    other_center_world: tuple[float, float, float]
    camera_axis_world: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


class ParameterizedForceAction(ActionTerm):
    """Apply one ``[mode, alpha]`` pair for every 10 Hz environment step."""

    cfg: "ParameterizedForceActionTermCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("parameterized-force visualization supports exactly one environment")
        self.capsule = env.scene[cfg.asset_name]
        self.camera_sensor = env.scene[cfg.camera_sensor_name]
        self.camera_center_local, self.other_center_local = camera_sphere_centers_local(
            np.asarray(self.camera_sensor.cfg.offset.pos, dtype=np.float64),
            cfg.cylinder_height_m,
        )
        self.config = ParameterizedForceConfig(
            move_min_ratio=cfg.move_min_ratio,
            move_max_ratio=cfg.move_max_ratio,
            view_min_ratio=cfg.view_min_ratio,
            view_max_ratio=cfg.view_max_ratio,
            up_min_ratio=cfg.up_min_ratio,
            up_max_ratio=cfg.up_max_ratio,
        )
        self.mass_kg = float(self.capsule.data.body_mass.torch.reshape(-1)[0].item())
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise RuntimeError("live capsule mass must be finite and positive")
        self._raw_actions = torch.zeros((1, 2), device=env.device)
        self._processed_actions = torch.zeros((1, 2), device=env.device)
        self.mode = ParameterizedForceMode.HOLD
        self.alpha = 0.5
        self.control_cycle = -1
        self.physics_step_in_cycle = 0
        self.last_telemetry: ParameterizedForceTelemetry | None = None
        self.current_cycle_trace: list[ParameterizedForceTelemetry] = []
        self._verify_dynamic_body_and_enable_ccd()

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        values = actions.reshape(-1)
        if values.numel() != 2:
            raise ValueError("parameterized action must contain exactly [mode_id, alpha]")
        mode_value = float(values[0].item())
        rounded = round(mode_value)
        if not np.isfinite(mode_value) or abs(mode_value - rounded) > 1.0e-6:
            raise ValueError("mode_id must be integral")
        mode, alpha = validate_parameterized_command(rounded, float(values[1].item()))
        self.mode = mode
        self.alpha = alpha
        self._raw_actions[:] = actions
        self._processed_actions[:, 0] = float(mode)
        self._processed_actions[:, 1] = alpha
        self.control_cycle += 1
        self.physics_step_in_cycle = 0
        self.current_cycle_trace = []
        if mode == ParameterizedForceMode.HOLD:
            self.capsule.permanent_wrench_composer.reset()

    def _geometry(self):
        link_pos = self.capsule.data.root_link_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        link_quat = self.capsule.data.root_link_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
        com = self.capsule.data.root_com_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        rotation = rotation_matrix_from_xyzw(link_quat)
        camera = link_pos + rotation @ self.camera_center_local
        other = link_pos + rotation @ self.other_center_local
        axis = camera - other
        axis /= np.linalg.norm(axis)
        return com, camera, other, axis, link_quat

    def apply_actions(self) -> None:
        com, camera, other, axis, quaternion = self._geometry()
        command = parameterized_endpoint_forces(
            self.mode,
            self.alpha,
            mass_kg=self.mass_kg,
            camera_axis_world=axis,
            config=self.config,
        )
        if self.mode == ParameterizedForceMode.HOLD:
            self.capsule.permanent_wrench_composer.reset()
            resultant = np.zeros(3, dtype=np.float64)
            submitted_torque = np.zeros(3, dtype=np.float64)
        else:
            points = []
            if np.linalg.norm(command.camera_force_world) > 0.0:
                points.append(type("Point", (), {"position_world": camera, "force_world": command.camera_force_world})())
            if np.linalg.norm(command.other_force_world) > 0.0:
                points.append(type("Point", (), {"position_world": other, "force_world": command.other_force_world})())
            resultant, submitted_torque = equivalent_com_wrench(tuple(points), com)
            application_position = None
            torque_tensor = torch.as_tensor(
                submitted_torque, device=self._env.device, dtype=torch.float32
            ).reshape(1, 1, 3)
            if self.mode == ParameterizedForceMode.UP:
                # UP is one physical force at the resolved camera-side sphere
                # center. PhysX generates its moment naturally at that point.
                resultant = command.camera_force_world
                submitted_torque = np.zeros(3, dtype=np.float64)
                torque_tensor = None
                application_position = camera
            force_tensor = torch.as_tensor(
                resultant, device=self._env.device, dtype=torch.float32
            ).reshape(1, 1, 3)
            position_tensor = None
            if application_position is not None:
                position_tensor = torch.as_tensor(
                    application_position, device=self._env.device, dtype=torch.float32
                ).reshape(1, 1, 3)
            self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force_tensor,
                torques=torque_tensor,
                positions=position_tensor,
                body_ids=None,
                env_ids=None,
                is_global=True,
            )
        self.last_telemetry = ParameterizedForceTelemetry(
            self.control_cycle,
            self.physics_step_in_cycle,
            int(self.mode),
            self.mode.name,
            self.alpha,
            command.force_ratio,
            command.target_total_force_n,
            tuple(float(value) for value in resultant),
            tuple(float(value) for value in submitted_torque),
            tuple(float(value) for value in command.direction_world),
            tuple(float(value) for value in com),
            tuple(float(value) for value in camera),
            tuple(float(value) for value in other),
            tuple(float(value) for value in axis),
            tuple(float(value) for value in quaternion),
        )
        self.current_cycle_trace.append(self.last_telemetry)
        self.physics_step_in_cycle += 1

    def reset(self, env_ids=None) -> None:
        self._raw_actions.zero_()
        self._processed_actions.zero_()
        self.mode = ParameterizedForceMode.HOLD
        self.alpha = 0.5
        self.control_cycle = -1
        self.physics_step_in_cycle = 0
        self.last_telemetry = None
        self.current_cycle_trace = []
        self.capsule.permanent_wrench_composer.reset(env_ids=env_ids)

    def _verify_dynamic_body_and_enable_ccd(self) -> None:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.capsule.root_view.prim_paths[0])
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        if not rigid_api or (rigid_api.GetKinematicEnabledAttr() and bool(rigid_api.GetKinematicEnabledAttr().Get())):
            raise RuntimeError("parameterized-force capsule must be non-kinematic")
        physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_api:
            physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if physx_api.GetDisableGravityAttr() and bool(physx_api.GetDisableGravityAttr().Get()):
            raise RuntimeError("parameterized-force capsule gravity must remain enabled")
        ccd = physx_api.GetEnableCCDAttr() or physx_api.CreateEnableCCDAttr()
        ccd.Set(True)


@configclass
class ParameterizedForceActionTermCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ParameterizedForceAction
    asset_name: str = "capsule"
    camera_sensor_name: str = "capsule_camera"
    cylinder_height_m: float = 0.012
    move_min_ratio: float = 0.70
    move_max_ratio: float = 1.40
    view_min_ratio: float = 0.20
    view_max_ratio: float = 0.50
    up_min_ratio: float = 0.80
    up_max_ratio: float = 1.05
