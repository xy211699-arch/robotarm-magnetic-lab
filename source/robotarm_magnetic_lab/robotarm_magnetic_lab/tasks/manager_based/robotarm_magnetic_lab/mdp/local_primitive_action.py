"""Isaac Lab COM-wrench adapter for TASK-004 local primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
import math

import numpy as np
import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from ..controllers.local_primitives import (
    CapsuleState,
    LocalPrimitiveController,
    LocalPrimitiveControllerCfg,
    PrimitiveId,
    PrimitiveRequest,
    PrimitiveTelemetry,
    make_local_primitive_controller_cfg,
    simulation_profile_sha256,
)


class PrimitiveCommandDecoder:
    """Decode a finite four-float action on the rising edge of start pulse."""

    def __init__(self) -> None:
        self._pulse_high = False

    def reset(self) -> None:
        self._pulse_high = False

    def decode(self, action) -> PrimitiveRequest | None:
        values = np.asarray(action, dtype=np.float64).reshape(4)
        if not bool(np.isfinite(values).all()):
            raise ValueError("local primitive action must be finite")
        pulse_high = bool(values[0] > 0.5)
        rising = pulse_high and not self._pulse_high
        self._pulse_high = pulse_high
        if not rising:
            return None
        code_value = float(values[1])
        if not code_value.is_integer() or not 0 <= int(code_value) <= 3:
            raise ValueError("primitive code must be one of 0, 1, 2, 3")
        direction = values[2:4]
        norm = float(np.linalg.norm(direction))
        azimuth = 0.0 if norm <= 1.0e-12 else math.atan2(float(direction[1]), float(direction[0]))
        return PrimitiveRequest(PrimitiveId(int(code_value)), azimuth)


class LocalPrimitiveAction(ActionTerm):
    """Apply the pure controller output as a persistent global COM wrench."""

    cfg: "LocalPrimitiveActionTermCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("LocalPrimitiveAction requires exactly one environment")
        self.capsule = env.scene[cfg.asset_name]
        self._physics_dt_s = float(env.physics_dt)
        if not math.isclose(self._physics_dt_s, 1.0 / 240.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("LocalPrimitiveAction requires 240 Hz physics")
        mass = float(self.capsule.data.body_mass.torch[0, 0].item())
        if not math.isfinite(mass) or mass <= 0.0:
            raise RuntimeError("live capsule mass must be finite and positive")
        if abs(mass - cfg.controller_cfg.capsule_mass_kg) > 1.0e-6:
            raise RuntimeError(
                f"LOCAL_PRIMITIVE_NEEDS_DECISION live mass {mass:.10g} kg does not match "
                f"design mass {cfg.controller_cfg.capsule_mass_kg:.10g} kg"
            )
        runtime_cfg = replace(cfg.controller_cfg, capsule_mass_kg=mass)
        if cfg.profile_sha256 != runtime_cfg.profile_sha256:
            raise RuntimeError("action profile digest does not match controller profile digest")
        self.controller = LocalPrimitiveController(runtime_cfg)
        self.decoder = PrimitiveCommandDecoder()
        self._raw_actions = torch.zeros((1, 4), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._applied_force_world = torch.zeros((1, 3), device=env.device)
        self._applied_torque_world = torch.zeros((1, 3), device=env.device)
        self._pending_request: PrimitiveRequest | None = None
        self._telemetry: PrimitiveTelemetry | None = None
        self._substep_telemetry: deque[PrimitiveTelemetry] = deque(maxlen=16384)
        self._substep_positions_world_m: deque[np.ndarray] = deque(maxlen=16384)
        self._sim_time_s = 0.0
        self._last_request_result = "none"
        self._verify_dynamic_invariants(env)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def applied_force_world(self) -> torch.Tensor:
        return self._applied_force_world

    @property
    def applied_torque_world(self) -> torch.Tensor:
        return self._applied_torque_world

    @property
    def telemetry(self) -> PrimitiveTelemetry | None:
        return self._telemetry

    @property
    def last_request_result(self) -> str:
        return self._last_request_result

    @property
    def substep_telemetry(self) -> tuple[PrimitiveTelemetry, ...]:
        return tuple(self._substep_telemetry)

    @property
    def substep_positions_world_m(self) -> tuple[np.ndarray, ...]:
        """Immutable copies of COM positions sampled at the 240 Hz action cadence."""

        return tuple(position.copy() for position in self._substep_positions_world_m)

    @property
    def profile_sha256(self) -> str:
        return self.cfg.profile_sha256

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("local primitive action must be finite")
        self._processed_actions[:] = actions
        request = self.decoder.decode(actions[0].detach().cpu().numpy())
        if request is not None:
            self._pending_request = request

    def apply_actions(self) -> None:
        state = self._read_state()
        if self._pending_request is not None:
            accepted = self.controller.start(
                self._pending_request.primitive_id,
                self._pending_request.azimuth_rad,
                state,
            )
            self._last_request_result = "accepted" if accepted else self.controller.last_request_result
            self._pending_request = None
        wrench, telemetry = self.controller.update(state, self._physics_dt_s)
        self._telemetry = telemetry
        self._substep_telemetry.append(telemetry)
        self._substep_positions_world_m.append(state.position_world_m.copy())
        self._applied_force_world[0] = torch.tensor(
            wrench.force_world_n, device=self._applied_force_world.device,
            dtype=self._applied_force_world.dtype,
        )
        self._applied_torque_world[0] = torch.tensor(
            wrench.torque_world_nm, device=self._applied_torque_world.device,
            dtype=self._applied_torque_world.dtype,
        )
        self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._applied_force_world[:, None, :],
            torques=self._applied_torque_world[:, None, :],
            positions=None,
            body_ids=None,
            env_ids=None,
            is_global=True,
        )
        self._sim_time_s += self._physics_dt_s

    def reset(self, env_ids=None) -> None:
        self._raw_actions.zero_()
        self._processed_actions.zero_()
        self._applied_force_world.zero_()
        self._applied_torque_world.zero_()
        self._pending_request = None
        self._telemetry = None
        self._substep_telemetry.clear()
        self._substep_positions_world_m.clear()
        self._sim_time_s = 0.0
        self._last_request_result = "none"
        self.decoder.reset()
        self.controller.reset()
        self.capsule.permanent_wrench_composer.reset(env_ids=env_ids)

    def _read_state(self) -> CapsuleState:
        com_pose = self.capsule.data.root_com_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
        link_pose = self.capsule.data.root_link_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
        velocity = self.capsule.data.root_com_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
        # COM position/velocity are the wrench feedback variables.  Capsule
        # posture, however, is defined by the authored rigid-body link frame:
        # the PhysX COM frame may include a principal-inertia rotation and is
        # therefore not guaranteed to share the capsule geometry's local -Z.
        quaternion_xyzw = link_pose[3:7]
        quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
        return CapsuleState(
            sim_time_s=self._sim_time_s,
            position_world_m=com_pose[:3],
            quaternion_wxyz=quaternion_wxyz,
            linear_velocity_world_m_s=velocity[:3],
            angular_velocity_world_rad_s=velocity[3:6],
        )

    def _verify_dynamic_invariants(self, env) -> None:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        prim_path = self.capsule.root_view.prim_paths[0]
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        if not prim.IsValid() or not rigid_api:
            raise RuntimeError(f"capsule dynamic rigid body is unavailable: {prim_path}")
        if rigid_api.GetKinematicEnabledAttr() and bool(rigid_api.GetKinematicEnabledAttr().Get()):
            raise RuntimeError("TASK-004 capsule must be non-kinematic")
        physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if physx_api.GetDisableGravityAttr() and bool(physx_api.GetDisableGravityAttr().Get()):
            raise RuntimeError("TASK-004 capsule gravity must remain enabled")
        # Reuse TASK-003's mandatory task-local CCD authoring. This is not a
        # calibration change: every dynamic capsule task requires the same
        # safety invariant before the first physics step.
        ccd_attr = physx_api.GetEnableCCDAttr()
        if not ccd_attr:
            ccd_attr = physx_api.CreateEnableCCDAttr()
        ccd_attr.Set(True)
        if not bool(ccd_attr.Get()):
            raise RuntimeError("TASK-004 capsule body CCD must be enabled")
        if not bool(getattr(env.cfg.sim.physics, "enable_ccd", False)):
            raise RuntimeError("TASK-004 scene CCD must be enabled")


@configclass
class LocalPrimitiveActionTermCfg(ActionTermCfg):
    """Configuration of the isolated four-command primitive action."""

    class_type: type[ActionTerm] = LocalPrimitiveAction
    asset_name: str = "capsule"
    controller_cfg_values: dict = asdict(make_local_primitive_controller_cfg())
    profile_sha256: str = simulation_profile_sha256()

    @property
    def controller_cfg(self) -> LocalPrimitiveControllerCfg:
        """Return an immutable controller config from configclass-safe values."""

        return LocalPrimitiveControllerCfg(**self.controller_cfg_values)


def make_local_primitive_action_cfg() -> LocalPrimitiveActionTermCfg:
    """Build the one action term used unchanged by flat and stomach tasks."""

    return LocalPrimitiveActionTermCfg(
        controller_cfg_values=asdict(make_local_primitive_controller_cfg()),
        profile_sha256=simulation_profile_sha256(),
    )
