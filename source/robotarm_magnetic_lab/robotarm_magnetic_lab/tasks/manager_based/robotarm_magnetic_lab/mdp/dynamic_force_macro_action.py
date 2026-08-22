"""240 Hz TASK-008 endpoint-force macro action term."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from ..controllers.dynamic_force_macro import (
    DynamicForceMacroActionId,
    DynamicForceMacroConfig,
    camera_sphere_centers_local,
    equivalent_com_wrench,
    lateral_direction_world,
    phase_for_substep,
    point_forces_for_action,
)


@dataclass(frozen=True)
class MacroTelemetry:
    action_id: int
    substep: int
    phase: str
    force_active: bool
    applied_force_world: tuple[float, float, float]
    applied_torque_world: tuple[float, float, float]
    com_world: tuple[float, float, float]
    camera_center_world: tuple[float, float, float]
    other_center_world: tuple[float, float, float]
    camera_axis_world: tuple[float, float, float]
    lateral_direction_world: tuple[float, float, float]
    force_application_position_world: tuple[float, float, float] | None


class DynamicForceMacroAction(ActionTerm):
    """Apply one six-ID macro without runtime pose or velocity writes."""

    cfg: "DynamicForceMacroActionTermCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("TASK-008 supports exactly one environment")
        self.capsule = env.scene[cfg.asset_name]
        self.camera_sensor = env.scene[cfg.camera_sensor_name]
        self.config = DynamicForceMacroConfig(
            move_force_ratio=cfg.move_force_ratio,
            view_force_ratio=cfg.view_force_ratio,
            up_force_ratio=cfg.up_force_ratio,
            capsule_radius_m=cfg.capsule_radius_m,
            cylinder_height_m=cfg.cylinder_height_m,
        )
        self._raw_actions = torch.zeros((1, 1), device=env.device)
        self._processed_actions = torch.zeros((1, 1), device=env.device)
        self.camera_center_local, self.other_center_local = camera_sphere_centers_local(
            np.asarray(self.camera_sensor.cfg.offset.pos, dtype=np.float64),
            self.config.cylinder_height_m,
        )
        self.lifecycle = "idle"
        self.current_action: DynamicForceMacroActionId | None = None
        self.substep = 0
        self.trace: list[MacroTelemetry] = []
        self.last_telemetry: MacroTelemetry | None = None
        self._verify_dynamic_body_and_enable_ccd()
        mass = self.capsule.data.body_mass.torch.detach()
        self.mass_kg = float(mass.reshape(-1)[0].item())
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise RuntimeError("live capsule mass must be finite and positive")

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def trace_digest(self) -> str:
        payload = [item.__dict__ for item in self.trace]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        value = float(actions.reshape(-1)[0].item())
        rounded = round(value)
        if not np.isfinite(value) or abs(value - rounded) > 1.0e-6 or not 0 <= rounded <= 5:
            raise ValueError("TASK-008 action must be one integral scalar ID in [0, 5]")
        self._processed_actions[:] = float(rounded)
        if self.lifecycle == "idle":
            self.current_action = DynamicForceMacroActionId(rounded)
            self.substep = 0
            self.trace.clear()
            self.lifecycle = "running"

    def _geometry(self):
        link_pos = self.capsule.data.root_link_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        link_quat = self.capsule.data.root_link_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
        com = self.capsule.data.root_com_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        # Isaac Lab rigid-body state is wxyz; SciPy explicitly consumes xyzw.
        rotation = Rotation.from_quat(link_quat[[1, 2, 3, 0]]).as_matrix()
        camera = link_pos + rotation @ self.camera_center_local
        other = link_pos + rotation @ self.other_center_local
        axis = (camera - other) / np.linalg.norm(camera - other)
        try:
            lateral = lateral_direction_world(axis)
        except ValueError:
            # A vertical long axis has no world-horizontal cross-product.
            # Preserve a deterministic camera-image direction from local +X
            # for MOVE/VIEW actions at that singular orientation.
            lateral = rotation[:, 0] - np.dot(rotation[:, 0], axis) * axis
            lateral /= np.linalg.norm(lateral)
        return com, camera, other, axis, lateral

    def apply_actions(self) -> None:
        if self.lifecycle != "running" or self.current_action is None:
            return
        phase = phase_for_substep(self.current_action, self.substep)
        com, camera, other, axis, lateral = self._geometry()
        commanded_lateral = (
            -lateral
            if self.current_action in (DynamicForceMacroActionId.MOVE_NEG, DynamicForceMacroActionId.VIEW_NEG)
            else lateral
        )
        points = ()
        if phase.force_active:
            points = point_forces_for_action(
                self.current_action,
                mass_kg=self.mass_kg,
                lateral_direction_world=lateral,
                camera_center_world=camera,
                other_center_world=other,
                config=self.config,
            )
        force, effective_torque = equivalent_com_wrench(points, com)
        submitted_torque = effective_torque
        application_position = None
        if self.current_action == DynamicForceMacroActionId.UP and points:
            # Submit one force at the resolved camera-side hemisphere center.
            # Do not submit a controller torque; PhysX obtains the natural
            # moment solely from the off-CoM force application position.
            force = np.asarray(points[0].force_world, dtype=np.float64)
            submitted_torque = np.zeros(3, dtype=np.float64)
            application_position = np.asarray(points[0].position_world, dtype=np.float64)
        force_tensor = torch.as_tensor(force, device=self._env.device, dtype=torch.float32).reshape(1, 1, 3)
        torque_tensor = None
        if self.current_action != DynamicForceMacroActionId.UP:
            torque_tensor = torch.as_tensor(
                submitted_torque, device=self._env.device, dtype=torch.float32
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
        item = MacroTelemetry(
            int(self.current_action), self.substep, phase.name, phase.force_active,
            tuple(force), tuple(submitted_torque), tuple(com), tuple(camera), tuple(other), tuple(axis),
            tuple(commanded_lateral),
            None if application_position is None else tuple(application_position),
        )
        self.trace.append(item)
        self.last_telemetry = item
        self.substep += 1
        if self.substep == self.config.action_substeps:
            self.lifecycle = "boundary_ready"

    def release_after_boundary_capture(self) -> None:
        if self.lifecycle != "boundary_ready":
            raise RuntimeError("macro boundary is not ready")
        self.capsule.permanent_wrench_composer.reset()
        self.lifecycle = "idle"

    def reset(self, env_ids=None) -> None:
        self._raw_actions.zero_()
        self._processed_actions.zero_()
        self.lifecycle = "idle"
        self.current_action = None
        self.substep = 0
        self.trace.clear()
        self.last_telemetry = None
        self.capsule.permanent_wrench_composer.reset(env_ids=env_ids)

    def _verify_dynamic_body_and_enable_ccd(self) -> None:
        """Preserve the TASK-003 dynamic/gravity contract and task-local body CCD."""
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        prim_path = self.capsule.root_view.prim_paths[0]
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"capsule rigid-body prim is unavailable: {prim_path}")
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        if not rigid_api:
            raise RuntimeError(f"capsule has no UsdPhysics.RigidBodyAPI: {prim_path}")
        kinematic_attr = rigid_api.GetKinematicEnabledAttr()
        if kinematic_attr and bool(kinematic_attr.Get()):
            raise RuntimeError("TASK-008 capsule must be non-kinematic")
        physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_api:
            physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        gravity_attr = physx_api.GetDisableGravityAttr()
        if gravity_attr and bool(gravity_attr.Get()):
            raise RuntimeError("TASK-008 capsule gravity must remain enabled")
        ccd_attr = physx_api.GetEnableCCDAttr()
        if not ccd_attr:
            ccd_attr = physx_api.CreateEnableCCDAttr()
        ccd_attr.Set(True)
        if not bool(ccd_attr.Get()):
            raise RuntimeError("failed to enable TASK-008 capsule body CCD")


@configclass
class DynamicForceMacroActionTermCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DynamicForceMacroAction
    asset_name: str = "capsule"
    camera_sensor_name: str = "capsule_camera"
    move_force_ratio: float = 0.9
    view_force_ratio: float = 0.9
    up_force_ratio: float = 0.9
    capsule_radius_m: float = 0.0065
    cylinder_height_m: float = 0.012

    def __post_init__(self) -> None:
        DynamicForceMacroConfig(
            move_force_ratio=self.move_force_ratio,
            view_force_ratio=self.view_force_ratio,
            up_force_ratio=self.up_force_ratio,
            capsule_radius_m=self.capsule_radius_m,
            cylinder_height_m=self.cylinder_height_m,
        )
