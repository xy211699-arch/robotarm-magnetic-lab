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


class DynamicForceMacroAction(ActionTerm):
    """Apply one six-ID macro without runtime pose or velocity writes."""

    cfg: "DynamicForceMacroActionTermCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("TASK-008 supports exactly one environment")
        self.capsule = env.scene[cfg.asset_name]
        self.config = DynamicForceMacroConfig(
            move_force_ratio=cfg.move_force_ratio,
            view_force_ratio=cfg.view_force_ratio,
            up_force_ratio=cfg.up_force_ratio,
            capsule_radius_m=cfg.capsule_radius_m,
            cylinder_height_m=cfg.cylinder_height_m,
        )
        self._raw_actions = torch.zeros((1, 1), device=env.device)
        self._processed_actions = torch.zeros((1, 1), device=env.device)
        self.lifecycle = "idle"
        self.current_action: DynamicForceMacroActionId | None = None
        self.substep = 0
        self.trace: list[MacroTelemetry] = []
        self.last_telemetry: MacroTelemetry | None = None
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
        rotation = Rotation.from_quat(link_quat).as_matrix()
        half = 0.5 * self.config.cylinder_height_m
        camera_local = np.array([0.0, 0.0, -half])
        other_local = np.array([0.0, 0.0, half])
        camera = link_pos + rotation @ camera_local
        other = link_pos + rotation @ other_local
        axis = (camera - com) / np.linalg.norm(camera - com)
        lateral = lateral_direction_world(axis)
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
        force, torque = equivalent_com_wrench(points, com)
        force_tensor = torch.as_tensor(force, device=self._env.device, dtype=torch.float32).reshape(1, 1, 3)
        torque_tensor = torch.as_tensor(torque, device=self._env.device, dtype=torch.float32).reshape(1, 1, 3)
        self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force_tensor,
            torques=torque_tensor,
            positions=None,
            body_ids=None,
            env_ids=None,
            is_global=True,
        )
        item = MacroTelemetry(
            int(self.current_action), self.substep, phase.name, phase.force_active,
            tuple(force), tuple(torque), tuple(com), tuple(camera), tuple(other), tuple(axis), tuple(commanded_lateral),
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


@configclass
class DynamicForceMacroActionTermCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DynamicForceMacroAction
    asset_name: str = "capsule"
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
