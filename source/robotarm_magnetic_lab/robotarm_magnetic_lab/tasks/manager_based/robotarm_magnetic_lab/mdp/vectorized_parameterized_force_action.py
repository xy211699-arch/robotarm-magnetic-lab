"""Vectorized Isaac Lab ActionTerm for TASK-009D0 parameterized forces."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from ..controllers.dynamic_force_macro import camera_sphere_centers_local
from ..controllers.parameterized_force import ParameterizedForceConfig
from ..controllers.vectorized_parameterized_force import (
    batched_equivalent_com_wrench,
    batched_parameterized_endpoint_forces,
)


def _torch(value) -> torch.Tensor:
    return getattr(value, "torch", value)


class VectorizedParameterizedForceAction(ActionTerm):
    """Apply one ``[mode, alpha]`` command per environment for all 24 substeps."""

    cfg: "VectorizedParameterizedForceActionTermCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.capsule = env.scene[cfg.asset_name]
        camera_sensor = env.scene[cfg.camera_sensor_name]
        camera_center, other_center = camera_sphere_centers_local(
            np.asarray(camera_sensor.cfg.offset.pos, dtype=np.float64),
            cfg.cylinder_height_m,
        )
        self.camera_center_local = torch.as_tensor(
            camera_center, device=env.device, dtype=torch.float32
        )
        self.other_center_local = torch.as_tensor(
            other_center, device=env.device, dtype=torch.float32
        )
        self.config = ParameterizedForceConfig(
            move_min_ratio=cfg.move_min_ratio,
            move_max_ratio=cfg.move_max_ratio,
            view_min_ratio=cfg.view_min_ratio,
            view_max_ratio=cfg.view_max_ratio,
            up_min_ratio=cfg.up_min_ratio,
            up_max_ratio=cfg.up_max_ratio,
        )
        masses = _torch(self.capsule.data.body_mass).reshape(env.num_envs, -1)[:, 0]
        if torch.any(~torch.isfinite(masses) | (masses <= 0)).item():
            raise RuntimeError("live capsule masses must be finite and positive")
        self.mass_kg = masses.to(device=env.device, dtype=torch.float32).clone()
        self._raw_actions = torch.zeros((env.num_envs, 2), device=env.device)
        self._processed_actions = torch.zeros((env.num_envs, 2), device=env.device)
        self._modes = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self._alpha = torch.zeros(env.num_envs, device=env.device)
        self._previous_action_features = torch.zeros(
            (env.num_envs, 7), device=env.device
        )
        self._all_env_ids = torch.arange(
            env.num_envs, device=env.device, dtype=torch.int64
        )
        self.last_camera_positions_world = torch.zeros(
            (env.num_envs, 3), device=env.device
        )
        self.last_other_positions_world = torch.zeros_like(
            self.last_camera_positions_world
        )
        self.last_directions_world = torch.zeros_like(self.last_camera_positions_world)
        self.last_resultant_forces_world = torch.zeros_like(
            self.last_camera_positions_world
        )
        self.last_camera_forces_world = torch.zeros_like(
            self.last_camera_positions_world
        )
        self.last_other_forces_world = torch.zeros_like(
            self.last_camera_positions_world
        )
        self.last_resultant_torques_world = torch.zeros_like(
            self.last_camera_positions_world
        )
        self._verify_dynamic_bodies_and_enable_ccd()

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def previous_action_features(self) -> torch.Tensor:
        return self._previous_action_features

    def process_actions(self, actions: torch.Tensor) -> None:
        expected = (self._env.num_envs, 2)
        if tuple(actions.shape) != expected:
            raise ValueError(f"vectorized parameterized action must have shape {expected}")
        values = actions.to(device=self._raw_actions.device, dtype=self._raw_actions.dtype)
        if torch.any(~torch.isfinite(values)).item():
            raise ValueError("vectorized parameterized actions must be finite")
        rounded = torch.round(values[:, 0])
        if torch.any(torch.abs(values[:, 0] - rounded) > 1.0e-6).item():
            raise ValueError("mode IDs must be integral")
        modes = rounded.to(torch.int64)
        alpha = values[:, 1]
        if torch.any((modes < 0) | (modes > 5)).item():
            raise ValueError("mode IDs must be in [0, 5]")
        if torch.any((alpha < 0) | (alpha > 1)).item():
            raise ValueError("alpha must be in [0, 1]")
        canonical_alpha = torch.where(modes == 0, torch.zeros_like(alpha), alpha)
        self._raw_actions.copy_(values)
        self._processed_actions[:, 0].copy_(modes.to(self._processed_actions.dtype))
        self._processed_actions[:, 1].copy_(canonical_alpha)
        self._modes.copy_(modes)
        self._alpha.copy_(canonical_alpha)
        self._previous_action_features[:, :6].copy_(
            functional.one_hot(modes, num_classes=6).to(
                dtype=self._previous_action_features.dtype
            )
        )
        self._previous_action_features[:, 6].copy_(canonical_alpha)
        # Remove the previous boundary's persistent wrench before the first
        # substep of this command. Active rows are written again in apply_actions.
        self.capsule.permanent_wrench_composer.reset(env_ids=self._all_env_ids)

    def apply_actions(self) -> None:
        link_pos = _torch(self.capsule.data.root_link_pos_w)
        link_quat = _torch(self.capsule.data.root_link_quat_w)
        com = _torch(self.capsule.data.root_com_pos_w)
        camera_local = self.camera_center_local.expand(self._env.num_envs, -1)
        other_local = self.other_center_local.expand(self._env.num_envs, -1)
        camera = link_pos + math_utils.quat_apply(link_quat, camera_local)
        other = link_pos + math_utils.quat_apply(link_quat, other_local)
        axes = camera - other
        command = batched_parameterized_endpoint_forces(
            self._modes,
            self._alpha,
            self.mass_kg,
            axes,
            self.config,
        )
        resultant, torque = batched_equivalent_com_wrench(
            command.camera_forces_world,
            command.other_forces_world,
            camera,
            other,
            com,
        )
        self.last_camera_positions_world.copy_(camera)
        self.last_other_positions_world.copy_(other)
        self.last_directions_world.copy_(command.directions_world)
        self.last_camera_forces_world.copy_(command.camera_forces_world)
        self.last_other_forces_world.copy_(command.other_forces_world)
        self.last_resultant_forces_world.copy_(resultant)
        self.last_resultant_torques_world.copy_(torque)
        com_rows = torch.nonzero(
            (self._modes >= 1) & (self._modes <= 4), as_tuple=False
        ).reshape(-1)
        if com_rows.numel():
            self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
                forces=resultant[com_rows, None, :],
                torques=torque[com_rows, None, :],
                positions=None,
                body_ids=None,
                env_ids=com_rows,
                is_global=True,
            )
        up_rows = torch.nonzero(self._modes == 5, as_tuple=False).reshape(-1)
        if up_rows.numel():
            # Keep UP as a true point force at the camera-side hemisphere.
            self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
                forces=command.camera_forces_world[up_rows, None, :],
                torques=None,
                positions=camera[up_rows, None, :],
                body_ids=None,
                env_ids=up_rows,
                is_global=True,
            )

    def reset(self, env_ids=None) -> None:
        rows = (
            self._all_env_ids
            if env_ids is None
            else torch.as_tensor(
                env_ids, device=self._raw_actions.device, dtype=torch.int64
            ).reshape(-1)
        )
        self._raw_actions[rows] = 0.0
        self._processed_actions[rows] = 0.0
        self._modes[rows] = 0
        self._alpha[rows] = 0.0
        self._previous_action_features[rows] = 0.0
        self.last_camera_positions_world[rows] = 0.0
        self.last_other_positions_world[rows] = 0.0
        self.last_directions_world[rows] = 0.0
        self.last_resultant_forces_world[rows] = 0.0
        self.last_camera_forces_world[rows] = 0.0
        self.last_other_forces_world[rows] = 0.0
        self.last_resultant_torques_world[rows] = 0.0
        self.capsule.permanent_wrench_composer.reset(env_ids=rows)

    def _verify_dynamic_bodies_and_enable_ccd(self) -> None:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        for prim_path in self.capsule.root_view.prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if not rigid_api or (
                rigid_api.GetKinematicEnabledAttr()
                and bool(rigid_api.GetKinematicEnabledAttr().Get())
            ):
                raise RuntimeError(
                    f"TASK-009D0 capsule must be non-kinematic: {prim_path}"
                )
            physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not physx_api:
                physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            if physx_api.GetDisableGravityAttr() and bool(
                physx_api.GetDisableGravityAttr().Get()
            ):
                raise RuntimeError(
                    f"TASK-009D0 capsule gravity must remain enabled: {prim_path}"
                )
            ccd = physx_api.GetEnableCCDAttr() or physx_api.CreateEnableCCDAttr()
            ccd.Set(True)


@configclass
class VectorizedParameterizedForceActionTermCfg(ActionTermCfg):
    class_type: type[ActionTerm] = VectorizedParameterizedForceAction
    asset_name: str = "capsule"
    camera_sensor_name: str = "capsule_camera"
    cylinder_height_m: float = 0.012
    move_min_ratio: float = 0.70
    move_max_ratio: float = 1.40
    view_min_ratio: float = 0.20
    view_max_ratio: float = 0.50
    up_min_ratio: float = 0.80
    up_max_ratio: float = 1.05
