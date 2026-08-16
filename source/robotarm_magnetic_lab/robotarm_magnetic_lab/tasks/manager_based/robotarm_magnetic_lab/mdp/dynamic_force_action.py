"""Real-dynamics world-frame force ActionTerm for TASK-003."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from ..controllers.dynamic_force import GRAVITY_M_S2, validate_force_weight_ratio


class DynamicForceAction(ActionTerm):
    """Apply a persistent force at the dynamic capsule center of mass."""

    cfg: "DynamicForceActionTermCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("DynamicForceAction requires exactly one environment")
        self.capsule = env.scene[cfg.asset_name]
        self._force_weight_ratio = validate_force_weight_ratio(cfg.force_weight_ratio)
        self._vertical_force_weight_ratio = validate_force_weight_ratio(
            cfg.vertical_force_weight_ratio
        )
        self._raw_actions = torch.zeros((env.num_envs, 3), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._applied_force_world = torch.zeros_like(self._raw_actions)
        self._applied_torque_world = torch.zeros_like(self._raw_actions)
        self._substep_positions_world: deque[np.ndarray] = deque(maxlen=8192)

        masses = self.capsule.data.body_mass.torch.detach()
        if tuple(masses.shape) != (1, 1):
            raise RuntimeError(f"unexpected capsule mass tensor shape: {tuple(masses.shape)}")
        self._mass_kg = float(masses[0, 0].item())
        if not np.isfinite(self._mass_kg) or self._mass_kg <= 0.0:
            raise RuntimeError("live capsule mass must be finite and positive")
        self._force_scale_world = torch.tensor(
            [
                self._force_weight_ratio,
                self._force_weight_ratio,
                self._vertical_force_weight_ratio,
            ],
            device=self._processed_actions.device,
            dtype=self._processed_actions.dtype,
        ) * (self._mass_kg * GRAVITY_M_S2)
        self._verify_dynamic_body_and_enable_ccd()

    @property
    def action_dim(self) -> int:
        return 3

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
    def mass_kg(self) -> float:
        return self._mass_kg

    @property
    def force_weight_ratio(self) -> float:
        return self._force_weight_ratio

    @property
    def vertical_force_weight_ratio(self) -> float:
        return self._vertical_force_weight_ratio

    @property
    def substep_positions_world(self) -> tuple[np.ndarray, ...]:
        return tuple(value.copy() for value in self._substep_positions_world)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        clipped = torch.clamp(actions, -1.0, 1.0)
        norm = torch.linalg.vector_norm(clipped, dim=-1, keepdim=True)
        self._processed_actions[:] = clipped / torch.clamp(norm, min=1.0)

    def apply_actions(self) -> None:
        self._applied_force_world[:] = self._processed_actions * self._force_scale_world
        self._applied_torque_world.zero_()
        self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._applied_force_world[:, None, :],
            torques=self._applied_torque_world[:, None, :],
            positions=None,
            body_ids=None,
            env_ids=None,
            is_global=True,
        )
        position = (
            self.capsule.data.root_com_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        )
        self._substep_positions_world.append(position)

    def reset(self, env_ids=None) -> None:
        self._raw_actions.zero_()
        self._processed_actions.zero_()
        self._applied_force_world.zero_()
        self._applied_torque_world.zero_()
        self._substep_positions_world.clear()
        self.capsule.permanent_wrench_composer.reset(env_ids=env_ids)

    def _verify_dynamic_body_and_enable_ccd(self) -> None:
        """Reject kinematic/gravity-disabled state and author task-local body CCD."""
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
            raise RuntimeError("TASK-003 capsule must be non-kinematic")
        physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_api:
            physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        gravity_attr = physx_api.GetDisableGravityAttr()
        if gravity_attr and bool(gravity_attr.Get()):
            raise RuntimeError("TASK-003 capsule gravity must remain enabled")
        ccd_attr = physx_api.GetEnableCCDAttr()
        if not ccd_attr:
            ccd_attr = physx_api.CreateEnableCCDAttr()
        ccd_attr.Set(True)
        if not bool(ccd_attr.Get()):
            raise RuntimeError("failed to enable capsule body CCD")


@configclass
class DynamicForceActionTermCfg(ActionTermCfg):
    """Configuration of the isolated three-dimensional force action."""

    class_type: type[ActionTerm] = DynamicForceAction
    asset_name: str = "capsule"
    force_weight_ratio: float = 0.9
    vertical_force_weight_ratio: float = 1.1

    def __post_init__(self) -> None:
        validate_force_weight_ratio(self.force_weight_ratio)
        validate_force_weight_ratio(self.vertical_force_weight_ratio)
