"""Isaac Lab action adapters for the TASK-007 virtual-magnet controller."""

from __future__ import annotations

import math

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass


class VirtualMagnetRequestAction(ActionTerm):
    """Expose exactly one scalar public action ID in the closed interval 0..10."""

    cfg: "VirtualMagnetRequestActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("TASK-007 supports exactly one environment")
        self._raw_actions = torch.zeros((env.num_envs, 1), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._pending_action_id: int | None = None
        self._bridge = None

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
    def bridge(self):
        return self._bridge

    def _get_bridge(self):
        if self._bridge is None:
            cfg = self._env.event_manager.get_term_cfg("virtual_magnet_bridge")
            self._bridge = cfg.func
        return self._bridge

    @staticmethod
    def validate_action_id(value: float) -> int:
        """Return a public ID; reject fractions, -1, NaN/Inf, and out-of-range values."""
        if not math.isfinite(float(value)):
            raise ValueError("virtual-magnet action ID must be finite")
        rounded = int(round(float(value)))
        if abs(float(value) - rounded) > 1.0e-6 or not 0 <= rounded <= 10:
            raise ValueError(f"virtual-magnet public action ID must be an integer in [0, 10], got {value}")
        return rounded

    def process_actions(self, actions: torch.Tensor) -> None:
        if tuple(actions.shape) != tuple(self._raw_actions.shape):
            raise ValueError(f"expected action shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}")
        self._raw_actions[:] = actions
        action_id = self.validate_action_id(float(actions[0, 0].item()))
        self._processed_actions.fill_(float(action_id))
        self._pending_action_id = action_id

    def apply_actions(self) -> None:
        if self._pending_action_id is None:
            return
        # submit() deliberately discards requests received while EXECUTING. No
        # pending queue is retained after this one attempt.
        self._get_bridge().submit(self._pending_action_id)
        self._pending_action_id = None

    def reset(self, env_ids=None) -> None:
        self._pending_action_id = None
        self._raw_actions.zero_()
        self._processed_actions.zero_()


class VirtualMagnetPhysicsAction(ActionTerm):
    """Zero-dimensional hook that applies the finite-model wrench at 240 Hz."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((env.num_envs, 0), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._bridge = None

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions
        self._processed_actions = actions

    def apply_actions(self) -> None:
        if self._bridge is None:
            cfg = self._env.event_manager.get_term_cfg("virtual_magnet_bridge")
            self._bridge = cfg.func
        self._bridge.physics_step()


@configclass
class VirtualMagnetRequestActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = VirtualMagnetRequestAction


@configclass
class VirtualMagnetPhysicsActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = VirtualMagnetPhysicsAction

