"""Physics-rate analytical magnetic-wrench action hook."""

from __future__ import annotations

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass


class MagneticPhysicsAction(ActionTerm):
    """Refresh passive magnetic forces without adding policy dimensions."""

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
            term_cfg = self._env.event_manager.get_term_cfg(
                "magnetic_collision_bridge"
            )
            self._bridge = term_cfg.func
        self._bridge.physics_step(self._env)


@configclass
class MagneticPhysicsActionCfg(ActionTermCfg):
    """Configuration for the zero-dimensional magnetic physics hook."""

    class_type: type[ActionTerm] = MagneticPhysicsAction
