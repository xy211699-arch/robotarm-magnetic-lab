"""TASK-010 lifecycle specialization over the accepted TASK-009D0 environment."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .task009d0_vector_env import Task009D0VectorEnv


def _assert_finite_tree(value, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"TASK-010 reset has non-finite {name}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{name}.{key}")


class Task010VectorEnv(Task009D0VectorEnv):
    """Accept raw-visible zero-reachable starts and use true task termination."""

    def _initial_reachable_coverage_is_valid(
        self, reachable: torch.Tensor, raw: torch.Tensor
    ) -> torch.Tensor:
        return torch.isfinite(reachable) & torch.isfinite(raw) & (raw > 0)

    def _validate_task_reset(self, initial) -> None:
        _assert_finite_tree(initial.reachable.coverage_fraction, "reachable coverage")
        _assert_finite_tree(initial.raw.coverage_fraction, "raw coverage")
        capsule = self.scene["capsule"]
        _assert_finite_tree(capsule.data.root_pose_w.torch, "capsule root pose")
        _assert_finite_tree(capsule.data.root_com_vel_w.torch, "capsule root velocity")
        robot = self.scene["robot"]
        _assert_finite_tree(robot.data.root_pose_w.torch, "robot root pose")
        _assert_finite_tree(robot.data.joint_pos, "robot joint position")
        _assert_finite_tree(robot.data.joint_vel, "robot joint velocity")
        camera = self.scene["capsule_camera"]
        _assert_finite_tree(camera.data.output, "RGB observation")
        encoder = getattr(self, "_task010_visual_encoder", None)
        if encoder is not None:
            encoder.reset(torch.arange(self.num_envs, device=self.device))

    def _horizon_termination_flags(self):
        terminated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, truncated, time_outs
