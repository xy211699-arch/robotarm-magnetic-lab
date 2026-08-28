"""Explicit synchronous lifecycle for the TASK-009D0 vector environment."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from robotarm_magnetic_lab.runtime.task009d0_config import (
    TASK009D0_CONFIG_PATH,
    load_task009d0_config,
)
from robotarm_magnetic_lab.runtime.task009d0_pose_batch import (
    Task009D0PoseBatchSampler,
)

from .controllers.parameterized_force import ParameterizedForceMode


FORMAL_STEPS = 1200
RESET_HOLD_CYCLES = 10


def _clone_observation(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_observation(item) for key, item in value.items()}
    return deepcopy(value)


class Task009D0VectorEnv(ManagerBasedRLEnv):
    """Run all rows synchronously and reset only at the 1200-step horizon."""

    def __init__(self, cfg, render_mode=None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        config = load_task009d0_config(TASK009D0_CONFIG_PATH)
        self._task009d0_config = config
        self._task009d0_training_seed = int(config["training_seed"])
        self._episode_indices = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._formal_step = 0
        self._task009d0_stabilizing = False
        self._pose_sampler = self._new_pose_sampler(self._task009d0_training_seed)
        self._last_pose_batch = None

    def _new_pose_sampler(self, training_seed: int) -> Task009D0PoseBatchSampler:
        return Task009D0PoseBatchSampler.from_config(
            self._task009d0_config,
            authorized_split=self.cfg.pose_split,
            training_seed=int(training_seed),
            repository_root=TASK009D0_CONFIG_PATH.parents[2],
        )

    def _sample_pose_batch(self):
        env_ids = np.arange(self.num_envs, dtype=np.int64)
        if self.cfg.pose_split == "train":
            return self._pose_sampler.sample(
                env_ids,
                self._episode_indices.detach().cpu().numpy().astype(np.int64),
            )
        pose_ids = self.cfg.explicit_pose_ids
        if pose_ids is None or len(pose_ids) != self.num_envs:
            raise ValueError(
                "validation/test TASK-009D0 launch requires one versioned pose ID per row"
            )
        return self._pose_sampler.resolve_explicit(env_ids, pose_ids)

    def reset(self, seed=None, env_ids=None, options=None):
        if env_ids is not None and len(env_ids) != self.num_envs:
            raise ValueError("formal task009d0 reset must include all environments")
        if seed is not None:
            self._task009d0_training_seed = int(seed)
            self._episode_indices.zero_()
            self._pose_sampler = self._new_pose_sampler(int(seed))
        return self._reset_all_with_hold(seed=seed, options=options)

    def _reset_all_with_hold(self, *, seed=None, options=None):
        all_rows = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)
        runtime = getattr(self, "_task009d0_coverage_runtime", None)
        if runtime is not None:
            runtime.reset_rows(all_rows)
        observation, extras = super().reset(seed=seed, env_ids=all_rows, options=options)
        pose_batch = self._sample_pose_batch()
        poses = torch.as_tensor(
            pose_batch.poses_world_xyzw, device=self.device, dtype=torch.float32
        )
        # Frozen library coordinates were authored with the single reference
        # environment at world origin, so they are environment-local poses.
        # Isaac Lab may place env_0 away from world origin when a vector batch
        # is centered; add every row's absolute origin (including row zero).
        origins = self.scene.env_origins.to(device=self.device, dtype=torch.float32)
        poses[:, :3] += origins
        capsule = self.scene["capsule"]
        action_term = self.action_manager.get_term("parameterized_force")
        action_term.reset(all_rows)
        capsule.permanent_wrench_composer.reset(env_ids=all_rows)
        capsule.write_root_pose_to_sim_index(root_pose=poses)
        capsule.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((self.num_envs, 6), device=self.device)
        )
        self.sim.forward()
        self.scene.update(0.0)
        restored = capsule.data.root_pose_w.torch
        position_error = torch.linalg.vector_norm(restored[:, :3] - poses[:, :3], dim=1)
        alignment = torch.abs((restored[:, 3:] * poses[:, 3:]).sum(dim=1))
        if torch.any(position_error > 1.0e-5).item() or torch.any(
            alignment < 1.0 - 1.0e-5
        ).item():
            raise RuntimeError(
                "TASK-009D0 vector pose write verification failed: "
                f"max_position_error={position_error.max().item()}, "
                f"min_quaternion_alignment={alignment.min().item()}"
            )
        hold = torch.zeros((self.num_envs, 2), device=self.device)
        hold[:, 0] = float(ParameterizedForceMode.HOLD)
        self._task009d0_stabilizing = True
        hold_trace = []
        try:
            for cycle in range(RESET_HOLD_CYCLES):
                observation, _, terminated, truncated, extras = super().step(hold)
                if torch.any(terminated).item() or torch.any(truncated).item():
                    raise RuntimeError(
                        "TASK-009D0 reset HOLD stabilization unexpectedly terminated"
                    )
                sync = self._task009d0_coverage_runtime.rgb_sync
                hold_trace.append(
                    {
                        "cycle": cycle,
                        "frame_ids": sync.latest.detach().cpu().tolist(),
                        "physics_substeps": int(self.cfg.decimation),
                    }
                )
        finally:
            self._task009d0_stabilizing = False
        self.episode_length_buf.zero_()
        self._formal_step = 0
        action_term.reset(all_rows)
        capsule.permanent_wrench_composer.reset(env_ids=all_rows)
        runtime = self._task009d0_coverage_runtime
        runtime.reset_rows(all_rows)
        initial = runtime.capture_initial(boundary=int(self.common_step_counter))
        if torch.any(initial.reachable.coverage_fraction <= 0).item():
            raise RuntimeError("TASK-009D0 reset produced non-positive initial C0")
        self._episode_indices += 1
        self._last_pose_batch = pose_batch
        self.obs_buf = self.observation_manager.compute(update_history=True)
        extras["task009d0_reset"] = {
            "pose_ids": list(pose_batch.pose_ids),
            "episode_indices": pose_batch.episode_indices.tolist(),
            "hold_cycles": hold_trace,
            "initial_coverage": initial.reachable.coverage_fraction.detach().cpu().tolist(),
        }
        return self.obs_buf, extras

    def step(self, action):
        observation, reward, terminated, truncated, extras = super().step(action)
        self._formal_step += 1
        if torch.any(terminated).item() or torch.any(truncated).item():
            raise RuntimeError("base environment terminated before synchronous horizon")
        if self._formal_step > FORMAL_STEPS:
            raise RuntimeError("TASK-009D0 formal horizon counter exceeded")
        if self._formal_step == FORMAL_STEPS:
            runtime = self._task009d0_coverage_runtime
            latest = runtime.latest_update
            if latest is None:
                raise RuntimeError("TASK-009D0 terminal coverage snapshot is unavailable")
            capsule = self.scene["capsule"]
            action_term = self.action_manager.get_term("parameterized_force")
            terminal_audit = {
                "formal_step": int(self._formal_step),
                "episode_length": self.episode_length_buf.detach().clone(),
                "frame_ids": latest.frame_ids.detach().clone(),
                "reachable_coverage": latest.reachable.coverage_fraction.detach().clone(),
                "raw_coverage": latest.raw.coverage_fraction.detach().clone(),
                "reachable_masks": runtime.reachable_accumulator.mask.detach().clone(),
                "raw_masks": runtime.raw_accumulator.mask.detach().clone(),
                "root_pose": capsule.data.root_pose_w.torch.detach().clone(),
                "root_velocity": capsule.data.root_com_vel_w.torch.detach().clone(),
                "previous_action": action_term.previous_action_features.detach().clone(),
            }
            terminal = _clone_observation(observation)
            observation, reset_extras = self._reset_all_with_hold(
                seed=None, options=None
            )
            extras.update(reset_extras)
            extras["terminal_observation"] = terminal
            extras["task009d0_terminal_audit"] = terminal_audit
            truncated = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        return observation, reward, terminated, truncated, extras

    def reset_rows_for_test(self, env_ids: torch.Tensor) -> None:
        """Isolation-only helper; formal training always resets the full batch."""
        rows = env_ids.to(device=self.device, dtype=torch.int64).reshape(-1)
        self.action_manager.get_term("parameterized_force").reset(rows)
        self._task009d0_coverage_runtime.reset_rows(rows)
        self._episode_indices[rows] = 0
