"""TASK-010 synchronous runner, checkpoint and authoritative JSONL logging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

from .task010_actor import Task010Actor
from .task010_critic import Task010Critic
from .task010_ppo import Task010PPO, Task010RolloutStorage
from robotarm_magnetic_lab.runtime.task010_visual_encoder import RESNET18_IMAGENET1K_V1_SHA256


ACTOR_OBSERVATION_SCHEMA_SHA256 = hashlib.sha256(b"visual_feature_512|previous_actual_action_7").hexdigest()
ACTION_SCHEMA_SHA256 = hashlib.sha256(b"mode_id_0_5|alpha_0_1|hold_alpha_zero").hexdigest()
TASK010_TASK_ID = "Template-Robotarm-Magnetic-Task010-CNN-GRU-Coverage-Lab-v0"


def _finite_metrics(value: dict[str, Any]) -> bool:
    return all(not isinstance(item, float) or np.isfinite(item) for item in value.values())


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        import os

        os.fsync(stream.fileno())


class Task010OnPolicyRunner:
    def __init__(
        self,
        actor: Task010Actor,
        critic: Task010Critic,
        *,
        output_dir: Path,
        config_hash: str,
        config_snapshot: dict[str, Any],
        dependency_audit_hash: str,
        seed: int,
        device: str | torch.device,
        ppo_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.algorithm = Task010PPO(self.actor, self.critic, **(ppo_kwargs or {}))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.events_path = self.output_dir / "events.jsonl"
        self.boundaries_path = self.output_dir / "boundaries.jsonl"
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.config_hash = str(config_hash)
        self.config_snapshot = config_snapshot
        self.dependency_audit_hash = str(dependency_audit_hash)
        self.seed = int(seed)
        self.current_update = 0
        self.total_transitions = 0
        self.init_at_random_ep_len = False
        self.visual_weight_identity_sha256 = RESNET18_IMAGENET1K_V1_SHA256
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        root = Path(__file__).resolve().parents[4]
        self.git_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()
        _append_jsonl(self.events_path, {"event": "runner_initialized", "seed": self.seed, "time_ns": time.time_ns()})

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].detach().cpu())
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.detach().cpu() for item in state["torch_cuda"]])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "robotarm_magnetic_lab.task010_checkpoint",
            "task_id": TASK010_TASK_ID,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.algorithm.optimizer.state_dict(),
            "learning_rate": float(self.algorithm.optimizer.param_groups[0]["lr"]),
            "current_update": self.current_update,
            "total_transitions": self.total_transitions,
            "rng": self._rng_state(),
            "config_hash": self.config_hash,
            "config_snapshot": self.config_snapshot,
            "git_commit": self.git_commit,
            "dependency_audit_hash": self.dependency_audit_hash,
            "visual_weight_identity_sha256": self.visual_weight_identity_sha256,
            "actor_observation_schema_sha256": ACTOR_OBSERVATION_SCHEMA_SHA256,
            "action_schema_sha256": ACTION_SCHEMA_SHA256,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            import os

            os.fsync(stream.fileno())
        temporary.replace(path)
        _append_jsonl(self.events_path, {"event": "checkpoint_saved", "path": str(path), "update": self.current_update, "time_ns": time.time_ns()})

    def load(self, path: Path, *, strict: bool = True) -> None:
        record = torch.load(Path(path), map_location=self.device, weights_only=False)
        checks = {
            "task ID": (record.get("task_id"), TASK010_TASK_ID),
            "config hash": (record.get("config_hash"), self.config_hash),
            "dependency hash": (record.get("dependency_audit_hash"), self.dependency_audit_hash),
            "visual weight identity": (
                record.get("visual_weight_identity_sha256"), self.visual_weight_identity_sha256
            ),
            "Actor observation schema": (record.get("actor_observation_schema_sha256"), ACTOR_OBSERVATION_SCHEMA_SHA256),
            "action schema": (record.get("action_schema_sha256"), ACTION_SCHEMA_SHA256),
        }
        if strict:
            for name, (actual, expected) in checks.items():
                if actual != expected:
                    raise ValueError(f"TASK-010 {name} mismatch")
        self.actor.load_state_dict(record["actor"], strict=True)
        self.critic.load_state_dict(record["critic"], strict=True)
        self.algorithm.optimizer.load_state_dict(record["optimizer"])
        self.current_update = int(record["current_update"])
        self.total_transitions = int(record["total_transitions"])
        self.actor._hidden_state = None
        self._restore_rng(record["rng"])
        _append_jsonl(self.events_path, {"event": "checkpoint_loaded", "path": str(path), "update": self.current_update, "time_ns": time.time_ns()})

    def _record_update(self, diagnostics: dict[str, Any], elapsed_s: float) -> None:
        payload = {
            "schema": "robotarm_magnetic_lab.task010_update_metric",
            "update": self.current_update,
            "total_transitions": self.total_transitions,
            "elapsed_s": float(elapsed_s),
            "time_ns": time.time_ns(),
            "transitions_per_second": float(
                diagnostics.pop("transitions_in_update", 0.0) / max(elapsed_s, 1.0e-12)
            ),
            "cuda_peak_memory_bytes": int(
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda" else 0
            ),
            "all_finite": _finite_metrics(diagnostics),
            **diagnostics,
        }
        _append_jsonl(self.metrics_path, payload)

    @staticmethod
    def _tensor_list(value: torch.Tensor) -> list:
        return value.detach().cpu().tolist()

    def _record_boundary(
        self, env, action: torch.Tensor, reward: torch.Tensor, extras: dict[str, Any]
    ) -> None:
        step = env._task010_last_reward_step
        terminal = extras.get("task009d0_terminal_audit")
        latest = env._task009d0_coverage_runtime.latest_update
        if terminal is None and latest is None:
            raise RuntimeError("TASK-010 boundary coverage update is unavailable")
        frame_ids = terminal["frame_ids"] if terminal is not None else latest.frame_ids
        reachable = terminal["reachable_coverage"] if terminal is not None else latest.reachable.coverage_fraction
        raw = terminal["raw_coverage"] if terminal is not None else latest.raw.coverage_fraction
        payload = {
            "schema": "robotarm_magnetic_lab.task010_boundary",
            "time_ns": time.time_ns(),
            "formal_step": int(env._formal_step),
            "physics_substeps": 24,
            "frame_ids": self._tensor_list(frame_ids),
            "action_mode": self._tensor_list(action[:, 0].to(torch.int64)),
            "action_alpha": self._tensor_list(action[:, 1]),
            "reachable_coverage": self._tensor_list(reachable),
            "raw_coverage": self._tensor_list(raw),
            "recovery_phase": self._tensor_list(torch.argmax(step.phase_one_hot_4, dim=1)),
            "reward_total": self._tensor_list(reward),
            "reward_coverage": self._tensor_list(step.coverage_reward),
            "reward_escape_progress": self._tensor_list(step.escape_reward),
            "reward_no_progress": self._tensor_list(step.no_progress_reward),
            "reward_coverage_resumed": self._tensor_list(step.coverage_resumed_reward),
            "no_progress": self._tensor_list(step.no_progress),
            "coverage_resumed": self._tensor_list(step.coverage_resumed),
        }
        _append_jsonl(self.boundaries_path, payload)

    def _new_episode_accumulator(self, env, reset_extras: dict[str, Any]) -> dict[str, Any]:
        initial = reset_extras["task009d0_reset"]["initial_coverage"]
        return {
            "pose_ids": list(reset_extras["task009d0_reset"]["pose_ids"]),
            "c0": torch.tensor(initial, device=self.device, dtype=torch.float64),
            "coverage_sum": torch.tensor(initial, device=self.device, dtype=torch.float64),
            "reward_sum": torch.zeros(env.num_envs, device=self.device, dtype=torch.float64),
            "mode_counts": torch.zeros((env.num_envs, 6), device=self.device, dtype=torch.int64),
            "alpha_sum": torch.zeros((env.num_envs, 6), device=self.device, dtype=torch.float64),
            "alpha_count": torch.zeros((env.num_envs, 6), device=self.device, dtype=torch.int64),
            "stagnation_count": torch.zeros(env.num_envs, device=self.device, dtype=torch.int64),
            "recovery_success_count": torch.zeros(env.num_envs, device=self.device, dtype=torch.int64),
            "steps": 0,
        }

    def _update_episode_accumulator(
        self, accumulator: dict[str, Any], env, action: torch.Tensor, reward: torch.Tensor,
        extras: dict[str, Any],
    ) -> None:
        terminal = extras.get("task009d0_terminal_audit")
        latest = env._task009d0_coverage_runtime.latest_update
        if terminal is None and latest is None:
            raise RuntimeError("TASK-010 episode coverage update is unavailable")
        coverage = terminal["reachable_coverage"] if terminal is not None else latest.reachable.coverage_fraction
        modes = action[:, 0].to(torch.int64)
        accumulator["coverage_sum"] += coverage.to(torch.float64)
        accumulator["reward_sum"] += reward.to(torch.float64)
        accumulator["mode_counts"].scatter_add_(
            1, modes[:, None], torch.ones_like(modes[:, None])
        )
        accumulator["alpha_sum"].scatter_add_(1, modes[:, None], action[:, 1:2].to(torch.float64))
        accumulator["alpha_count"].scatter_add_(
            1, modes[:, None], (modes != 0).to(torch.int64)[:, None]
        )
        step = env._task010_last_reward_step
        accumulator["stagnation_count"] += step.no_progress.to(torch.int64)
        accumulator["recovery_success_count"] += step.coverage_resumed.to(torch.int64)
        accumulator["steps"] += 1

    def _record_episodes(self, accumulator: dict[str, Any], c120: torch.Tensor) -> None:
        steps = int(accumulator["steps"])
        for row, pose_id in enumerate(accumulator["pose_ids"]):
            counts = accumulator["mode_counts"][row]
            alpha_counts = accumulator["alpha_count"][row]
            alpha_mean = accumulator["alpha_sum"][row] / alpha_counts.clamp_min(1)
            _append_jsonl(
                self.episodes_path,
                {
                    "schema": "robotarm_magnetic_lab.task010_episode",
                    "time_ns": time.time_ns(),
                    "pose_id": pose_id,
                    "steps": steps,
                    "c0": float(accumulator["c0"][row].item()),
                    "c120": float(c120[row].item()),
                    "nauc120": float((accumulator["coverage_sum"][row] / (steps + 1)).item()),
                    "total_reward": float(accumulator["reward_sum"][row].item()),
                    "action_proportions": (counts.to(torch.float64) / max(steps, 1)).cpu().tolist(),
                    "alpha_mean_by_mode": alpha_mean.cpu().tolist(),
                    "stagnation_count": int(accumulator["stagnation_count"][row].item()),
                    "recovery_success_count": int(accumulator["recovery_success_count"][row].item()),
                    "fault_type": None,
                },
            )

    def learn_fake(self, *, num_updates: int, rollout_steps: int, num_envs: int, save_interval: int | None = None) -> None:
        """CPU contract backend; exercises real GRU/PPO without simulation claims."""
        reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        for _ in range(int(num_updates)):
            start = time.perf_counter()
            storage = Task010RolloutStorage(rollout_steps, num_envs, self.device)
            for _step in range(rollout_steps):
                actor_obs = torch.randn((num_envs, 519), device=self.device)
                critic_obs = torch.randn((num_envs, 65), device=self.device)
                transition = self.algorithm.act(actor_obs, critic_obs)
                action = transition["action"]
                reward = 0.01 * action[:, 1] + 0.001 * torch.randn(num_envs, device=self.device)
                storage.add(
                    actor_observation=actor_obs, critic_observation=critic_obs,
                    action=action, reward=reward,
                    terminated=torch.zeros(num_envs, dtype=torch.bool, device=self.device),
                    reset_mask=reset_mask, value=transition["value"].detach(),
                    log_prob=transition["log_prob"].detach(),
                    distribution_parameters=transition["distribution_parameters"],
                    hidden_state=transition["hidden_state"],
                )
            with torch.no_grad():
                last_value = self.critic(torch.randn((num_envs, 65), device=self.device)).squeeze(1)
            storage.compute_returns(last_value, gamma=0.999, lam=0.95, sampler_interrupted=True)
            diagnostics = self.algorithm.update(storage)
            diagnostics["transitions_in_update"] = float(rollout_steps * num_envs)
            self.actor.detach_hidden_state()
            self.current_update += 1
            self.total_transitions += rollout_steps * num_envs
            self._record_update(diagnostics, time.perf_counter() - start)
            if save_interval and self.current_update % int(save_interval) == 0:
                self.save(self.output_dir / "checkpoints" / f"update_{self.current_update:04d}.pt")

    def learn_environment(self, env, *, num_updates: int, rollout_steps: int = 64, save_interval: int | None = None) -> None:
        observations, reset_extras = env.reset(seed=self.seed)
        episode = self._new_episode_accumulator(env, reset_extras)
        reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(int(num_updates)):
            start = time.perf_counter()
            storage = Task010RolloutStorage(rollout_steps, env.num_envs, self.device)
            for _step in range(rollout_steps):
                actor_obs = observations["policy"]
                critic_obs = observations["privileged"]
                transition = self.algorithm.act(actor_obs, critic_obs)
                next_observations, reward, terminated, truncated, extras = env.step(transition["action"])
                self._record_boundary(env, transition["action"], reward, extras)
                self._update_episode_accumulator(episode, env, transition["action"], reward, extras)
                done = terminated | truncated
                storage.add(
                    actor_observation=actor_obs, critic_observation=critic_obs,
                    action=transition["action"], reward=reward,
                    terminated=terminated, reset_mask=reset_mask,
                    value=transition["value"].detach(), log_prob=transition["log_prob"].detach(),
                    distribution_parameters=transition["distribution_parameters"], hidden_state=transition["hidden_state"],
                )
                self.actor.reset(done)
                if done.any().item():
                    terminal = extras["task009d0_terminal_audit"]
                    self._record_episodes(episode, terminal["reachable_coverage"])
                    episode = self._new_episode_accumulator(env, extras)
                reset_mask = done
                observations = next_observations
            with torch.no_grad():
                last_value = self.critic(observations["privileged"]).squeeze(1)
            storage.compute_returns(last_value, gamma=0.999, lam=0.95, sampler_interrupted=True)
            diagnostics = self.algorithm.update(storage)
            diagnostics["transitions_in_update"] = float(rollout_steps * env.num_envs)
            self.actor.detach_hidden_state()
            self.current_update += 1
            self.total_transitions += rollout_steps * env.num_envs
            self._record_update(diagnostics, time.perf_counter() - start)
            if save_interval and self.current_update % int(save_interval) == 0:
                self.save(self.output_dir / "checkpoints" / f"update_{self.current_update:04d}.pt")
