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
        self.config_hash = str(config_hash)
        self.config_snapshot = config_snapshot
        self.dependency_audit_hash = str(dependency_audit_hash)
        self.seed = int(seed)
        self.current_update = 0
        self.total_transitions = 0
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
            "all_finite": _finite_metrics(diagnostics),
            **diagnostics,
        }
        _append_jsonl(self.metrics_path, payload)

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
            self.actor.detach_hidden_state()
            self.current_update += 1
            self.total_transitions += rollout_steps * num_envs
            self._record_update(diagnostics, time.perf_counter() - start)
            if save_interval and self.current_update % int(save_interval) == 0:
                self.save(self.output_dir / "checkpoints" / f"update_{self.current_update:04d}.pt")

    def learn_environment(self, env, *, num_updates: int, rollout_steps: int = 64, save_interval: int | None = None) -> None:
        observations, _ = env.reset(seed=self.seed)
        reset_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(int(num_updates)):
            start = time.perf_counter()
            storage = Task010RolloutStorage(rollout_steps, env.num_envs, self.device)
            for _step in range(rollout_steps):
                actor_obs = observations["policy"]
                critic_obs = observations["privileged"]
                transition = self.algorithm.act(actor_obs, critic_obs)
                next_observations, reward, terminated, truncated, extras = env.step(transition["action"])
                done = terminated | truncated
                storage.add(
                    actor_observation=actor_obs, critic_observation=critic_obs,
                    action=transition["action"], reward=reward,
                    terminated=terminated, reset_mask=reset_mask,
                    value=transition["value"].detach(), log_prob=transition["log_prob"].detach(),
                    distribution_parameters=transition["distribution_parameters"], hidden_state=transition["hidden_state"],
                )
                self.actor.reset(done)
                reset_mask = done
                observations = next_observations
            with torch.no_grad():
                last_value = self.critic(observations["privileged"]).squeeze(1)
            storage.compute_returns(last_value, gamma=0.999, lam=0.95, sampler_interrupted=True)
            diagnostics = self.algorithm.update(storage)
            self.actor.detach_hidden_state()
            self.current_update += 1
            self.total_transitions += rollout_steps * env.num_envs
            self._record_update(diagnostics, time.perf_counter() - start)
            if save_interval and self.current_update % int(save_interval) == 0:
                self.save(self.output_dir / "checkpoints" / f"update_{self.current_update:04d}.pt")
