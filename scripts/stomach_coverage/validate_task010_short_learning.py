#!/usr/bin/env python3
"""TASK-010 Gate 3: eight real GPU PPO updates with strict resume at update four."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
SCHEMA = "robotarm_magnetic_lab.task010_short_learning"


def validate_summary(value: dict) -> dict:
    required = {"schema": SCHEMA, "seed": 991010, "num_envs": 12, "updates_completed": 8, "all_finite": True, "resume_verified": True}
    for name, expected in required.items():
        if value.get(name) != expected: raise ValueError(f"TASK-010 Gate 3 {name} mismatch")
    if float(value.get("actor_parameter_delta_l2", 0.0)) <= 0: raise ValueError("TASK-010 Gate 3 Actor parameters did not change")
    if float(value.get("critic_parameter_delta_l2", 0.0)) <= 0: raise ValueError("TASK-010 Gate 3 Critic parameters did not change")
    return value


def _state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.named_parameters()}


def _delta(before, module):
    total = 0.0
    for name, tensor in module.named_parameters(): total += float((tensor.detach().cpu() - before[name]).double().square().sum().item())
    return total ** 0.5


def _sha(path: Path):
    digest = hashlib.sha256(path.read_bytes()); return digest.hexdigest()


def _runner(config, config_path, output, seed, device, dependency_hash):
    from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
    from robotarm_magnetic_lab.learning.task010_critic import Task010Critic
    from robotarm_magnetic_lab.learning.task010_runner import Task010OnPolicyRunner
    snapshot = json.loads(Path(config_path).read_text(encoding="utf-8"))
    kwargs = {"num_learning_epochs": config.ppo.num_learning_epochs, "num_mini_batches": config.ppo.num_mini_batches,
              "clip_param": config.ppo.clip_param, "value_loss_coef": config.ppo.value_loss_coef,
              "learning_rate": config.ppo.learning_rate, "desired_kl": config.ppo.desired_kl,
              "max_grad_norm": config.ppo.max_grad_norm, "entropy_coef": config.ppo.entropy_coef}
    return Task010OnPolicyRunner(Task010Actor(), Task010Critic(), output_dir=output, config_hash=config.config_sha256,
                                 config_snapshot=snapshot, dependency_audit_hash=dependency_hash,
                                 seed=seed, device=device, ppo_kwargs=kwargs)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True); parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--split-update", type=int, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser); parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if (args.seed, args.num_envs, args.updates, args.split_update) != (991010, 12, 8, 4): parser.error("TASK-010 Gate 3 contract is fixed to seed 991010, 12 envs, 8 updates split at 4")
    args.enable_cameras = True; app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.runtime.task010_config import load_task010_config

    config = load_task010_config(args.config); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    audit = Path("/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate0/prerequisites.json")
    dependency_hash = _sha(audit)
    cfg = parse_env_cfg(config.task_id, device=args.device, num_envs=12)
    with launch_simulation(app, cfg):
        env = gym.make(config.task_id, cfg=cfg).unwrapped
        try:
            first = _runner(config, args.config, output, args.seed, args.device, dependency_hash)
            actor_before = _state(first.actor); critic_before = _state(first.critic)
            first.learn_environment(env, num_updates=4, rollout_steps=config.ppo.rollout_steps, save_interval=4)
            checkpoint4 = output / "checkpoints/update_0004.pt"
            second = _runner(config, args.config, output, args.seed, args.device, dependency_hash); second.load(checkpoint4)
            resume_verified = second.current_update == 4 and second.total_transitions == 4 * 64 * 12
            second.learn_environment(env, num_updates=4, rollout_steps=config.ppo.rollout_steps, save_interval=4)
            checkpoint8 = output / "checkpoints/update_0008.pt"
            metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
            finite = len(metrics) == 8 and all(row.get("all_finite") for row in metrics) and all(torch.isfinite(p).all().item() for p in list(second.actor.parameters()) + list(second.critic.parameters()))
            summary = validate_summary({"schema": SCHEMA, "seed": args.seed, "num_envs": 12,
                "updates_completed": second.current_update, "transitions": second.total_transitions,
                "actor_parameter_delta_l2": _delta(actor_before, second.actor), "critic_parameter_delta_l2": _delta(critic_before, second.critic),
                "all_finite": bool(finite), "resume_verified": bool(resume_verified),
                "checkpoint4": str(checkpoint4), "checkpoint4_sha256": _sha(checkpoint4),
                "checkpoint8": str(checkpoint8), "checkpoint8_sha256": _sha(checkpoint8),
                "metrics_records": len(metrics), "device": str(env.device)})
            (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("TASK010_GATE3", json.dumps(summary, sort_keys=True))
        finally: env.close()
    app.close()


if __name__ == "__main__": main()
