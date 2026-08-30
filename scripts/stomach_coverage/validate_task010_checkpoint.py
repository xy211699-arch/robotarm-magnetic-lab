#!/usr/bin/env python3
"""Run deterministic TASK-010 validation on the frozen twenty validation poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_task010_validation import VALIDATION_POSE_IDS, file_sha256, summarize, validation_batches


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> None:
    from isaaclab.app import AppLauncher

    command = parser()
    AppLauncher.add_app_launcher_args(command)
    args = command.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
    from robotarm_magnetic_lab.runtime.task010_config import load_task010_config

    frozen = load_task010_config(args.config)
    record = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if record.get("config_hash") != frozen.config_sha256:
        raise RuntimeError("TASK-010 validation checkpoint/config mismatch")
    checkpoint_hash = file_sha256(args.checkpoint)
    actor = Task010Actor().to(args.device)
    actor.load_state_dict(record["actor"], strict=True)
    actor.eval()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "pose_records.jsonl"
    trajectory_path = output / "coverage_trajectories.jsonl"
    for stale in (records_path, trajectory_path):
        if stale.exists():
            stale.unlink()

    cfg = parse_env_cfg(frozen.task_id, device=args.device, num_envs=12)
    cfg.pose_split = "validation"
    cfg.explicit_pose_ids = tuple(validation_batches()[0])
    with launch_simulation(app, cfg):
        env = gym.make(frozen.task_id, cfg=cfg).unwrapped
        try:
            for batch_index, batch in enumerate(validation_batches()):
                padded = tuple(batch) + tuple(batch[:1]) * (12 - len(batch))
                env.cfg.explicit_pose_ids = padded
                observations, extras = env.reset(seed=frozen.training.seed + batch_index)
                actual = tuple(extras["task009d0_reset"]["pose_ids"])
                if actual[: len(batch)] != tuple(batch):
                    raise RuntimeError("TASK-010 validation reset returned different pose IDs")
                actor.reset()
                actor_totals = torch.zeros((len(batch),), device=args.device)
                alpha_totals = torch.zeros_like(actor_totals)
                mode_counts = torch.zeros((len(batch), 6), device=args.device)
                coverage_trajectories = torch.empty((len(batch), 1201), device=args.device, dtype=torch.float64)
                coverage_trajectories[:, 0] = torch.as_tensor(
                    extras["task009d0_reset"]["initial_coverage"][: len(batch)],
                    device=args.device,
                    dtype=torch.float64,
                )
                terminal = None
                for step_index in range(1200):
                    # Keep only the Actor forward pass gradient-free.  Wrapping
                    # env.step() in inference_mode makes mutable coverage state
                    # into inference tensors, which cannot be reset in-place
                    # before the second (8-pose) validation batch.
                    with torch.no_grad():
                        action = actor(observations["policy"], stochastic_output=False)
                    observations, reward, terminated, truncated, step_extras = env.step(action)
                    actor_totals += reward[: len(batch)]
                    alpha_totals += action[: len(batch), 1]
                    mode_counts.scatter_add_(1, action[: len(batch), :1].long(), torch.ones((len(batch), 1), device=args.device))
                    terminal_audit = step_extras.get("task009d0_terminal_audit")
                    latest = env._task009d0_coverage_runtime.latest_update
                    current_coverage = (
                        terminal_audit["reachable_coverage"]
                        if terminal_audit is not None
                        else latest.reachable.coverage_fraction
                    )
                    coverage_trajectories[:, step_index + 1] = current_coverage[: len(batch)]
                    if torch.any(terminated[: len(batch)]).item():
                        terminal = terminal_audit
                if terminal is None:
                    raise RuntimeError("TASK-010 validation did not reach true terminal")
                for row, pose_id in enumerate(batch):
                    payload = {
                        "schema": "robotarm_magnetic_lab.task010_validation_pose",
                        "pose_id": pose_id,
                        "formal_steps": 1200,
                        "checkpoint_sha256": checkpoint_hash,
                        "config_sha256": frozen.config_sha256,
                        "final_coverage": float(terminal["reachable_coverage"][row].item()),
                        "total_reward": float(actor_totals[row].item()),
                        "mean_alpha": float(alpha_totals[row].item() / 1200.0),
                        "mode_fraction": (mode_counts[row] / 1200.0).cpu().tolist(),
                        "failure_reason": None,
                    }
                    with records_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(payload, sort_keys=True) + "\n")
                    trajectory = {
                        "schema": "robotarm_magnetic_lab.task010_validation_coverage_trajectory",
                        "pose_id": pose_id,
                        "checkpoint_sha256": checkpoint_hash,
                        "config_sha256": frozen.config_sha256,
                        "control_hz": 10,
                        "coverage_fraction": coverage_trajectories[row].cpu().tolist(),
                    }
                    with trajectory_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(trajectory, sort_keys=True) + "\n")
        finally:
            env.close()
    result = summarize(records_path, checkpoint_sha256=checkpoint_hash, config_sha256=frozen.config_sha256)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    app.close()


if __name__ == "__main__":
    main()
