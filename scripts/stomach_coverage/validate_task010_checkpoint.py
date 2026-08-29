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
    if records_path.exists():
        records_path.unlink()

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
                terminal = None
                with torch.inference_mode():
                    for _step in range(1200):
                        action = actor(observations["policy"], stochastic_output=False)
                        observations, reward, terminated, truncated, step_extras = env.step(action)
                        actor_totals += reward[: len(batch)]
                        alpha_totals += action[: len(batch), 1]
                        mode_counts.scatter_add_(1, action[: len(batch), :1].long(), torch.ones((len(batch), 1), device=args.device))
                        if torch.any(terminated[: len(batch)]).item():
                            terminal = step_extras["task009d0_terminal_audit"]
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
        finally:
            env.close()
    result = summarize(records_path, checkpoint_sha256=checkpoint_hash, config_sha256=frozen.config_sha256)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    app.close()


if __name__ == "__main__":
    main()
