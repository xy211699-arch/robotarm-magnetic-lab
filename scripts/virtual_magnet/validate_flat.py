#!/usr/bin/env python3
"""Run deterministic TASK-007 flat-scene validation and save compact evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from common import file_evidence, generate_manifest, manifest_digest, run_live_trial, summarize_trials, write_json


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("no-disturbance",), default="no-disturbance")
parser.add_argument("--trials-per-action", type=int, default=20)
parser.add_argument("--held-out", action="store_true")
parser.add_argument("--seed", type=int, default=7007)
parser.add_argument("--action_ids", type=int, nargs="+", default=list(range(11)))
parser.add_argument("--task", default="Template-Robotarm-Magnetic-Virtual-Magnet-Flat-Lab-v0")
parser.add_argument("--output", type=Path, default=ROOT / "handoffs/reports/TASK-007-flat-no-disturbance-summary.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import robotarm_magnetic_lab.tasks  # noqa: F401,E402
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    split = "held-out-flat-no-disturbance" if args_cli.held_out else "development-flat-no-disturbance"
    valid = generate_manifest(
        split,
        args_cli.trials_per_action,
        base_seed=args_cli.seed,
        action_ids=tuple(args_cli.action_ids),
    )
    move_ids = tuple(item for item in args_cli.action_ids if item in (9, 10))
    invalid = (
        generate_manifest(
            split + "-invalid",
            args_cli.trials_per_action,
            base_seed=args_cli.seed + 50000,
            action_ids=move_ids,
            valid_move=False,
        )
        if move_ids
        else []
    )
    specs = valid + invalid
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    cfg.scene.capsule_camera = None
    rows = []
    with launch_simulation(cfg, args_cli):
        env = gym.make(args_cli.task, cfg=cfg)
        for index, spec in enumerate(specs, 1):
            row = run_live_trial(env, spec)
            rows.append(row)
            print(
                f"VIRTUAL_MAGNET_TRIAL {index}/{len(specs)} class={row['class']} "
                f"result={row['result']} pass={row['pass']}",
                flush=True,
            )
        env.close()
    aggregate = summarize_trials(rows)
    payload = {
        "schema_version": "task007_flat_no_disturbance_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args_cli.mode,
        "split": split,
        "manifest_sha256": manifest_digest(specs),
        "manifest": [asdict(item) for item in specs],
        "aggregate": aggregate,
        "rows": rows,
    }
    write_json(args_cli.output, payload)
    print("VIRTUAL_MAGNET_SUMMARY " + json.dumps({**aggregate, "evidence": file_evidence(args_cli.output)}, sort_keys=True), flush=True)
    if not aggregate["all_classes_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
