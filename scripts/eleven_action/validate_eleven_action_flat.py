#!/usr/bin/env python3
"""按固定随机种子执行 TASK-005 平面单动作定量验收。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
TASK_ID = "Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0"


def evaluate_samples(samples: list[dict]) -> dict:
    summary = {"status": "pass", "fault_count": 0, "actions": {}, "failures": []}
    for action_id in range(11):
        rows = [row for row in samples if int(row["action_id"]) == action_id]
        item: dict = {"total": len(rows)}
        summary["actions"][str(action_id)] = item
        faults = [row for row in rows if row.get("fault") or row.get("result") == "fault"]
        summary["fault_count"] += len(faults)
        if action_id == 0:
            valid = [row for row in rows if row.get("category") == "valid"]
            item["valid_count"] = len(valid)
            passed = len(valid) >= 10 and all(
                row["result"] == "completed" and row["substeps"] == 240
                and float(row["angle_delta_deg"]) <= 3.0
                and float(row["max_support_drift_m"]) <= 0.002
                for row in valid
            )
        elif 1 <= action_id <= 8:
            unblocked = [row for row in rows if row.get("category") == "valid" and not row.get("constrained")]
            blocked = [row for row in rows if row.get("constrained")]
            item.update(unblocked_count=len(unblocked), blocked_count=len(blocked))
            passed = len(unblocked) >= 10 and all(
                row["result"] == "completed" and row["substeps"] == 240
                and abs(float(row["angle_delta_deg"]) - 15.0) <= 3.0
                and float(row["max_support_drift_m"]) <= 0.002
                for row in unblocked
            )
        else:
            valid = [row for row in rows if row.get("category") == "valid"]
            invalid_angle = [row for row in rows if row.get("category") == "invalid_angle"]
            invalid_contact = [row for row in rows if row.get("category") == "invalid_contact"]
            success_rate = (
                sum(float(row["move_signed_displacement_m"]) >= 0.005 for row in valid) / len(valid)
                if valid else 0.0
            )
            item.update(
                valid_count=len(valid), invalid_angle_count=len(invalid_angle),
                invalid_contact_count=len(invalid_contact), valid_success_rate=success_rate,
            )
            invalid = invalid_angle + invalid_contact
            passed = (
                len(valid) >= 10 and len(invalid_angle) >= 5 and len(invalid_contact) >= 5
                and success_rate >= 0.9
                and all(row["result"] == "completed" and row["substeps"] == 240 for row in valid)
                and all(row["result"] == "rejected" and row["substeps"] == 240 for row in invalid)
            )
        item["passed"] = bool(passed and not faults)
        if not item["passed"]:
            summary["failures"].append(f"action_{action_id}_gate_failed")
    if summary["fault_count"] or summary["failures"]:
        summary["status"] = "fail"
    return summary


def main() -> int:
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--render_fps", type=int, choices=(60, 120, 240), default=120)
    parser.add_argument("--output_directory", type=Path, default=ROOT / "logs/eleven_action_flat_validation")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    import gymnasium as gym
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
        dynamic_profile_sha256,
        load_dynamic_profile,
    )
    from calibrate_eleven_action import reset_flat_trial, run_one_action

    rng = np.random.default_rng(args.seed)
    profile = load_dynamic_profile()
    output = args.output_directory / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    samples: list[dict] = []
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.sim.render_interval = 240 // args.render_fps
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            for action_id in range(1, 9):
                print(f"ELEVEN_ACTION_FLAT_VIEW_BEGIN action_id={action_id}", flush=True)
                accepted = 0
                attempts = 0
                while accepted < 10 and attempts < 30:
                    attempts += 1
                    term = reset_flat_trial(
                        env, profile,
                        tilt_deg=float(rng.uniform(0.0, 90.0)),
                        azimuth_deg=float(rng.uniform(0.0, 360.0)),
                        roll_deg=float(rng.uniform(0.0, 360.0)),
                    )
                    row = run_one_action(env, term, action_id)
                    row.update(category="valid", fault=row["result"] == "fault")
                    samples.append(row)
                    if not row["constrained"]:
                        accepted += 1
                print(
                    f"ELEVEN_ACTION_FLAT_VIEW_END action_id={action_id} "
                    f"unblocked={accepted} attempts={attempts}",
                    flush=True,
                )
            print("ELEVEN_ACTION_FLAT_HOLD_BEGIN", flush=True)
            for _ in range(10):
                term = reset_flat_trial(
                    env, profile,
                    tilt_deg=float(rng.uniform(0.0, 90.0)),
                    azimuth_deg=float(rng.uniform(0.0, 360.0)),
                    roll_deg=float(rng.uniform(0.0, 360.0)),
                )
                row = run_one_action(env, term, 0)
                row.update(category="valid", fault=row["result"] == "fault")
                samples.append(row)
            print("ELEVEN_ACTION_FLAT_HOLD_END trials=10", flush=True)
            for action_id in (9, 10):
                print(f"ELEVEN_ACTION_FLAT_MOVE_BEGIN action_id={action_id}", flush=True)
                for trial_index in range(10):
                    term = reset_flat_trial(
                        env, profile,
                        tilt_deg=(60.0, 75.0, 90.0)[trial_index % 3],
                        azimuth_deg=float(rng.uniform(0.0, 360.0)),
                        roll_deg=float(rng.uniform(0.0, 360.0)),
                    )
                    row = run_one_action(env, term, action_id)
                    row.update(category="valid", fault=row["result"] == "fault")
                    samples.append(row)
                for category in ("invalid_angle", "invalid_contact"):
                    for _ in range(5):
                        term = reset_flat_trial(
                            env, profile,
                            tilt_deg=float(rng.uniform(0.0, 45.0)) if category == "invalid_angle" else 90.0,
                            azimuth_deg=float(rng.uniform(0.0, 360.0)),
                            roll_deg=float(rng.uniform(0.0, 360.0)),
                            elevated=category == "invalid_contact",
                            settle_steps=12 if category == "invalid_angle" else 0,
                        )
                        row = run_one_action(env, term, action_id)
                        row.update(category=category, fault=row["result"] == "fault")
                        samples.append(row)
                print(f"ELEVEN_ACTION_FLAT_MOVE_END action_id={action_id} trials=20", flush=True)
        finally:
            env.close()
    summary = evaluate_samples(samples)
    with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in samples:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary.update(seed=args.seed, profile_sha256=dynamic_profile_sha256(), sample_count=len(samples))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    label = "ELEVEN_ACTION_FLAT_ACCEPTANCE_PASS" if summary["status"] == "pass" else "ELEVEN_ACTION_FLAT_ACCEPTANCE_FAIL"
    print(f"{label} path={output / 'summary.json'}", flush=True)
    simulation_app.close()
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
