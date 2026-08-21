#!/usr/bin/env python3
"""Exhaustive TASK-008 flat calibration and one-shot held-out acceptance."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from common import coarse_candidates, evaluate_trace, make_manifest, manifest_sha256, write_json


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0")
parser.add_argument("--calibration_samples", type=int, default=20)
parser.add_argument("--held_out_samples", type=int, default=20)
parser.add_argument("--initial_ratio", type=float, default=0.9)
parser.add_argument("--growth", type=float, default=1.25)
parser.add_argument("--max_ratio", type=float, default=3.0)
parser.add_argument("--refinement_rounds", type=int, default=3)
parser.add_argument("--output_dir", type=Path, default=Path("/tmp/task008-dynamic-force-calibration"))
parser.add_argument("--headless", action="store_true", help="Compatibility flag for the current non-windowed launcher.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import robotarm_magnetic_lab.tasks  # noqa: E402,F401
import torch
from scipy.spatial.transform import Rotation
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg

from robotarm_magnetic_lab.runtime import SynchronousMacroRunner


GROUPS = {"move": (1, 2), "view": (3, 4), "up": (5,)}


def apply_reset(env, spec) -> None:
    env.reset(seed=spec.seed)
    base = env.unwrapped
    capsule = base.scene["capsule"]
    pose = capsule.data.root_pose_w.torch.clone()
    pose[0, 0] += spec.x_offset_m
    pose[0, 1] += spec.y_offset_m
    pose[0, 2] = max(float(pose[0, 2]), 0.0067)
    rotation = Rotation.from_euler("Z", spec.yaw_rad) * Rotation.from_euler("Y", np.pi / 2) * Rotation.from_euler("Z", spec.roll_rad)
    pose[0, 3:7] = torch.as_tensor(rotation.as_quat(), device=base.device, dtype=torch.float32)
    capsule.write_root_pose_to_sim_index(root_pose=pose)
    capsule.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))
    base.action_manager.get_term("dynamic_force_macro").reset()


def set_ratio(term, group: str, ratio: float) -> None:
    updates = {"move_force_ratio": ratio} if group == "move" else {"view_force_ratio": ratio} if group == "view" else {"up_force_ratio": ratio}
    term.config = replace(term.config, **updates)


def run_candidate(env, runner, specs, group: str, ratio: float) -> dict:
    rows = []
    term = env.unwrapped.action_manager.get_term("dynamic_force_macro")
    set_ratio(term, group, ratio)
    for spec in specs:
        apply_reset(env, spec)
        # Authorized reset-only settling outside the evaluated action.
        SynchronousMacroRunner(env).step(0)
        transition = runner.step(spec.action_id)
        finite = all(np.isfinite(np.asarray(item.com_world)).all() for item in term.trace)
        passed, metrics = evaluate_trace(spec.action_id, term.trace) if finite else (False, {"reason": "nonfinite"})
        rows.append({**asdict(spec), "ratio": ratio, "pass": passed, "fault": not finite, "metrics": metrics, "trace_digest": transition.trace_digest, "boundary_frame": transition.boundary_rgb_frame_id})
    counts = {action: sum(row["pass"] for row in rows if row["action_id"] == action) for action in GROUPS[group]}
    faults = sum(row["fault"] for row in rows)
    group_pass = faults == 0 and all(counts[action] >= 16 for action in GROUPS[group])
    return {"group": group, "ratio": ratio, "counts": counts, "faults": faults, "pass": group_pass, "rows": rows}


def search_group(env, runner, manifest, group: str) -> tuple[float | None, list[dict]]:
    specs = [row for row in manifest if row.action_id in GROUPS[group]]
    evidence = []
    lower_failure = None
    first_pass = None
    for ratio in coarse_candidates(args_cli.initial_ratio, args_cli.growth, args_cli.max_ratio):
        result = run_candidate(env, runner, specs, group, ratio)
        evidence.append(result)
        print(f"TASK008_CANDIDATE group={group} ratio={ratio:.9g} counts={result['counts']} faults={result['faults']} pass={result['pass']}", flush=True)
        if result["pass"]:
            first_pass = ratio
            break
        lower_failure = ratio
    if first_pass is None:
        return None, evidence
    if lower_failure is None:
        return first_pass, evidence
    lower, upper = lower_failure, first_pass
    for _ in range(args_cli.refinement_rounds):
        middle = 0.5 * (lower + upper)
        result = run_candidate(env, runner, specs, group, middle)
        evidence.append(result)
        print(f"TASK008_REFINEMENT group={group} ratio={middle:.9g} counts={result['counts']} faults={result['faults']} pass={result['pass']}", flush=True)
        if result["pass"]: upper = middle
        else: lower = middle
    return upper, evidence


def main() -> None:
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = make_manifest("calibration", args_cli.calibration_samples, 8008)
    held = make_manifest("held-out", args_cli.held_out_samples, 18008)
    write_json(args_cli.output_dir / "calibration_manifest.json", [asdict(row) for row in calibration])
    write_json(args_cli.output_dir / "held_out_manifest.json", [asdict(row) for row in held])
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    all_candidates = {}
    held_rows = []
    selected = {}
    with launch_simulation(cfg, args_cli):
        env = gym.make(args_cli.task, cfg=cfg)
        runner = SynchronousMacroRunner(env)
        for group in GROUPS:
            selected[group], all_candidates[group] = search_group(env, runner, calibration, group)
        if all(value is not None for value in selected.values()):
            for group, actions in GROUPS.items():
                result = run_candidate(env, runner, [row for row in held if row.action_id in actions], group, selected[group])
                held_rows.extend(result["rows"])
        env.close()
    profile = {"schema": "task008_force_profile_v1", "move_force_ratio": selected["move"], "view_force_ratio": selected["view"], "up_force_ratio": selected["up"]}
    profile_path = write_json(args_cli.output_dir / "selected_profile.json", profile)
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    summary = {"planning": "TASK-008", "calibration_manifest_sha256": manifest_sha256(calibration), "held_out_manifest_sha256": manifest_sha256(held), "selected": selected, "profile_sha256": profile_sha, "candidates": all_candidates, "held_out": held_rows}
    summary_path = write_json(args_cli.output_dir / "summary.json", summary)
    all_pass = bool(held_rows)
    for action in range(1, 6):
        rows = [row for row in held_rows if row["action_id"] == action]
        success, faults = sum(row["pass"] for row in rows), sum(row["fault"] for row in rows)
        passed = len(rows) == 20 and success >= 16 and faults == 0
        all_pass &= passed
        print(f"TASK008_HELD_OUT action={action} success={success}/20 faults={faults} status={'PASS' if passed else 'FAIL'}", flush=True)
    print(f"TASK008_SUMMARY path={summary_path.resolve()} bytes={summary_path.stat().st_size} sha256={hashlib.sha256(summary_path.read_bytes()).hexdigest()}")
    if all_pass: print("TASK008_TABLE_ACCEPTANCE_PASS")
    else: raise SystemExit(2)


if __name__ == "__main__":
    try: main()
    finally: simulation_app.close()
