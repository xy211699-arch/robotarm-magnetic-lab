#!/usr/bin/env python3
"""Run TASK-009C reset validation, smoke, or formal random baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(ROOT / "scripts"))

from _artifact_paths import artifact_root
from isaaclab.app import AppLauncher


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--reset_only", action="store_true")
mode.add_argument("--smoke", action="store_true")
mode.add_argument("--formal", action="store_true")
parser.add_argument(
    "--config",
    type=Path,
    default=ROOT / "configs/task009c/random_baseline_preexperiment_v1.json",
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=artifact_root(ROOT) / "task009c_random_baseline_preexperiment",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, visualizer=[])
args_cli = parser.parse_args()
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.baselines.random_policies import load_random_baseline_config
from robotarm_magnetic_lab.coverage.entry_pose_library import file_sha256
from robotarm_magnetic_lab.coverage.simulator_runtime import P0CoverageRuntime
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.task009b_training_env import (
    RESET_HOLD_CYCLES,
    TASK009C_OPTION_KEY,
    _load_task009c_pose_records,
    _stable_rgb_digest,
)


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def _write_json(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _pose_request(record: dict, manifest_hash: str) -> dict:
    return {
        "pose_id": record["pose_id"],
        "split": record["split"],
        "pose_world_xyzw": record["pose_world_xyzw"],
        "pose_library_manifest_config_sha256": manifest_hash,
    }


def _assert_gpu(base) -> dict:
    camera_tensor = base.scene["capsule_camera"].data.output["rgb"].torch
    physics_view = base.sim.physics_sim_view
    devices = {
        "environment": str(base.device),
        "simulation": str(base.sim.device),
        "physics": str(getattr(physics_view, "device", getattr(physics_view, "_device", "unknown"))),
        "camera": str(camera_tensor.device),
    }
    if any(not value.startswith("cuda") for value in devices.values()):
        raise RuntimeError(f"TASK-009C requires GPU PhysX and camera tensors: {devices}")
    return devices


def _run_reset_only(env, output: Path, config: dict, manifest: dict, allowed: dict) -> dict:
    rows_path = output / "reset_only.jsonl"
    base = env.unwrapped
    devices = _assert_gpu(base)
    for pose_id in config["validation_pose_ids"]:
        record = allowed[pose_id]
        observation, extras = env.reset(
            seed=int(config["environment_seeds"][pose_id]),
            options={TASK009C_OPTION_KEY: _pose_request(record, manifest["config_sha256"])},
        )
        info = extras[TASK009C_OPTION_KEY]
        trace = info["hold_cycles"]
        frames = [int(item["actor_rgb_frame"]) for item in trace]
        expected_frames = list(range(frames[0], frames[0] + RESET_HOLD_CYCLES))
        if len(trace) != RESET_HOLD_CYCLES or frames != expected_frames:
            raise RuntimeError(f"{pose_id} reset did not produce ten consecutive HOLD frames")
        term_trace = base.action_manager.get_term("parameterized_force").current_cycle_trace
        if len(term_trace) != 24 or any(item.target_total_force_n != 0.0 for item in term_trace):
            raise RuntimeError(f"{pose_id} final reset cycle retained active force")
        evaluator = P0CoverageRuntime(
            env,
            output / f"coverage-{pose_id}",
            task_id=TASK_ID,
            seed=int(config["environment_seeds"][pose_id]),
            commit=_git("rev-parse", "HEAD"),
            branch=_git("branch", "--show-current"),
            require_camera_facing_normal=True,
            camera_facing_normal_sign=-1,
            raycast_device=str(base.device),
            print_updates=False,
            unreachable_region_path=ROOT / config["unreachable_region"]["path"],
        )
        sync = dict(base._task009b_policy_rgb_sync_latest)
        rgb = observation["policy"]["rgb"]
        digest = _stable_rgb_digest(rgb)
        update = evaluator.maybe_update(
            expected_camera_frame=int(sync["frame"]), rgb_content_sha256=digest
        )
        if update is None or update.coverage_fraction <= 0.0:
            raise RuntimeError(f"{pose_id} final HOLD frame did not initialize nonzero C0")
        pose = base.scene["capsule"].data.root_pose_w.torch[0].detach().cpu().numpy()
        velocity = base.scene["capsule"].data.root_com_vel_w.torch[0].detach().cpu().numpy()
        finite = bool(np.isfinite(pose).all() and np.isfinite(velocity).all() and torch.isfinite(rgb).all())
        row = {
            "pose_id": pose_id,
            "environment_seed": int(config["environment_seeds"][pose_id]),
            "requested_pose_world_xyzw": info["requested_pose_world_xyzw"],
            "write_position_error_m": info["write_position_error_m"],
            "write_quaternion_absolute_alignment": info["write_quaternion_absolute_alignment"],
            "hold_frames": frames,
            "hold_start_sim_time_s": float(trace[0]["start_sim_time_s"]),
            "hold_end_sim_time_s": float(trace[-1]["end_sim_time_s"]),
            "stable_pose_world_xyzw": info["stable_pose_world_xyzw"],
            "stable_velocity_world": info["stable_velocity_world"],
            "final_rgb_content_sha256": digest,
            "initial_reachable_coverage_fraction": float(update.coverage_fraction),
            "initial_raw_coverage_fraction": float(evaluator.raw_accumulator.coverage_fraction),
            "coverage_updates": 1,
            "episode_length_buf": int(base.episode_length_buf[0].item()),
            "active_force_zero": True,
            "finite": finite,
        }
        if not finite or row["episode_length_buf"] != 0:
            raise RuntimeError(f"{pose_id} reset returned invalid state: {row}")
        _append(rows_path, row)
        evaluator.finalize("task009c_reset_only")
        print(
            f"TASK009C_RESET pose={pose_id} frames={frames[0]}..{frames[-1]} "
            f"C0={100.0 * update.coverage_fraction:.3f}% pass=True",
            flush=True,
        )
    return {
        "status": "pass",
        "gate": 2,
        "validated_pose_ids": config["validation_pose_ids"],
        "validated_pose_count": len(config["validation_pose_ids"]),
        "devices": devices,
        "rows": _artifact(rows_path),
    }


def main() -> int:
    config = load_random_baseline_config(args_cli.config)
    loaded_config, manifest, allowed = _load_task009c_pose_records(args_cli.config)
    if config["config_sha256"] != loaded_config["config_sha256"]:
        raise RuntimeError("TASK-009C configuration changed between loaders")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    mode_name = "reset_only" if args_cli.reset_only else "smoke" if args_cli.smoke else "formal"
    output = args_cli.output_root / f"{mode_name}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=True)
    cfg.sim.render_interval = 24
    env = None
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(TASK_ID, cfg=cfg)
            if args_cli.reset_only:
                summary = _run_reset_only(env, output, config, manifest, allowed)
            else:
                raise NotImplementedError("Gate 3 episode runner is not implemented yet")
        finally:
            if env is not None:
                env.close()
    summary.update(
        {
            "mode": mode_name,
            "config_sha256": config["config_sha256"],
            "repository_commit": _git("rev-parse", "HEAD"),
            "repository_branch": _git("branch", "--show-current"),
        }
    )
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    print("TASK009C_COMPLETE " + json.dumps({**summary, "summary": _artifact(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if simulation_app.is_running():
            simulation_app.close()
