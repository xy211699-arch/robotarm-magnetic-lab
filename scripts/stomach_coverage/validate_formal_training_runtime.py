#!/usr/bin/env python3
"""Validate formal TASK-009B GPU stepping, RGB synchronization, and coverage."""

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
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--cycles", type=int, default=1000)
parser.add_argument("--resets", type=int, default=100)
parser.add_argument(
    "--unreachable_region",
    type=Path,
    default=None,
    help="Optional frozen unreachable-region JSON for reachable-ROI validation.",
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=artifact_root(ROOT) / "task009b_formal_runtime_validation",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, visualizer=[])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this validator only accepts {TASK_ID}")
if args_cli.cycles < 1000 or args_cli.resets < 100:
    parser.error("the acceptance contract requires at least 1000 cycles and 100 resets")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.coverage.entry_pose_library import file_sha256
from robotarm_magnetic_lab.coverage.simulator_runtime import P0CoverageRuntime
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    CONTROL_HZ,
    PHYSICS_HZ,
    PHYSICS_STEPS_PER_CONTROL,
    ParameterizedForceMode,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.task009b_training_env import (
    RESET_HOLD_CYCLES,
    _stable_rgb_digest,
)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append(stream, payload: dict) -> None:
    stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()


def _device_string(value) -> str:
    return str(value if value is not None else "unknown")


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = args_cli.output_root / stamp
    output.mkdir(parents=True, exist_ok=False)
    boundary_path = output / "control_boundaries.jsonl"
    reset_path = output / "reset_stabilization.jsonl"
    summary_path = output / "summary.json"
    coverage_directory = output / "coverage"

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    cfg.sim.render_interval = PHYSICS_STEPS_PER_CONTROL
    env = evaluator = None
    status = "fail"
    forced_captures = 0
    frame_ids: list[int] = []
    coverage_fractions: list[float] = []
    reset_frames_checked = 0
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            observation, reset_extras = env.reset()
            base = env.unwrapped
            camera = base.scene["capsule_camera"]
            camera_rgb = camera.data.output["rgb"].torch
            physics_view = base.sim.physics_sim_view
            device_info = {
                "requested_device": str(args_cli.device),
                "environment_device": _device_string(base.device),
                "cfg_sim_device": _device_string(base.cfg.sim.device),
                "simulation_context_device": _device_string(base.sim.device),
                "physics_sim_view_type": type(physics_view).__name__,
                "physics_sim_view_device": _device_string(
                    getattr(physics_view, "device", getattr(physics_view, "_device", base.sim.device))
                ),
                "camera_tensor_device": _device_string(camera_rgb.device),
                "coverage_raycast_device": _device_string(args_cli.device),
            }
            required_gpu_fields = (
                "environment_device",
                "cfg_sim_device",
                "simulation_context_device",
                "physics_sim_view_device",
                "camera_tensor_device",
                "coverage_raycast_device",
            )
            if any(not device_info[name].startswith("cuda") for name in required_gpu_fields):
                raise RuntimeError(f"formal environment is not fully on GPU: {device_info}")

            evaluator = P0CoverageRuntime(
                env,
                coverage_directory,
                task_id=TASK_ID,
                seed=909,
                commit=_git("rev-parse", "HEAD"),
                branch=_git("branch", "--show-current"),
                enable_view=False,
                require_camera_facing_normal=True,
                camera_facing_normal_sign=-1,
                raycast_device=str(args_cli.device),
                print_updates=False,
                unreachable_region_path=args_cli.unreachable_region,
            )
            initial_sync = dict(base._task009b_policy_rgb_sync_latest)
            initial_rgb = observation["policy"]["rgb"]
            initial_digest = _stable_rgb_digest(initial_rgb)
            initial_update = evaluator.maybe_update(
                expected_camera_frame=int(initial_sync["frame"]),
                rgb_content_sha256=initial_digest,
            )
            if initial_update is None or initial_update.coverage_fraction <= 0.0:
                raise RuntimeError("formal reset did not initialize nonzero C0 from final HOLD RGB")

            sequence = (
                (ParameterizedForceMode.HOLD, 0.5),
                (ParameterizedForceMode.MOVE_POS, 0.0),
                (ParameterizedForceMode.MOVE_NEG, 0.0),
                (ParameterizedForceMode.HOLD, 0.5),
                (ParameterizedForceMode.VIEW_POS, 0.0),
                (ParameterizedForceMode.VIEW_NEG, 0.0),
                (ParameterizedForceMode.HOLD, 0.5),
                (ParameterizedForceMode.UP, 0.0),
                (ParameterizedForceMode.HOLD, 0.5),
                (ParameterizedForceMode.HOLD, 0.5),
            )
            previous_frame = int(initial_sync["frame"])
            previous_fraction = float(initial_update.coverage_fraction)
            with boundary_path.open("w", encoding="utf-8") as stream:
                for index in range(args_cli.cycles):
                    mode, alpha = sequence[index % len(sequence)]
                    start_time = float(base.common_step_counter) * float(base.step_dt)
                    action = torch.tensor(
                        [[float(mode), float(alpha)]], device=base.device, dtype=torch.float32
                    )
                    observation, _, terminated, truncated, _ = env.step(action)
                    if bool(torch.any(terminated).item()) or bool(torch.any(truncated).item()):
                        raise RuntimeError(f"unexpected episode termination at cycle {index}")
                    trace = base.action_manager.get_term("parameterized_force").current_cycle_trace
                    sync = dict(base._task009b_policy_rgb_sync_latest)
                    actor_frame = int(sync["frame"])
                    rgb = observation["policy"]["rgb"]
                    digest = _stable_rgb_digest(rgb)
                    update = evaluator.maybe_update(
                        expected_camera_frame=actor_frame,
                        rgb_content_sha256=digest,
                    )
                    if update is None:
                        raise RuntimeError(f"coverage omitted control boundary {index}")
                    end_time = float(base.common_step_counter) * float(base.step_dt)
                    fraction = float(update.coverage_fraction)
                    checks = {
                        "physics_substeps": len(trace) == PHYSICS_STEPS_PER_CONTROL,
                        "duration": abs((end_time - start_time) - 0.1) <= 1.0e-9,
                        "actor_frame_increment": actor_frame == previous_frame + 1,
                        "actor_coverage_same_frame": (
                            int(evaluator.latest_record["camera_frame"]) == actor_frame
                        ),
                        "rgb_finite": bool(torch.isfinite(rgb).all().item()),
                        "coverage_monotonic": fraction + 1.0e-15 >= previous_fraction,
                    }
                    if not all(checks.values()):
                        raise RuntimeError(f"boundary {index} failed synchronization checks: {checks}")
                    row = {
                        "environment_step": index,
                        "mode_id": int(mode),
                        "mode": mode.name,
                        "alpha": float(alpha),
                        "physics_substeps": len(trace),
                        "boundary_start_sim_time_s": start_time,
                        "boundary_end_sim_time_s": end_time,
                        "actor_rgb_frame": actor_frame,
                        "coverage_rgb_frame": int(evaluator.latest_record["camera_frame"]),
                        "rgb_content_sha256": digest,
                        "forced_capture": bool(sync["forced_capture"]),
                        "coverage_fraction": fraction,
                        "checks": checks,
                    }
                    _append(stream, row)
                    frame_ids.append(actor_frame)
                    coverage_fractions.append(fraction)
                    forced_captures += int(bool(sync["forced_capture"]))
                    previous_frame = actor_frame
                    previous_fraction = fraction

            with reset_path.open("w", encoding="utf-8") as stream:
                for reset_index in range(args_cli.resets):
                    evaluator.reset(save_snapshot=False)
                    observation, reset_extras = env.reset()
                    trace = reset_extras["task009b_reset_stabilization"]
                    frames = [int(item["actor_rgb_frame"]) for item in trace]
                    expected_frames = list(range(frames[0], frames[0] + RESET_HOLD_CYCLES))
                    if len(trace) != RESET_HOLD_CYCLES or frames != expected_frames:
                        raise RuntimeError(
                            f"reset {reset_index} did not acquire exactly ten fresh episode frames: {frames}"
                        )
                    if any(
                        item["physics_substeps"] != PHYSICS_STEPS_PER_CONTROL
                        or not item["rgb_finite"]
                        or abs((item["end_sim_time_s"] - item["start_sim_time_s"]) - 0.1) > 1.0e-9
                        for item in trace
                    ):
                        raise RuntimeError(f"reset {reset_index} HOLD trace violated the contract")
                    if int(base.episode_length_buf[0].item()) != 0:
                        raise RuntimeError("reset HOLD stabilization was charged to episode budget")
                    final_rgb = observation["policy"]["rgb"]
                    final_digest = _stable_rgb_digest(final_rgb)
                    sync = dict(base._task009b_policy_rgb_sync_latest)
                    update = evaluator.maybe_update(
                        expected_camera_frame=int(sync["frame"]),
                        rgb_content_sha256=final_digest,
                        write_record=False,
                    )
                    if update is None or update.coverage_fraction <= 0.0:
                        raise RuntimeError(f"reset {reset_index} did not initialize C0")
                    row = {
                        "reset_index": reset_index,
                        "hold_cycles": len(trace),
                        "physics_substeps_per_cycle": PHYSICS_STEPS_PER_CONTROL,
                        "actor_rgb_frames": frames,
                        "final_actor_rgb_frame": int(sync["frame"]),
                        "coverage_rgb_frame": int(evaluator.latest_record["camera_frame"]),
                        "final_rgb_content_sha256": final_digest,
                        "initial_coverage_fraction": float(update.coverage_fraction),
                        "episode_length_after_stabilization": int(base.episode_length_buf[0].item()),
                    }
                    _append(stream, row)
                    reset_frames_checked += len(trace)
            status = "pass"
            coverage_final = evaluator.finalize("formal_runtime_validation")
            evaluator = None
        finally:
            if evaluator is not None:
                evaluator.finalize("exception")
            if env is not None:
                env.close()

    summary = {
        "status": status,
        "task_id": TASK_ID,
        "physics_hz": PHYSICS_HZ,
        "control_hz": CONTROL_HZ,
        "physics_substeps_per_control": PHYSICS_STEPS_PER_CONTROL,
        "long_sequence_cycles": len(frame_ids),
        "long_sequence_unique_frame_ids": len(set(frame_ids)),
        "long_sequence_forced_captures": forced_captures,
        "reset_count": args_cli.resets,
        "reset_hold_frames_checked": reset_frames_checked,
        "coverage_monotonic": all(
            second + 1.0e-15 >= first
            for first, second in zip(coverage_fractions, coverage_fractions[1:])
        ),
        "device_info": device_info,
        "coverage_target": {
            "geometry_sha256": evaluator.reference.geometry_sha256 if evaluator is not None else "see coverage metadata",
            "target_vertex_count": 24529,
            "target_triangle_count": 49047,
            "target_total_area_m2": 0.0644836229259155,
            "maximum_distance_m": 0.07,
        },
        "private_api_compatibility_risk": (
            "Formal observation synchronization isolates Camera._update_buffers_impl(); "
            "re-audit this call whenever Isaac Lab is upgraded."
        ),
        "logs": {},
    }
    for name, path in (
        ("control_boundaries", boundary_path),
        ("reset_stabilization", reset_path),
    ):
        summary["logs"][name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    summary["coverage_artifact_directory"] = str(coverage_final.resolve())
    _write_json(summary_path, summary)
    print("TASK009B_FORMAL_RUNTIME_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
