#!/usr/bin/env python3
"""Run TASK-009C reset validation, smoke, or formal random baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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
    "--save_best_pose_snapshots",
    action="store_true",
    help="capture 30-second candidates and retain per-policy plus overall best pose images",
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
from robotarm_magnetic_lab.baselines.random_policies import build_policy, load_random_baseline_config
from robotarm_magnetic_lab.baselines.random_baseline_comparison import (
    preserve_best_snapshot_images,
)
from robotarm_magnetic_lab.coverage.entry_pose_library import file_sha256
from robotarm_magnetic_lab.coverage.simulator_runtime import P0CoverageRuntime
from robotarm_magnetic_lab.runtime.task009c_episode_runner import (
    EpisodeProtocolError,
    EpisodeSpec,
    SynchronousEpisodeRunner,
    read_episode_jsonl,
    validate_episode_records,
)
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


def _atomic_write_json(path: Path, row: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    _write_json(temporary, row)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _pose_request(record: dict, manifest_hash: str, config: dict) -> dict:
    return {
        "pose_id": record["pose_id"],
        "split": record["split"],
        "pose_world_xyzw": record["pose_world_xyzw"],
        "pose_library_manifest_config_sha256": manifest_hash,
        "config_path": str(args_cli.config.resolve()),
        "config_sha256": config["config_sha256"],
    }


def _latest_pointer_path(output_root: Path, kind: str) -> Path:
    return output_root / f"latest_{kind}_manifest.json"


def _update_latest_pointer(output_root: Path, kind: str, run_id: str, manifest_path: Path) -> None:
    _atomic_write_json(
        _latest_pointer_path(output_root, kind),
        {
            "schema": "robotarm_magnetic_lab.task009c_latest_manifest",
            "version": 1,
            "kind": kind,
            "run_id": run_id,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": _sha256(manifest_path),
        },
    )


def _append_run_record(
    manifest_path: Path, output_root: Path, kind: str, run_id: str, row: dict
) -> None:
    _append(manifest_path, row)
    _update_latest_pointer(output_root, kind, run_id, manifest_path)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _valid_completed_episode(entry: dict, spec: EpisodeSpec) -> bool:
    try:
        if entry.get("record_type") != "episode" or entry.get("status") != "pass":
            return False
        path = Path(entry["boundary_log_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["boundary_log_bytes"])
            or _sha256(path) != entry["boundary_log_sha256"]
        ):
            return False
        validate_episode_records(
            read_episode_jsonl(path),
            expected_cycles=spec.action_cycles,
            expected_episode_id=spec.episode_id,
        )
        return True
    except (KeyError, OSError, TypeError, ValueError, EpisodeProtocolError):
        return False


def _select_run_directory(output_root: Path, kind: str, config: dict) -> tuple[str, Path, Path, list[dict]]:
    """Resume only an incomplete, hash-verified formal manifest."""
    output_root.mkdir(parents=True, exist_ok=True)
    pointer_path = _latest_pointer_path(output_root, kind)
    if kind == "formal" and pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_path = Path(pointer.get("manifest_path", ""))
        valid_pointer = (
            pointer.get("kind") == kind
            and manifest_path.is_file()
            and manifest_path.stat().st_size == int(pointer.get("manifest_bytes", -1))
            and _sha256(manifest_path) == pointer.get("manifest_sha256")
        )
        if valid_pointer:
            rows = _read_jsonl(manifest_path)
            incomplete = rows and rows[-1].get("record_type") != "run_complete"
            matching = rows and rows[0].get("config_sha256") == config["config_sha256"]
            if incomplete and matching:
                return str(pointer["run_id"]), manifest_path.parent, manifest_path, rows
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    run_id = f"{kind}-{stamp}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    return run_id, output, output / "run_manifest.jsonl", []


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


def _active_reset_events(base) -> list[str]:
    manager = getattr(base, "event_manager", None)
    terms = getattr(manager, "active_terms", ()) if manager is not None else ()
    if isinstance(terms, dict):
        return sorted(
            f"{mode}:{name}"
            for mode, names in terms.items()
            for name in names
        )
    return sorted(str(name) for name in terms)


def _run_reset_only(env, output: Path, config: dict, manifest: dict, allowed: dict) -> dict:
    rows_path = output / "reset_only.jsonl"
    base = env.unwrapped
    devices = _assert_gpu(base)
    for pose_id in config["validation_pose_ids"]:
        record = allowed[pose_id]
        observation, extras = env.reset(
            seed=int(config["environment_seeds"][pose_id]),
            options={TASK009C_OPTION_KEY: _pose_request(record, manifest["config_sha256"], config)},
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


def _run_episode_batch(
    env,
    output: Path,
    output_root: Path,
    manifest_path: Path,
    existing_rows: list[dict],
    config: dict,
    pose_manifest: dict,
    allowed: dict,
    *,
    kind: str,
    run_id: str,
) -> dict:
    base = env.unwrapped
    devices = _assert_gpu(base)
    episode_records = config["smoke_episodes" if kind == "smoke" else "formal_episodes"]
    specs = [EpisodeSpec.from_record(record) for record in episode_records]
    completed_entries = {
        row.get("episode_id"): row for row in existing_rows if row.get("record_type") == "episode"
    }
    if not existing_rows:
        _append_run_record(
            manifest_path,
            output_root,
            kind,
            run_id,
            {
                "record_type": "run_start",
                "schema": "robotarm_magnetic_lab.task009c_run_manifest",
                "version": 1,
                "kind": kind,
                "run_id": run_id,
                "config_sha256": config["config_sha256"],
                "config_file": _artifact(args_cli.config),
                "repository_commit": _git("rev-parse", "HEAD"),
                "repository_branch": _git("branch", "--show-current"),
                "devices": devices,
                "num_envs": int(base.num_envs),
                "physics_hz": 1.0 / float(base.sim.get_physics_dt()),
                "control_hz": 1.0 / float(base.step_dt),
                "physics_steps_per_action": int(config["clocks"]["physics_steps_per_action"]),
                "active_reset_events": _active_reset_events(base),
                "expected_episode_ids": [spec.episode_id for spec in specs],
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    passed = 0
    skipped = 0
    for index, spec in enumerate(specs, start=1):
        existing = completed_entries.get(spec.episode_id)
        if existing is not None and _valid_completed_episode(existing, spec):
            passed += 1
            skipped += 1
            print(
                f"TASK009C_EPISODE_SKIP index={index}/{len(specs)} id={spec.episode_id} hash_valid=True",
                flush=True,
            )
            continue
        observation, extras = env.reset(
            seed=spec.environment_seed,
            options={
                TASK009C_OPTION_KEY: _pose_request(
                    allowed[spec.pose_id], pose_manifest["config_sha256"], config
                )
            },
        )
        reset_info = extras[TASK009C_OPTION_KEY]
        evaluator = P0CoverageRuntime(
            env,
            output / "coverage" / spec.episode_id,
            task_id=TASK_ID,
            seed=spec.environment_seed,
            commit=_git("rev-parse", "HEAD"),
            branch=_git("branch", "--show-current"),
            require_camera_facing_normal=True,
            camera_facing_normal_sign=-1,
            raycast_device=str(base.device),
            print_updates=False,
            unreachable_region_path=ROOT / config["unreachable_region"]["path"],
        )
        runner = SynchronousEpisodeRunner(
            env, evaluator, config_sha256=config["config_sha256"], run_id=run_id
        )
        boundary_path = output / "episodes" / f"{spec.episode_id}.jsonl"
        summary_path = output / "episode_summaries" / f"{spec.episode_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_boundaries = {
                int(round(float(second) / 0.1)): int(second)
                for second in config["candidate_times_s"]
            }

            def capture_candidate(boundary: int, _row: dict) -> None:
                second = snapshot_boundaries.get(boundary)
                if args_cli.save_best_pose_snapshots and second is not None:
                    evaluator.snapshot(f"candidate_{second:03d}s")

            _, episode_summary = runner.run(
                spec=spec,
                policy=build_policy(spec.policy_id, spec.policy_seed, config),
                initial_observation=observation,
                output_path=boundary_path,
                boundary_callback=capture_candidate,
            )
            coverage_directory = evaluator.finalize("task009c_episode_complete")
        except Exception as exc:
            try:
                evaluator.finalize("task009c_episode_failed")
            finally:
                _append_run_record(
                    manifest_path,
                    output_root,
                    kind,
                    run_id,
                    {
                        "record_type": "episode_failure",
                        "episode_id": spec.episode_id,
                        "status": "failed",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            raise
        episode_summary.update(
            {
                "reset_write_position_error_m": reset_info["write_position_error_m"],
                "reset_write_quaternion_absolute_alignment": reset_info[
                    "write_quaternion_absolute_alignment"
                ],
                "coverage_artifact_directory": str(Path(coverage_directory).resolve()),
            }
        )
        _atomic_write_json(summary_path, episode_summary)
        entry = {
            "record_type": "episode",
            "episode_id": spec.episode_id,
            "kind": kind,
            "policy_id": spec.policy_id,
            "pose_id": spec.pose_id,
            "environment_seed": spec.environment_seed,
            "policy_seed": spec.policy_seed,
            "status": "pass",
            "boundary_log_path": str(boundary_path.resolve()),
            "boundary_log_bytes": boundary_path.stat().st_size,
            "boundary_log_sha256": _sha256(boundary_path),
            "summary_path": str(summary_path.resolve()),
            "summary_bytes": summary_path.stat().st_size,
            "summary_sha256": _sha256(summary_path),
            "boundary_count": episode_summary["boundary_count"],
            "action_cycles": episode_summary["action_cycles"],
            "C0_reachable": episode_summary["C0_reachable"],
            "C_final_reachable": episode_summary["C_final_reachable"],
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_run_record(manifest_path, output_root, kind, run_id, entry)
        completed_entries[spec.episode_id] = entry
        passed += 1
        print(
            f"TASK009C_EPISODE index={index}/{len(specs)} id={spec.episode_id} "
            f"points={episode_summary['boundary_count']} "
            f"C0={100.0 * episode_summary['C0_reachable']:.3f}% "
            f"Cend={100.0 * episode_summary['C_final_reachable']:.3f}% pass=True",
            flush=True,
        )
    if passed != len(specs):
        raise EpisodeProtocolError(f"only {passed}/{len(specs)} configured episodes passed")
    best_snapshot_manifest = None
    if args_cli.save_best_pose_snapshots:
        best_snapshot_manifest = preserve_best_snapshot_images(
            output,
            completed_entries.values(),
            [int(value) for value in config["candidate_times_s"]],
        )
    _append_run_record(
        manifest_path,
        output_root,
        kind,
        run_id,
        {
            "record_type": "run_complete",
            "status": "pass",
            "episode_count": passed,
            "resumed_episode_count": skipped,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    result = {
        "status": "pass",
        "gate": 4 if kind == "smoke" else 5,
        "run_id": run_id,
        "episode_count": passed,
        "resumed_episode_count": skipped,
        "devices": devices,
        "run_manifest": _artifact(manifest_path),
        "stable_pointer": _artifact(_latest_pointer_path(output_root, kind)),
    }
    if best_snapshot_manifest is not None:
        result["best_pose_snapshot_manifest"] = _artifact(best_snapshot_manifest)
    return result


def main() -> int:
    config = load_random_baseline_config(args_cli.config)
    loaded_config, manifest, allowed = _load_task009c_pose_records(args_cli.config)
    if config["config_sha256"] != loaded_config["config_sha256"]:
        raise RuntimeError("TASK-009C configuration changed between loaders")
    mode_name = "reset_only" if args_cli.reset_only else "smoke" if args_cli.smoke else "formal"
    if mode_name == "reset_only":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        output = args_cli.output_root / f"reset_only-{stamp}"
        output.mkdir(parents=True, exist_ok=False)
        run_id = output.name
        manifest_path = None
        existing_rows = []
    else:
        run_id, output, manifest_path, existing_rows = _select_run_directory(
            args_cli.output_root, mode_name, config
        )
    cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=True)
    cfg.sim.render_interval = 24
    env = None
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(TASK_ID, cfg=cfg)
            if args_cli.reset_only:
                summary = _run_reset_only(env, output, config, manifest, allowed)
            else:
                assert manifest_path is not None
                summary = _run_episode_batch(
                    env,
                    output,
                    args_cli.output_root,
                    manifest_path,
                    existing_rows,
                    config,
                    manifest,
                    allowed,
                    kind=mode_name,
                    run_id=run_id,
                )
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
