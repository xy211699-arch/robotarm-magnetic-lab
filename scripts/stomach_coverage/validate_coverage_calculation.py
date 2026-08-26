#!/usr/bin/env python3
"""Validate TASK-009B 70 mm area-weighted coverage from frozen library poses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

from isaaclab.app import AppLauncher


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument(
    "--pose_manifest",
    type=Path,
    default=ROOT / "configs/task009b/pose_library_manifest_v1.json",
)
parser.add_argument(
    "--coverage_manifest",
    type=Path,
    default=ROOT / "configs/task009b/coverage_manifest_v1.json",
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=Path("/mnt/isaac-linux/robotarm_magnetic_lab_artifacts/task009b_coverage_validation"),
)
parser.add_argument("--raycast_device", default="cuda:0")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, visualizer=[])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this validator only accepts {TASK_ID}")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.coverage.accumulator import CoverageAccumulator
from robotarm_magnetic_lab.coverage.area_weights import (
    target_vertex_area_weights,
    weights_sha256,
)
from robotarm_magnetic_lab.coverage.entry_pose_library import (
    POSE_LIBRARY_MANIFEST_SCHEMA,
    SPLIT_COUNTS,
    file_sha256,
    manifest_hash,
    read_jsonl,
)
from robotarm_magnetic_lab.coverage.simulator_runtime import (
    DEFAULT_INNER_SURFACE_PATH,
    reference_from_stage,
)
from robotarm_magnetic_lab.coverage.visibility import (
    FOV_HALF_ANGLE_DEG,
    HIT_DISTANCE_TOLERANCE_M,
    MAX_OBSERVATION_DISTANCE_M,
    WarpFirstHitRaycaster,
    camera_facing_first_hits,
    candidate_vertices,
    visible_from_first_hits,
)
from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    ParameterizedForceMode,
)


COVERAGE_MANIFEST_SCHEMA = "robotarm_magnetic_lab.task009b_coverage_manifest"
CAMERA_FACING_NORMAL_SIGN = -1


def _tensor(value):
    return getattr(value, "torch", value)


def _write_json(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_hashed_manifest(path: Path, schema: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise RuntimeError(f"manifest schema mismatch: {path}")
    expected = payload.get("config_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "config_sha256"}
    if manifest_hash(unhashed) != expected:
        raise RuntimeError(f"manifest hash mismatch: {path}")
    return payload


def _write_state(capsule, pose_xyzw: np.ndarray, device: str) -> None:
    pose = torch.as_tensor(pose_xyzw, device=device, dtype=torch.float32).reshape(1, 7)
    capsule.write_root_pose_to_sim_index(root_pose=pose)
    capsule.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=device, dtype=torch.float32)
    )


def main() -> int:
    if not np.isclose(MAX_OBSERVATION_DISTANCE_M, 0.07, atol=0.0, rtol=0.0):
        raise RuntimeError("TASK-009B coverage distance must be exactly 70 mm")
    pose_manifest = _load_hashed_manifest(args_cli.pose_manifest, POSE_LIBRARY_MANIFEST_SCHEMA)
    if pose_manifest.get("live_reload_validation", {}).get("status") != "pass":
        raise RuntimeError("pose library has not passed Gate 3 live reload")
    data_path = Path(pose_manifest["data_path"])
    if not data_path.is_file() or file_sha256(data_path) != pose_manifest["data_sha256"]:
        raise RuntimeError("pose library data is missing or has changed")
    records = read_jsonl(data_path)
    by_id = {record["pose_id"]: record for record in records}
    selected_ids = pose_manifest["fixed_live_reload_pose_ids"]
    if any(len(selected_ids[split]) != 20 for split in SPLIT_COUNTS):
        raise RuntimeError("Gate 4 requires the frozen 20/20/20 pose selection")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = args_cli.output_root / timestamp
    output.mkdir(parents=True, exist_ok=False)
    frame_log_path = output / "coverage_boundaries.jsonl"
    summary_path = output / "summary.json"
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    # The task's short default episode would reset the camera frame counter after
    # 20 boundaries.  Gate 4 requires one uninterrupted 60-boundary union test.
    cfg.episode_length_s = 100.0
    env = None
    rows: list[dict] = []
    started_wall = time.perf_counter()

    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            base = env.unwrapped
            capsule = base.scene["capsule"]
            camera = base.scene["capsule_camera"]
            term = base.action_manager.get_term("parameterized_force")
            reference = reference_from_stage()
            if reference.geometry_sha256 != pose_manifest["stomach_geometry_sha256"]:
                raise RuntimeError("live coverage target geometry differs from the pose library")
            weights = target_vertex_area_weights(reference)
            accumulator = CoverageAccumulator(len(reference.vertices_world), weights)
            raycaster = WarpFirstHitRaycaster(reference, device=args_cli.raycast_device)
            previous_fraction = 0.0
            previous_camera_frame = None

            def evaluate_boundary(pose_id: str, split: str, phase: str) -> dict:
                nonlocal previous_fraction, previous_camera_frame
                requested = np.asarray(by_id[pose_id]["pose_world_xyzw"], dtype=np.float64)
                term.reset()
                capsule.permanent_wrench_composer.reset()
                _write_state(capsule, requested, base.device)
                base.sim.forward()
                base.scene.update(0.0)
                hold = torch.tensor(
                    [[float(ParameterizedForceMode.HOLD), 0.5]],
                    device=base.device,
                    dtype=torch.float32,
                )
                observation, *_ = env.step(hold)
                rgb_values = _tensor(camera.data.output["rgb"]).detach().cpu().numpy()
                camera_frame = int(_tensor(camera.frame)[0].item())
                forced_sensor_sync = False
                if previous_camera_frame is not None and camera_frame <= previous_camera_frame:
                    # A stored-pose reload permits necessary sensor synchronization without
                    # advancing physics or adding an environment action.  Only force a capture
                    # when the 0.1 s floating-point scheduler reused the previous buffer.
                    # Isaac Lab 3.0's public force_recompute only processes sensors already
                    # marked outdated.  The floating-point boundary miss leaves that mask
                    # false, so invoke the camera's no-time-check capture implementation.
                    camera._update_buffers_impl(camera._ALL_ENV_MASK)
                    rgb_values = _tensor(camera.data.output["rgb"]).detach().cpu().numpy()
                    camera_frame = int(_tensor(camera.frame)[0].item())
                    forced_sensor_sync = True
                if previous_camera_frame is not None and camera_frame <= previous_camera_frame:
                    raise RuntimeError("camera RGB frame did not advance after required synchronization")
                previous_camera_frame = camera_frame
                center = _tensor(camera.data.pos_w)[0].detach().cpu().numpy().astype(np.float64)
                quaternion = (
                    _tensor(camera.data.quat_w_ros)[0].detach().cpu().numpy().astype(np.float64)
                )
                optical_axis = rotation_matrix_from_xyzw(quaternion) @ np.asarray([0.0, 0.0, 1.0])
                candidates, distances = candidate_vertices(
                    reference.vertices_world,
                    center,
                    optical_axis,
                    max_distance_m=MAX_OBSERVATION_DISTANCE_M,
                    half_angle_deg=FOV_HALF_ANGLE_DEG,
                )
                query_started = time.perf_counter()
                hit_distances, hit_faces = raycaster.query(
                    center, reference.vertices_world[candidates]
                )
                first_hit = visible_from_first_hits(
                    candidates,
                    distances,
                    hit_distances,
                    hit_faces,
                    reference.incident_triangles,
                    tolerance_m=HIT_DISTANCE_TOLERANCE_M,
                )
                normal_facing = camera_facing_first_hits(
                    center,
                    reference.vertices_world[candidates],
                    hit_faces,
                    reference,
                    normal_sign=CAMERA_FACING_NORMAL_SIGN,
                )
                visible_indices = candidates[first_hit & normal_facing]
                query_elapsed = time.perf_counter() - query_started
                update = accumulator.update(camera_frame, visible_indices)
                finite = bool(
                    observation is not None
                    and np.isfinite(rgb_values).all()
                    and np.isfinite(center).all()
                    and np.isfinite(quaternion).all()
                    and np.isfinite(hit_distances[np.isfinite(hit_distances)]).all()
                )
                if not finite:
                    raise RuntimeError(f"non-finite coverage boundary for {pose_id}")
                if not 0.0 <= update.coverage_fraction <= 1.0:
                    raise RuntimeError("coverage fraction is outside [0,1]")
                if update.coverage_fraction + 1.0e-15 < previous_fraction:
                    raise RuntimeError("cumulative area coverage decreased")
                previous_fraction = update.coverage_fraction
                row = {
                    "phase": phase,
                    "pose_id": pose_id,
                    "split": split,
                    "rgb_frame_id": camera_frame,
                    "rgb_shape": list(rgb_values.shape),
                    "forced_sensor_sync": forced_sensor_sync,
                    "candidate_count": int(len(candidates)),
                    "first_hit_visible_count": int(np.count_nonzero(first_hit)),
                    "normal_facing_count": int(np.count_nonzero(normal_facing)),
                    "current_visible_count": int(update.visible_count),
                    "newly_covered_count": int(update.newly_covered_count),
                    "cumulative_count": int(update.cumulative_count),
                    "target_vertex_count": int(len(weights)),
                    "current_visible_area_m2": float(update.visible_area_m2),
                    "newly_covered_area_m2": float(update.newly_covered_area_m2),
                    "cumulative_area_m2": float(update.cumulative_area_m2),
                    "target_total_area_m2": float(update.total_area_m2),
                    "coverage_fraction": float(update.coverage_fraction),
                    "coverage_query_s": float(query_elapsed),
                    "finite": finite,
                }
                rows.append(row)
                print("TASK009B_COVERAGE_BOUNDARY " + json.dumps(row, sort_keys=True), flush=True)
                return row

            for split in SPLIT_COUNTS:
                for pose_id in selected_ids[split]:
                    evaluate_boundary(pose_id, split, "monotonic_union")

            before_reset_fraction = accumulator.coverage_fraction
            before_reset_count = accumulator.recorded_frame_count
            accumulator.reset()
            if accumulator.recorded_frame_count != 0 or accumulator.coverage_fraction != 0.0:
                raise RuntimeError("coverage reset did not clear cumulative state and frame IDs")
            previous_fraction = 0.0
            initial_pose_id = selected_ids["train"][0]
            initial_row = evaluate_boundary(initial_pose_id, "train", "post_reset_initial_C0")
            reset_check = {
                "before_reset_coverage_fraction": before_reset_fraction,
                "before_reset_frame_count": before_reset_count,
                "after_reset_coverage_fraction": 0.0,
                "after_reset_frame_count": 0,
                "initial_C0_coverage_fraction": initial_row["coverage_fraction"],
                "pass": True,
            }
        finally:
            if env is not None:
                env.close()

    with frame_log_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    target_stats = {
        "source": "legacy P0 approved luminal target set",
        "surface_prim_path": DEFAULT_INNER_SURFACE_PATH,
        "stomach_geometry_sha256": pose_manifest["stomach_geometry_sha256"],
        "target_vertex_count": int(len(weights)),
        "positive_weight_vertex_count": int(np.count_nonzero(weights > 0.0)),
        "target_triangle_count": int(len(reference.triangles)),
        "target_total_area_m2": float(weights.sum()),
        "vertex_weights_sha256": weights_sha256(weights),
        "weight_min_positive_m2": float(weights[weights > 0.0].min()),
        "weight_max_m2": float(weights.max()),
    }
    summary = {
        "status": "pass",
        "pose_manifest_config_sha256": pose_manifest["config_sha256"],
        "distance_limit_m": MAX_OBSERVATION_DISTANCE_M,
        "fov_half_angle_deg": FOV_HALF_ANGLE_DEG,
        "normal_sign": CAMERA_FACING_NORMAL_SIGN,
        "first_hit_tolerance_m": HIT_DISTANCE_TOLERANCE_M,
        "validated_boundaries_before_reset": 60,
        "validated_boundaries_total": len(rows),
        "all_finite": all(row["finite"] for row in rows),
        "monotonic_union_pass": True,
        "range_pass": True,
        "reset_check": reset_check,
        "final_coverage_fraction_after_reset_C0": rows[-1]["coverage_fraction"],
        "maximum_coverage_fraction_before_reset": max(
            row["coverage_fraction"] for row in rows if row["phase"] == "monotonic_union"
        ),
        "target": target_stats,
        "frame_log_path": str(frame_log_path.resolve()),
        "frame_log_bytes": frame_log_path.stat().st_size,
        "frame_log_sha256": file_sha256(frame_log_path),
        "elapsed_wall_s": time.perf_counter() - started_wall,
    }
    _write_json(summary_path, summary)
    coverage_manifest = {
        "schema": COVERAGE_MANIFEST_SCHEMA,
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": args_cli.task,
        "pose_manifest_config_sha256": pose_manifest["config_sha256"],
        "target": target_stats,
        "visibility": {
            "max_distance_m": MAX_OBSERVATION_DISTANCE_M,
            "fov_full_angle_deg": 2.0 * FOV_HALF_ANGLE_DEG,
            "normal_sign": CAMERA_FACING_NORMAL_SIGN,
            "first_hit_tolerance_m": HIT_DISTANCE_TOLERANCE_M,
            "ray_backend": "isaaclab.utils.warp.ops.raycast_mesh CUDA first hit",
        },
        "validation": summary,
    }
    coverage_manifest["config_sha256"] = manifest_hash(coverage_manifest)
    _write_json(args_cli.coverage_manifest, coverage_manifest)
    print("TASK009B_COVERAGE_VALIDATION_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
