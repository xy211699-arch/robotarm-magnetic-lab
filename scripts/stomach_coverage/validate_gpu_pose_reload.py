#!/usr/bin/env python3
"""Reload the frozen TASK-009B 20/20/20 poses in the formal GPU environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(ROOT / "scripts"))

from _artifact_paths import artifact_root
from isaaclab.app import AppLauncher


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument(
    "--manifest_path", type=Path, default=ROOT / "configs/task009b/pose_library_manifest_v1.json"
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=artifact_root(ROOT) / "task009b_gpu_pose_reload_validation",
)
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
from robotarm_magnetic_lab.coverage.entry_pose_library import (
    LIVE_RELOAD_COUNT_PER_SPLIT,
    MIN_UNORIENTED_AXIS_ANGLE_DEG,
    POSE_LIBRARY_MANIFEST_SCHEMA,
    SPLIT_COUNTS,
    file_sha256,
    manifest_hash,
    read_jsonl,
    stable_record_is_valid,
    unoriented_axis_angle_deg,
)
from robotarm_magnetic_lab.coverage.entry_pose_library import INWARD_NORMAL_SIGN
from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage
from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface.surface_mesh import (
    SurfaceNavigationMesh,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    PHYSICS_STEPS_PER_CONTROL,
    ParameterizedForceMode,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.task009b_training_env import (
    _stable_rgb_digest,
)


def _tensor(value):
    return getattr(value, "torch", value)


def _pose(capsule) -> np.ndarray:
    return _tensor(capsule.data.root_pose_w)[0].detach().cpu().numpy().astype(np.float64)


def _velocity(capsule) -> np.ndarray:
    return _tensor(capsule.data.root_com_vel_w)[0].detach().cpu().numpy().astype(np.float64)


def _write_state(capsule, pose_xyzw: np.ndarray, device: str) -> None:
    pose = torch.as_tensor(pose_xyzw, device=device, dtype=torch.float32).reshape(1, 7)
    capsule.write_root_pose_to_sim_index(root_pose=pose)
    capsule.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=device, dtype=torch.float32)
    )


def _load_manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != POSE_LIBRARY_MANIFEST_SCHEMA:
        raise RuntimeError("pose-library manifest schema mismatch")
    payload = {key: value for key, value in manifest.items() if key != "config_sha256"}
    if manifest_hash(payload) != manifest.get("config_sha256"):
        raise RuntimeError("pose-library manifest hash mismatch")
    data_path = Path(manifest["data_path"])
    if not data_path.is_file() or file_sha256(data_path) != manifest["data_sha256"]:
        raise RuntimeError("pose-library data path or hash mismatch")
    records = read_jsonl(data_path)
    return manifest, records


def _signed_lumen_distance(surface, point: np.ndarray) -> float:
    hit = surface.closest_hit(point)
    return float(np.dot(point - hit.point_world, hit.normal_world))


def main() -> int:
    manifest, records = _load_manifest(args_cli.manifest_path)
    by_id = {record["pose_id"]: record for record in records}
    selected = manifest["fixed_live_reload_pose_ids"]
    if any(len(selected[name]) != LIVE_RELOAD_COUNT_PER_SPLIT for name in SPLIT_COUNTS):
        raise RuntimeError("fixed GPU reload set is not 20/20/20")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = args_cli.output_root / stamp
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "gpu_pose_reload.jsonl"
    summary_path = output / "summary.json"

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env = None
    rows = []
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            base = env.unwrapped
            if not str(base.device).startswith("cuda") or not str(base.sim.device).startswith("cuda"):
                raise RuntimeError(f"GPU reload requires CUDA PhysX, got {base.device}/{base.sim.device}")
            capsule = base.scene["capsule"]
            term = base.action_manager.get_term("parameterized_force")
            surface = SurfaceNavigationMesh.from_reference(reference_from_stage(), INWARD_NORMAL_SIGN)
            hold = torch.tensor(
                [[float(ParameterizedForceMode.HOLD), 0.5]],
                device=base.device,
                dtype=torch.float32,
            )
            with log_path.open("w", encoding="utf-8") as stream:
                for split in SPLIT_COUNTS:
                    for pose_id in selected[split]:
                        record = by_id[pose_id]
                        if not stable_record_is_valid(record):
                            raise RuntimeError(f"stored pose {pose_id} violates frozen CPU gate")
                        requested = np.asarray(record["pose_world_xyzw"], dtype=np.float64)
                        term.reset()
                        capsule.permanent_wrench_composer.reset()
                        _write_state(capsule, requested, base.device)
                        base.sim.forward()
                        base.scene.update(0.0)
                        restored = _pose(capsule)
                        reload_position_error = float(np.linalg.norm(restored[:3] - requested[:3]))
                        reload_quaternion_alignment = abs(float(np.dot(restored[3:], requested[3:])))
                        observation = None
                        for _ in range(10):
                            observation, _, terminated, truncated, _ = env.step(hold)
                            if bool(torch.any(terminated).item()) or bool(torch.any(truncated).item()):
                                raise RuntimeError(f"pose {pose_id} terminated during 1 s HOLD")
                            if len(term.current_cycle_trace) != PHYSICS_STEPS_PER_CONTROL:
                                raise RuntimeError(f"pose {pose_id} HOLD did not execute 24 substeps")
                        pose = _pose(capsule)
                        velocity = _velocity(capsule)
                        rgb = observation["policy"]["rgb"]
                        rotation = rotation_matrix_from_xyzw(pose[3:])
                        axis_angle = unoriented_axis_angle_deg(rotation[:, 2])
                        _, camera_center, other_center, _, _ = term._geometry()
                        camera_distance = _signed_lumen_distance(surface, camera_center)
                        other_distance = _signed_lumen_distance(surface, other_center)
                        finite = bool(
                            np.isfinite(pose).all()
                            and np.isfinite(velocity).all()
                            and torch.isfinite(rgb).all().item()
                        )
                        passed = bool(
                            finite
                            and reload_position_error <= 1.0e-5
                            and reload_quaternion_alignment >= 1.0 - 1.0e-5
                            and axis_angle >= MIN_UNORIENTED_AXIS_ANGLE_DEG
                            and camera_distance > 0.0
                            and other_distance > 0.0
                        )
                        row = {
                            "pose_id": pose_id,
                            "split": split,
                            "device": str(base.device),
                            "hold_cycles": 10,
                            "physics_substeps_per_cycle": PHYSICS_STEPS_PER_CONTROL,
                            "reload_position_error_m": reload_position_error,
                            "reload_quaternion_absolute_alignment": reload_quaternion_alignment,
                            "final_pose_world_xyzw": pose.tolist(),
                            "final_velocity_world": velocity.tolist(),
                            "unoriented_axis_angle_deg": axis_angle,
                            "camera_signed_lumen_distance_m": camera_distance,
                            "other_end_signed_lumen_distance_m": other_distance,
                            "camera_inside_lumen": camera_distance > 0.0,
                            "other_end_inside_lumen": other_distance > 0.0,
                            "rgb_content_sha256": _stable_rgb_digest(rgb),
                            "rgb_shape": list(rgb.shape),
                            "finite": finite,
                            "pass": passed,
                        }
                        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                        stream.flush()
                        rows.append(row)
                        print(
                            f"TASK009B_GPU_POSE_RELOAD pose={pose_id} split={split} pass={passed}",
                            flush=True,
                        )
                        if not passed:
                            raise RuntimeError(f"GPU pose reload failed for {pose_id}: {row}")
        finally:
            if env is not None:
                env.close()

    summary = {
        "status": "pass",
        "device": str(args_cli.device),
        "validated_total": len(rows),
        "validated_per_split": {
            split: sum(row["split"] == split for row in rows) for split in SPLIT_COUNTS
        },
        "all_finite": all(row["finite"] for row in rows),
        "all_camera_inside_lumen": all(row["camera_inside_lumen"] for row in rows),
        "all_other_end_inside_lumen": all(row["other_end_inside_lumen"] for row in rows),
        "all_axis_angles_at_least_45_deg": all(
            row["unoriented_axis_angle_deg"] >= MIN_UNORIENTED_AXIS_ANGLE_DEG for row in rows
        ),
        "log_path": str(log_path.resolve()),
        "log_bytes": log_path.stat().st_size,
        "log_sha256": file_sha256(log_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("TASK009B_GPU_POSE_RELOAD_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
