#!/usr/bin/env python3
"""Live-reload the frozen 20/20/20 TASK-009B pose-library sample."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

from isaaclab.app import AppLauncher


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument(
    "--manifest_path",
    type=Path,
    default=ROOT / "configs/task009b/pose_library_manifest_v1.json",
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=Path("/mnt/isaac-linux/robotarm_magnetic_lab_artifacts/task009b_pose_library_validation"),
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
from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    ParameterizedForceMode,
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


def _write_json(path: Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != POSE_LIBRARY_MANIFEST_SCHEMA:
        raise RuntimeError("pose-library manifest schema mismatch")
    expected = manifest.get("config_sha256")
    payload = {key: value for key, value in manifest.items() if key != "config_sha256"}
    if manifest_hash(payload) != expected:
        raise RuntimeError("pose-library manifest hash mismatch")
    return manifest


def main() -> int:
    manifest = _load_manifest(args_cli.manifest_path)
    data_path = Path(manifest["data_path"])
    if not data_path.is_absolute() or not data_path.is_file():
        raise RuntimeError("external pose-library data path is unavailable")
    if file_sha256(data_path) != manifest["data_sha256"]:
        raise RuntimeError("external pose-library data hash mismatch")
    records = read_jsonl(data_path)
    if len(records) != sum(SPLIT_COUNTS.values()):
        raise RuntimeError("pose-library total count mismatch")
    by_id = {record["pose_id"]: record for record in records}
    if len(by_id) != len(records):
        raise RuntimeError("pose IDs are not unique")
    split_id_sets = []
    for split, expected_count in SPLIT_COUNTS.items():
        ids = set(manifest["split_pose_ids"][split])
        if len(ids) != expected_count:
            raise RuntimeError(f"{split} manifest count mismatch")
        if any(by_id[pose_id]["split"] != split for pose_id in ids):
            raise RuntimeError(f"{split} contains an ID assigned to another split")
        split_id_sets.append(ids)
    if any(split_id_sets[i] & split_id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("train/validation/test pose IDs overlap")
    selected_ids = manifest["fixed_live_reload_pose_ids"]
    if any(len(selected_ids[split]) != LIVE_RELOAD_COUNT_PER_SPLIT for split in SPLIT_COUNTS):
        raise RuntimeError("fixed live-reload sample must contain 20 IDs per split")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = args_cli.output_root / timestamp
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "live_reload.jsonl"
    summary_path = output / "summary.json"
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    env = None
    rows = []
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            base = env.unwrapped
            capsule = base.scene["capsule"]
            camera = base.scene["capsule_camera"]
            term = base.action_manager.get_term("parameterized_force")
            for split in SPLIT_COUNTS:
                for pose_id in selected_ids[split]:
                    record = by_id[pose_id]
                    if not stable_record_is_valid(record):
                        raise RuntimeError(f"stored pose {pose_id} no longer satisfies the frozen gate")
                    requested = np.asarray(record["pose_world_xyzw"], dtype=np.float64)
                    term.reset()
                    capsule.permanent_wrench_composer.reset()
                    _write_state(capsule, requested, base.device)
                    base.sim.forward()
                    base.scene.update(0.0)
                    restored = _pose(capsule)
                    restored_velocity = _velocity(capsule)
                    position_error = float(np.linalg.norm(restored[:3] - requested[:3]))
                    quaternion_alignment = abs(float(np.dot(restored[3:], requested[3:])))
                    hold = torch.tensor(
                        [[float(ParameterizedForceMode.HOLD), 0.5]],
                        device=base.device,
                        dtype=torch.float32,
                    )
                    observation, *_ = env.step(hold)
                    rgb = camera.data.output["rgb"]
                    rgb_tensor = _tensor(rgb)
                    rgb_values = rgb_tensor.detach().cpu().numpy()
                    synchronized_pose = _pose(capsule)
                    synchronized_velocity = _velocity(capsule)
                    axis = rotation_matrix_from_xyzw(synchronized_pose[3:])[:, 2]
                    angle = unoriented_axis_angle_deg(axis)
                    finite = bool(
                        np.isfinite(restored).all()
                        and np.isfinite(restored_velocity).all()
                        and np.isfinite(synchronized_pose).all()
                        and np.isfinite(synchronized_velocity).all()
                        and np.isfinite(rgb_values).all()
                    )
                    passed = bool(
                        finite
                        and position_error <= 1.0e-5
                        and quaternion_alignment >= 1.0 - 1.0e-5
                        and angle >= MIN_UNORIENTED_AXIS_ANGLE_DEG
                        and rgb_values.size > 0
                        and observation is not None
                    )
                    row = {
                        "pose_id": pose_id,
                        "split": split,
                        "position_reload_error_m": position_error,
                        "quaternion_absolute_alignment": quaternion_alignment,
                        "zero_velocity_on_reload": bool(np.linalg.norm(restored_velocity) <= 1.0e-8),
                        "synchronized_pose_world_xyzw": synchronized_pose.tolist(),
                        "synchronized_velocity_world": synchronized_velocity.tolist(),
                        "unoriented_axis_angle_deg": angle,
                        "rgb_shape": list(rgb_values.shape),
                        "rgb_finite": bool(np.isfinite(rgb_values).all()),
                        "physical_state_finite": finite,
                        "pass": passed,
                    }
                    rows.append(row)
                    print("TASK009B_POSE_RELOAD " + json.dumps(row, sort_keys=True), flush=True)
                    if not passed:
                        raise RuntimeError(f"live reload failed for {pose_id}")
        finally:
            if env is not None:
                env.close()

    with log_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "status": "pass",
        "validated_total": len(rows),
        "validated_per_split": {
            split: sum(row["split"] == split for row in rows) for split in SPLIT_COUNTS
        },
        "all_finite": all(row["physical_state_finite"] for row in rows),
        "all_rgb_finite": all(row["rgb_finite"] for row in rows),
        "all_axis_angles_valid": all(
            row["unoriented_axis_angle_deg"] >= MIN_UNORIENTED_AXIS_ANGLE_DEG for row in rows
        ),
        "log_path": str(log_path.resolve()),
        "log_bytes": log_path.stat().st_size,
        "log_sha256": file_sha256(log_path),
    }
    _write_json(summary_path, summary)
    payload = {key: value for key, value in manifest.items() if key != "config_sha256"}
    payload["live_reload_validation"] = {
        **summary,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_pose_ids": selected_ids,
    }
    payload["config_sha256"] = manifest_hash(payload)
    _write_json(args_cli.manifest_path, payload)
    print("TASK009B_POSE_LIBRARY_RELOAD_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
