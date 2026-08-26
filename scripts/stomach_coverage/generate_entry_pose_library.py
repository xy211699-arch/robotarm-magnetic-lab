#!/usr/bin/env python3
"""Generate the frozen 1200-state TASK-009B entry pose library."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
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
    "--entry_anchor", type=Path, default=ROOT / "configs/task009b/entry_anchor_v1.json"
)
parser.add_argument(
    "--entry_region", type=Path, default=ROOT / "configs/task009b/entry_region_v1.json"
)
parser.add_argument(
    "--output_root",
    type=Path,
    default=Path("/mnt/isaac-linux/robotarm_magnetic_lab_artifacts/task009b_pose_library"),
)
parser.add_argument(
    "--manifest_path",
    type=Path,
    default=ROOT / "configs/task009b/pose_library_manifest_v1.json",
)
parser.add_argument("--maximum_attempt_multiplier", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, visualizer=[])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this generator only accepts {TASK_ID}")
if args_cli.maximum_attempt_multiplier < 1:
    parser.error("--maximum_attempt_multiplier must be positive")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import omni.usd
import torch

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.coverage.entry_pose_library import (
    INWARD_NORMAL_SIGN,
    LIVE_RELOAD_COUNT_PER_SPLIT,
    MIN_UNORIENTED_AXIS_ANGLE_DEG,
    POSE_LIBRARY_MANIFEST_SCHEMA,
    POSE_LIBRARY_SCHEMA,
    POSE_LIBRARY_VERSION,
    SPLIT_BASE_SEEDS,
    SPLIT_COUNTS,
    deterministic_candidate_seed,
    file_sha256,
    manifest_hash,
    pose_fingerprint,
    sample_surface_pose,
    stable_record_is_valid,
    unoriented_axis_angle_deg,
    write_jsonl,
)
from robotarm_magnetic_lab.coverage.entry_surface_region import (
    ANCHOR_SCHEMA,
    REGION_SCHEMA,
    load_and_validate,
)
from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage
from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface.surface_mesh import (
    SurfaceNavigationMesh,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    PHYSICS_HZ,
    PHYSICS_STEPS_PER_CONTROL,
    ParameterizedForceMode,
)


PHYSICS_DT_S = 1.0 / PHYSICS_HZ
STABLE_STEPS = 60
MAX_SETTLE_STEPS = 480
MAX_LINEAR_SPEED_M_S = 0.002
MAX_ANGULAR_SPEED_RAD_S = np.deg2rad(5.0)
CAMERA_LUMEN_SIDE_TOLERANCE_M = 0.0


def _tensor(value):
    return getattr(value, "torch", value)


def _pose(capsule) -> np.ndarray:
    return _tensor(capsule.data.root_pose_w)[0].detach().cpu().numpy().astype(np.float64)


def _link_pose(capsule) -> tuple[np.ndarray, np.ndarray]:
    position = _tensor(capsule.data.root_link_pos_w)[0].detach().cpu().numpy().astype(np.float64)
    quaternion = _tensor(capsule.data.root_link_quat_w)[0].detach().cpu().numpy().astype(np.float64)
    return position, quaternion


def _velocity(capsule) -> np.ndarray:
    return _tensor(capsule.data.root_com_vel_w)[0].detach().cpu().numpy().astype(np.float64)


def _write_state(capsule, pose_xyzw: np.ndarray, device: str) -> None:
    pose = torch.as_tensor(pose_xyzw, device=device, dtype=torch.float32).reshape(1, 7)
    capsule.write_root_pose_to_sim_index(root_pose=pose)
    capsule.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=device, dtype=torch.float32)
    )


def _camera_center_world(capsule, camera) -> np.ndarray:
    link_position, link_quaternion = _link_pose(capsule)
    offset = np.asarray(camera.cfg.offset.pos, dtype=np.float64).reshape(3)
    return link_position + rotation_matrix_from_xyzw(link_quaternion) @ offset


def _fixed_live_reload_ids(records: list[dict]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for split_index, split in enumerate(SPLIT_COUNTS):
        ids = [record["pose_id"] for record in records if record["split"] == split]
        rng = np.random.default_rng(940_001 + split_index)
        choices = np.sort(rng.choice(len(ids), size=LIVE_RELOAD_COUNT_PER_SPLIT, replace=False))
        selected[split] = [ids[int(index)] for index in choices]
    return selected


def _write_json(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    anchor = load_and_validate(args_cli.entry_anchor, ANCHOR_SCHEMA)
    region = load_and_validate(args_cli.entry_region, REGION_SCHEMA)
    if region["anchor_config_sha256"] != anchor["config_sha256"]:
        raise RuntimeError("entry region does not reference the confirmed anchor")
    if region["stomach_geometry_sha256"] != anchor["stomach_geometry_sha256"]:
        raise RuntimeError("entry anchor and region use different stomach geometry")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = args_cli.output_root / timestamp
    output.mkdir(parents=True, exist_ok=False)
    rejection_path = output / "rejections.jsonl"
    data_path = output / "pose_library_v1.jsonl"
    summary_path = output / "generation_summary.json"
    seed_path = output / "accepted_seed_lists.json"
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    cfg.sim.render_interval = 1_000_000
    env = None
    started_wall = time.perf_counter()
    accepted: list[dict] = []
    rejected_counts: Counter[str] = Counter()
    used_fingerprints: set[str] = set()
    attempt_counts = {split: 0 for split in SPLIT_COUNTS}
    accepted_seed_lists = {split: [] for split in SPLIT_COUNTS}

    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            base = env.unwrapped
            capsule = base.scene["capsule"]
            camera = base.scene["capsule_camera"]
            term = base.action_manager.get_term("parameterized_force")
            reference = reference_from_stage()
            if reference.geometry_sha256 != region["stomach_geometry_sha256"]:
                raise RuntimeError("live stomach geometry does not match the confirmed entry region")
            entry_triangles = np.asarray(region["selected_triangle_indices"], dtype=np.int64)
            surface = SurfaceNavigationMesh.from_reference(reference, INWARD_NORMAL_SIGN)
            camera_offset_local = np.asarray(camera.cfg.offset.pos, dtype=np.float64).reshape(3)
            if not np.isfinite(camera_offset_local).all():
                raise RuntimeError("camera local offset is non-finite")
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(capsule.root_view.prim_paths[0])
            from pxr import UsdPhysics

            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            if not rigid_api or (
                rigid_api.GetKinematicEnabledAttr() and bool(rigid_api.GetKinematicEnabledAttr().Get())
            ):
                raise RuntimeError("pose-library capsule must remain Dynamic")
            sim = base.sim
            sim.physics_manager.play()
            with rejection_path.open("w", encoding="utf-8") as rejection_stream:
                for split, target_count in SPLIT_COUNTS.items():
                    accepted_in_split = 0
                    maximum_attempts = target_count * args_cli.maximum_attempt_multiplier
                    while accepted_in_split < target_count and attempt_counts[split] < maximum_attempts:
                        attempt_index = attempt_counts[split]
                        attempt_counts[split] += 1
                        seed = deterministic_candidate_seed(split, attempt_index)
                        candidate = sample_surface_pose(reference, entry_triangles, seed)
                        term.reset()
                        capsule.permanent_wrench_composer.reset()
                        _write_state(capsule, candidate.pose_world_xyzw, base.device)
                        sim.forward()
                        base.scene.update(0.0)
                        stable_steps = 0
                        stable = False
                        finite = True
                        final_pose = _pose(capsule)
                        final_velocity = _velocity(capsule)
                        for physics_step in range(MAX_SETTLE_STEPS):
                            if physics_step % PHYSICS_STEPS_PER_CONTROL == 0:
                                hold = torch.tensor(
                                    [[float(ParameterizedForceMode.HOLD), 0.5]],
                                    device=base.device,
                                    dtype=torch.float32,
                                )
                                base.action_manager.process_action(hold)
                            base.action_manager.apply_action()
                            base.scene.write_data_to_sim()
                            sim.step(render=False)
                            base.scene.update(PHYSICS_DT_S)
                            final_pose = _pose(capsule)
                            final_velocity = _velocity(capsule)
                            finite = bool(
                                np.isfinite(final_pose).all() and np.isfinite(final_velocity).all()
                            )
                            if not finite:
                                break
                            linear_speed = float(np.linalg.norm(final_velocity[:3]))
                            angular_speed = float(np.linalg.norm(final_velocity[3:]))
                            stable_steps = (
                                stable_steps + 1
                                if linear_speed <= MAX_LINEAR_SPEED_M_S
                                and angular_speed <= MAX_ANGULAR_SPEED_RAD_S
                                else 0
                            )
                            if stable_steps >= STABLE_STEPS:
                                stable = True
                                break

                        reason = None
                        if not finite:
                            reason = "non_finite_state"
                        elif not stable:
                            reason = "not_stable_within_2_seconds"
                        camera_center = _camera_center_world(capsule, camera) if finite else np.full(3, np.nan)
                        camera_hit = surface.closest_hit(camera_center) if finite else None
                        camera_signed_lumen_distance = (
                            float(np.dot(camera_center - camera_hit.point_world, camera_hit.normal_world))
                            if camera_hit is not None
                            else -math.inf
                        )
                        camera_inside = camera_signed_lumen_distance > CAMERA_LUMEN_SIDE_TOLERANCE_M
                        rotation = (
                            rotation_matrix_from_xyzw(final_pose[3:])
                            if finite
                            else np.full((3, 3), np.nan)
                        )
                        axis = rotation[:, 2]
                        axis_angle = unoriented_axis_angle_deg(axis) if finite else -math.inf
                        if reason is None and not camera_inside:
                            reason = "camera_not_in_lumen"
                        if reason is None and axis_angle < MIN_UNORIENTED_AXIS_ANGLE_DEG:
                            reason = "axis_angle_below_45_deg"
                        fingerprint = pose_fingerprint(final_pose) if finite else ""
                        if reason is None and fingerprint in used_fingerprints:
                            reason = "duplicate_final_pose"

                        if reason is not None:
                            rejected_counts[reason] += 1
                            rejection_stream.write(
                                json.dumps(
                                    {
                                        "split": split,
                                        "attempt_index": attempt_index,
                                        "candidate_seed": seed,
                                        "reason": reason,
                                        "initial_pose_world_xyzw": candidate.pose_world_xyzw.tolist(),
                                        "final_pose_world_xyzw": final_pose.tolist(),
                                        "stable_steps": stable_steps,
                                        "camera_signed_lumen_distance_m": camera_signed_lumen_distance,
                                        "unoriented_axis_angle_deg": axis_angle,
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            if sum(rejected_counts.values()) % 100 == 0:
                                print(
                                    "TASK009B_POSE_LIBRARY_REJECTIONS "
                                    + json.dumps(dict(sorted(rejected_counts.items())), sort_keys=True),
                                    flush=True,
                                )
                            continue

                        pose_id = f"{split}-{accepted_in_split:04d}"
                        record = {
                            "schema": POSE_LIBRARY_SCHEMA,
                            "version": POSE_LIBRARY_VERSION,
                            "pose_id": pose_id,
                            "split": split,
                            "candidate_seed": seed,
                            "attempt_index": attempt_index,
                            "entry_region_config_sha256": region["config_sha256"],
                            "stomach_geometry_sha256": reference.geometry_sha256,
                            "surface_triangle_index": candidate.triangle_index,
                            "surface_point_world_m": candidate.surface_point_world_m.tolist(),
                            "surface_normal_world": candidate.surface_normal_world.tolist(),
                            "surface_barycentric": candidate.barycentric.tolist(),
                            "tangent_azimuth_rad": candidate.tangent_azimuth_rad,
                            "camera_end_sign": candidate.camera_end_sign,
                            "roll_rad": candidate.roll_rad,
                            "initial_pose_world_xyzw": candidate.pose_world_xyzw.tolist(),
                            "pose_world_xyzw": final_pose.tolist(),
                            "stable": True,
                            "settle_elapsed_s": (physics_step + 1) * PHYSICS_DT_S,
                            "stable_duration_s": stable_steps * PHYSICS_DT_S,
                            "final_linear_speed_m_s": float(np.linalg.norm(final_velocity[:3])),
                            "final_angular_speed_rad_s": float(np.linalg.norm(final_velocity[3:])),
                            "camera_center_world_m": camera_center.tolist(),
                            "camera_inside_lumen": camera_inside,
                            "camera_signed_lumen_distance_m": camera_signed_lumen_distance,
                            "camera_nearest_triangle_index": int(camera_hit.triangle_id),
                            "capsule_axis_world": axis.tolist(),
                            "unoriented_axis_angle_deg": axis_angle,
                            "pose_fingerprint_sha256": fingerprint,
                        }
                        if not stable_record_is_valid(record):
                            raise RuntimeError("internal error: accepted pose does not satisfy the frozen gate")
                        used_fingerprints.add(fingerprint)
                        accepted.append(record)
                        accepted_seed_lists[split].append(seed)
                        accepted_in_split += 1
                        if accepted_in_split % 25 == 0 or accepted_in_split == target_count:
                            print(
                                "TASK009B_POSE_LIBRARY_PROGRESS "
                                + json.dumps(
                                    {
                                        "split": split,
                                        "accepted": accepted_in_split,
                                        "target": target_count,
                                        "attempts": attempt_counts[split],
                                        "rejected_total": int(sum(rejected_counts.values())),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                    if accepted_in_split != target_count:
                        raise RuntimeError(
                            f"{split} produced {accepted_in_split}/{target_count} valid states after "
                            f"{attempt_counts[split]} attempts"
                        )

            data_bytes, data_sha = write_jsonl(data_path, accepted)
            _write_json(seed_path, accepted_seed_lists)
            split_ids = {
                split: [record["pose_id"] for record in accepted if record["split"] == split]
                for split in SPLIT_COUNTS
            }
            reload_ids = _fixed_live_reload_ids(accepted)
            manifest = {
                "schema": POSE_LIBRARY_MANIFEST_SCHEMA,
                "version": POSE_LIBRARY_VERSION,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "task_id": args_cli.task,
                "entry_anchor_config_sha256": anchor["config_sha256"],
                "entry_region_config_sha256": region["config_sha256"],
                "stomach_geometry_sha256": reference.geometry_sha256,
                "data_path": str(data_path.resolve()),
                "data_bytes": data_bytes,
                "data_sha256": data_sha,
                "rejection_log_path": str(rejection_path.resolve()),
                "rejection_log_bytes": rejection_path.stat().st_size,
                "rejection_log_sha256": file_sha256(rejection_path),
                "accepted_seed_list_path": str(seed_path.resolve()),
                "accepted_seed_list_sha256": file_sha256(seed_path),
                "split_counts": SPLIT_COUNTS,
                "split_base_seeds": SPLIT_BASE_SEEDS,
                "split_pose_ids": split_ids,
                "accepted_candidate_seeds": accepted_seed_lists,
                "fixed_live_reload_pose_ids": reload_ids,
                "total_count": len(accepted),
                "unique_pose_fingerprint_count": len(used_fingerprints),
                "attempt_counts": attempt_counts,
                "rejection_counts": dict(sorted(rejected_counts.items())),
                "physics": {
                    "physics_hz": PHYSICS_HZ,
                    "maximum_settle_s": MAX_SETTLE_STEPS * PHYSICS_DT_S,
                    "required_stable_s": STABLE_STEPS * PHYSICS_DT_S,
                    "linear_speed_limit_m_s": MAX_LINEAR_SPEED_M_S,
                    "angular_speed_limit_rad_s": MAX_ANGULAR_SPEED_RAD_S,
                    "minimum_unoriented_axis_angle_deg": MIN_UNORIENTED_AXIS_ANGLE_DEG,
                    "inward_normal_sign": INWARD_NORMAL_SIGN,
                    "active_force": "HOLD only",
                },
                "live_reload_validation": {"status": "pending", "count_per_split": 20},
            }
            manifest["config_sha256"] = manifest_hash(manifest)
            _write_json(args_cli.manifest_path, manifest)
            summary = {
                "status": "generated_pending_live_reload",
                "manifest_path": str(args_cli.manifest_path.resolve()),
                "manifest_config_sha256": manifest["config_sha256"],
                "data_path": str(data_path.resolve()),
                "data_bytes": data_bytes,
                "data_sha256": data_sha,
                "counts": SPLIT_COUNTS,
                "attempt_counts": attempt_counts,
                "rejection_counts": dict(sorted(rejected_counts.items())),
                "elapsed_wall_s": time.perf_counter() - started_wall,
            }
            _write_json(summary_path, summary)
            print("TASK009B_POSE_LIBRARY_GENERATED " + json.dumps(summary, sort_keys=True), flush=True)
        finally:
            if env is not None:
                env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
