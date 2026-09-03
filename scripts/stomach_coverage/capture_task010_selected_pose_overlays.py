#!/usr/bin/env python3
"""Re-run one frozen TASK-010 pose and export 15 s coverage/path overlays."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

SNAPSHOT_STEPS = tuple(range(0, 1201, 150))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--update", type=int, choices=(750, 1000), required=True)
    parser.add_argument("--pose-id", required=True)
    parser.add_argument("--historical-final-coverage", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(), AppLauncher


def selected_display_override(reference, unreachable_record: dict):
    vertices = np.asarray(reference.vertices_world, dtype=np.float64)
    triangles = np.asarray(reference.triangles, dtype=np.int64)
    triangle_vertices = vertices[triangles]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    regions = []
    for kind, records in (("seed", unreachable_record["seeds"]), ("box", unreachable_record["boxes"])):
        for record in records:
            faces = np.asarray(record["selected_triangle_indices"], dtype=np.int64)
            regions.append(
                {
                    "kind": kind,
                    "id": int(record[f"{kind}_id"]),
                    "faces": faces,
                    "area_m2": float(triangle_areas[faces].sum()),
                }
            )
    if len(regions) < 2:
        raise RuntimeError("display override requires at least two excluded regions")
    selected = min(regions, key=lambda item: (item["area_m2"], item["kind"], item["id"]))
    vertex_mask = np.zeros(len(vertices), dtype=np.bool_)
    vertex_mask[np.unique(triangles[selected["faces"]].reshape(-1))] = True
    audit = {
        "selection_rule": "smallest source excluded region by triangle surface area",
        "selected_kind": selected["kind"],
        "selected_id": selected["id"],
        "selected_area_m2": selected["area_m2"],
        "selected_triangle_count": int(len(selected["faces"])),
        "selected_vertex_count": int(vertex_mask.sum()),
        "all_regions": [
            {
                "kind": item["kind"],
                "id": item["id"],
                "area_m2": item["area_m2"],
                "triangle_count": int(len(item["faces"])),
            }
            for item in regions
        ],
        "render_only": True,
        "coverage_metric_unchanged": True,
    }
    return vertex_mask, audit


def contact_record(base, row_index: int) -> dict:
    try:
        values = base.scene["capsule_contact"].data.net_forces_w.torch[int(row_index)]
        norms = values.detach().to(dtype=values.dtype).norm(dim=-1).cpu().numpy()
        return {
            "contact_body_count": int(np.count_nonzero(norms > 1.0e-6)),
            "maximum_net_contact_force_n": float(norms.max(initial=0.0)),
        }
    except Exception:
        return {"contact_body_count": None, "maximum_net_contact_force_n": None}


def main() -> None:
    args, AppLauncher = parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from robotarm_magnetic_lab.coverage.entry_pose_library import file_sha256 as pose_file_sha256
    from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
    from robotarm_magnetic_lab.runtime.task010_config import load_task010_config
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
        ParameterizedForceMode,
    )
    from robotarm_magnetic_lab.ui.coverage_view import export_coverage_projection

    frozen = load_task010_config(args.config)
    frozen_json = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if checkpoint.get("config_hash") != frozen.config_sha256:
        raise RuntimeError("checkpoint/config mismatch")
    actor = Task010Actor().to(args.device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    checkpoint_hash = pose_file_sha256(args.checkpoint)

    validation_pose_ids = tuple(frozen_json["validation"]["pose_ids"])
    if args.pose_id not in validation_pose_ids:
        raise ValueError(f"pose is outside the frozen validation set: {args.pose_id}")
    pose_index = validation_pose_ids.index(args.pose_id)
    batch_index = pose_index // 12
    target_row = pose_index % 12
    batch = validation_pose_ids[batch_index * 12 : (batch_index + 1) * 12]
    padded_batch = batch + batch[:1] * (12 - len(batch))
    reset_seed = int(frozen_json["training"]["seed"]) + batch_index

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "overlays_15s"
    snapshots.mkdir()
    telemetry_path = output / "boundary_telemetry_10hz.jsonl"
    selected_source = {
        "seed": int(args.seed),
        "update": int(args.update),
        "pose_id": str(args.pose_id),
        "historical_final_coverage": float(args.historical_final_coverage),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(args.config.resolve()),
        "config_sha256": frozen.config_sha256,
        "validation_batch_index": batch_index,
        "target_row": target_row,
        "reset_seed": reset_seed,
        "validation_batch_pose_ids": list(padded_batch),
    }

    # Match the original validator's 12-row layout and padding exactly.  The
    # rendered Actor input changes if the selected pose is run as a lone clone.
    cfg = parse_env_cfg(frozen.task_id, device=args.device, num_envs=12)
    cfg.pose_split = "validation"
    cfg.explicit_pose_ids = padded_batch
    rows = []
    snapshot_records = []
    display_override_audit = None
    started = datetime.now(timezone.utc)

    with launch_simulation(app, cfg):
        env = gym.make(frozen.task_id, cfg=cfg).unwrapped
        try:
            observations, extras = env.reset(seed=reset_seed)
            actual_pose = tuple(extras["task009d0_reset"]["pose_ids"])
            if actual_pose != padded_batch or actual_pose[target_row] != args.pose_id:
                raise RuntimeError(f"reset returned the wrong pose: {actual_pose}")
            actor.reset()
            runtime = env._task009d0_coverage_runtime
            env_origin = env.scene.env_origins[target_row].detach().cpu().numpy().astype(np.float64)
            unreachable_path = ROOT / "configs/task009b/unreachable_region_v1.json"
            unreachable_record = json.loads(unreachable_path.read_text(encoding="utf-8"))
            small_override, display_override_audit = selected_display_override(
                runtime.reference_local, unreachable_record
            )
            raw_target = runtime.raw_accumulator.weights.detach().cpu().numpy() > 0.0
            reachable_target = runtime.reachable_vertex_mask.detach().cpu().numpy()
            excluded = raw_target & ~reachable_target
            display_excluded = excluded & ~small_override
            raw_weights = runtime.raw_accumulator.weights.detach().cpu().numpy()
            trajectory_local = []

            def state_from_live():
                pose = env.scene["capsule"].data.root_pose_w.torch[target_row].detach().cpu().numpy().astype(np.float64)
                velocity = env.scene["capsule"].data.root_com_vel_w.torch[target_row].detach().cpu().numpy().astype(np.float64)
                return pose, velocity

            def actor_observation_sha256(value) -> str:
                array = value.detach().cpu().contiguous().numpy()
                digest = hashlib.sha256()
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes())
                return digest.hexdigest()

            def append_row(step, action, reward, terminated, truncated, pose, velocity, reach_mask, raw_mask, frame_id, actor_observation):
                ratio = 0.0
                if action is not None:
                    commanded_mode = ParameterizedForceMode(int(action[0].item())).name
                    range_name = (
                        "MOVE" if commanded_mode.startswith("MOVE")
                        else "VIEW" if commanded_mode.startswith("VIEW")
                        else commanded_mode
                    )
                    lower, upper = frozen_json["action"]["force_ratio_mg"][range_name]
                    ratio = float(lower + (upper - lower) * float(action[1].item()))
                reach_fraction = float(
                    np.dot(reach_mask.astype(np.float64), runtime.reachable_accumulator.weights.detach().cpu().numpy())
                    / float(runtime.reachable_accumulator.total_area_m2)
                )
                raw_fraction = float(np.dot(raw_mask.astype(np.float64), raw_weights) / raw_weights.sum())
                mode_id = None if action is None else int(action[0].item())
                row = {
                    "schema": "robotarm_magnetic_lab.task010_selected_pose_boundary_telemetry",
                    "seed": int(args.seed),
                    "update": int(args.update),
                    "pose_id": args.pose_id,
                    "boundary_index": int(step),
                    "sim_time_s": float(step) / 10.0,
                    "physics_substeps": 0 if step == 0 else 24,
                    "mode_id": mode_id,
                    "mode_name": "C0" if mode_id is None else ParameterizedForceMode(mode_id).name,
                    "alpha": None if action is None else float(action[1].item()),
                    "force_ratio_mg": ratio,
                    "reward": None if reward is None else float(reward),
                    "actor_rgb_frame": int(frame_id),
                    "actor_observation_sha256": actor_observation_sha256(actor_observation),
                    "reachable_coverage_fraction": reach_fraction,
                    "raw_coverage_fraction": raw_fraction,
                    "capsule_position_world_m": pose[:3].tolist(),
                    "capsule_quaternion_xyzw": pose[3:].tolist(),
                    "capsule_linear_velocity_world_m_s": velocity[:3].tolist(),
                    "capsule_angular_velocity_world_rad_s": velocity[3:].tolist(),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "finite": bool(np.isfinite(pose).all() and np.isfinite(velocity).all()),
                    **(
                        {"contact_body_count": None, "maximum_net_contact_force_n": None}
                        if terminated or truncated
                        else contact_record(env, target_row)
                    ),
                }
                rows.append(row)
                with telemetry_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                return row

            def save_snapshot(step, row, reach_mask, raw_mask, pose):
                display_covered = reach_mask | small_override
                trajectory = np.asarray(trajectory_local, dtype=np.float64).reshape(-1, 3)
                stem = f"t{step // 10:03d}s"
                png_path = snapshots / f"{stem}_coverage_path_overlay.png"
                projection = export_coverage_projection(
                    png_path,
                    runtime.reference_local.vertices_world,
                    display_covered,
                    pose[:3] - env_origin,
                    trajectory,
                    row["reachable_coverage_fraction"],
                    row["sim_time_s"],
                    excluded_mask=display_excluded,
                    raw_coverage_fraction=row["raw_coverage_fraction"],
                )
                arrays_path = snapshots / f"{stem}_audit_arrays.npz"
                np.savez_compressed(
                    arrays_path,
                    reachable_coverage_mask=reach_mask,
                    raw_coverage_mask=raw_mask,
                    display_covered_mask=display_covered,
                    display_excluded_mask=display_excluded,
                    small_excluded_rendered_as_covered_mask=small_override,
                    trajectory_local_m=trajectory,
                    capsule_position_local_m=pose[:3] - env_origin,
                )
                metadata = {
                    "schema": "robotarm_magnetic_lab.task010_selected_pose_overlay",
                    "seed": int(args.seed),
                    "update": int(args.update),
                    "pose_id": args.pose_id,
                    "boundary_index": int(step),
                    "sim_time_s": row["sim_time_s"],
                    "reachable_coverage_fraction": row["reachable_coverage_fraction"],
                    "raw_coverage_fraction": row["raw_coverage_fraction"],
                    "trajectory_points": int(len(trajectory)),
                    "display_override": display_override_audit,
                    "png": str(png_path),
                    "arrays": str(arrays_path),
                    **projection,
                }
                metadata_path = snapshots / f"{stem}_metadata.json"
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                snapshot_records.append(metadata)
                print(
                    "TASK010_SELECTED_POSE_SNAPSHOT "
                    + json.dumps(
                        {
                            "seed": args.seed,
                            "update": args.update,
                            "pose_id": args.pose_id,
                            "time_s": row["sim_time_s"],
                            "coverage": row["reachable_coverage_fraction"],
                            "path": str(png_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            initial_pose, initial_velocity = state_from_live()
            initial_reach = runtime.reachable_accumulator.mask[target_row].detach().cpu().numpy().copy()
            initial_raw = runtime.raw_accumulator.mask[target_row].detach().cpu().numpy().copy()
            initial_frame = int(runtime.latest_update.frame_ids[target_row].item())
            initial_actor_observation = observations["policy"][target_row]
            trajectory_local.append(initial_pose[:3] - env_origin)
            row = append_row(
                0, None, None, False, False, initial_pose, initial_velocity,
                initial_reach, initial_raw, initial_frame, initial_actor_observation,
            )
            save_snapshot(0, row, initial_reach, initial_raw, initial_pose)

            for step in range(1, 1201):
                with torch.no_grad():
                    action = actor(observations["policy"], stochastic_output=False)
                observations, reward, terminated, truncated, step_extras = env.step(action)
                terminal = bool((terminated[target_row] | truncated[target_row]).item())
                if terminal:
                    audit = step_extras["task009d0_terminal_audit"]
                    pose = audit["root_pose"][target_row].detach().cpu().numpy().astype(np.float64)
                    velocity = audit["root_velocity"][target_row].detach().cpu().numpy().astype(np.float64)
                    reach_mask = audit["reachable_masks"][target_row].detach().cpu().numpy().copy()
                    raw_mask = audit["raw_masks"][target_row].detach().cpu().numpy().copy()
                    frame_id = int(audit["frame_ids"][target_row].item())
                    actor_observation = step_extras["terminal_observation"]["policy"][target_row]
                else:
                    pose, velocity = state_from_live()
                    reach_mask = runtime.reachable_accumulator.mask[target_row].detach().cpu().numpy().copy()
                    raw_mask = runtime.raw_accumulator.mask[target_row].detach().cpu().numpy().copy()
                    frame_id = int(runtime.latest_update.frame_ids[target_row].item())
                    actor_observation = observations["policy"][target_row]
                trajectory_local.append(pose[:3] - env_origin)
                row = append_row(
                    step, action[target_row], reward[target_row].item(),
                    terminated[target_row].item(), truncated[target_row].item(),
                    pose, velocity, reach_mask, raw_mask, frame_id, actor_observation,
                )
                if step in SNAPSHOT_STEPS:
                    save_snapshot(step, row, reach_mask, raw_mask, pose)
                if terminal and step != 1200:
                    raise RuntimeError(f"episode terminated early at step {step}")
        finally:
            env.close()

    completed = datetime.now(timezone.utc)
    if len(rows) != 1201 or len(snapshot_records) != len(SNAPSHOT_STEPS):
        raise RuntimeError("incomplete selected-pose acquisition")
    coverage = np.asarray([row["reachable_coverage_fraction"] for row in rows])
    if not np.isfinite(coverage).all() or np.any(np.diff(coverage) < -1.0e-12):
        raise RuntimeError("selected-pose coverage is invalid")
    if any(not row["finite"] for row in rows):
        raise RuntimeError("selected-pose telemetry contains non-finite state")

    summary = {
        "schema": "robotarm_magnetic_lab.task010_selected_pose_overlay_summary",
        "status": "complete",
        "source": selected_source,
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "wall_time_s": (completed - started).total_seconds(),
        "boundary_count": len(rows),
        "frequency_hz": 10,
        "duration_s": 120,
        "snapshot_times_s": [step // 10 for step in SNAPSHOT_STEPS],
        "snapshot_count": len(snapshot_records),
        "C0": float(coverage[0]),
        "C120": float(coverage[-1]),
        "coverage_monotonic": True,
        "all_state_finite": True,
        "display_override": display_override_audit,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = [path for path in output.rglob("*") if path.is_file()]
    manifest = {
        "schema": "robotarm_magnetic_lab.task010_selected_pose_overlay_manifest",
        "source": selected_source,
        "artifacts": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(artifacts)
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("TASK010_SELECTED_POSE_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    app.close()


if __name__ == "__main__":
    main()
