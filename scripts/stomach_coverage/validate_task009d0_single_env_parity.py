#!/usr/bin/env python3
"""Gate 2: prove one-row vector force/coverage parity with scalar oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

TASK_ID = "Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0"
POSE_IDS = (
    "validation-0006",
    "validation-0011",
    "validation-0015",
    "validation-0017",
    "validation-0019",
)


def _equal_mask(row: dict, prefix: str) -> bool:
    scalar_key = f"scalar_{prefix}_mask"
    vector_key = f"vector_{prefix}_mask"
    if scalar_key in row:
        return np.array_equal(
            np.asarray(row[scalar_key], dtype=np.bool_),
            np.asarray(row[vector_key], dtype=np.bool_),
        )
    return row[f"scalar_{prefix}_mask_sha256"] == row[f"vector_{prefix}_mask_sha256"]


def validate_parity_records(records: list[dict], expected_count: int) -> dict:
    if len(records) != int(expected_count):
        raise ValueError("parity boundary count mismatch")
    maximum_area_error = 0.0
    for row in records:
        if int(row["physics_substeps"]) != 24:
            raise ValueError("every boundary must contain exactly 24 substeps")
        if int(row["scalar_frame_id"]) != int(row["vector_frame_id"]):
            raise ValueError("scalar/vector frame IDs differ")
        for endpoint in ("camera", "other"):
            if not np.allclose(
                row[f"scalar_{endpoint}_force"],
                row[f"vector_{endpoint}_force"],
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise ValueError(f"{endpoint} force parity failed")
        if not _equal_mask(row, "current"):
            raise ValueError("current mask parity failed")
        if not _equal_mask(row, "cumulative"):
            raise ValueError("cumulative mask parity failed")
        maximum_area_error = max(maximum_area_error, abs(float(row["area_error_m2"])))
        if abs(float(row["area_error_m2"])) > 1.0e-12:
            raise ValueError("float64 area parity failed")
        if not bool(row["finite"]):
            raise ValueError("non-finite parity state")
    return {
        "status": "pass",
        "boundary_count": len(records),
        "maximum_area_error_m2": maximum_area_error,
    }


def _mask_hash(mask) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.bool_).tobytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def main() -> None:
    from isaaclab.app import AppLauncher

    # Isaac Lab 3.0 reserves the legacy --headless name but its current parser
    # exposes visualizer=[] instead. Accept the contract command and translate it.
    requested_headless = "--headless" in sys.argv
    if requested_headless:
        sys.argv.remove("--headless")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_directory", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, visualizer=[])
    args = parser.parse_args()
    args.enable_cameras = True
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    import gymnasium as gym
    import torch

    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab.utils import math as math_utils
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.coverage.accumulator import CoverageAccumulator
    from robotarm_magnetic_lab.coverage.visibility import (
        camera_facing_first_hits,
        candidate_vertices,
        visible_from_first_hits,
    )
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
        parameterized_endpoint_forces,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(TASK_ID, device=args.device, num_envs=1)
    cfg.pose_split = "validation"
    cfg.explicit_pose_ids = (POSE_IDS[0],)
    records = []
    pose_summaries = []
    try:
        with launch_simulation(simulation_app, cfg):
            env = gym.make(TASK_ID, cfg=cfg).unwrapped
            sequence = [(mode, alpha) for mode in range(6) for alpha in (0.0, 0.5, 1.0)]
            for pose_id in POSE_IDS:
                env.cfg.explicit_pose_ids = (pose_id,)
                observation, extras = env.reset()
                runtime = env._task009d0_coverage_runtime
                raw_weights = runtime.raw_accumulator.weights.detach().cpu().numpy()
                reach_weights = runtime.reachable_accumulator.weights.detach().cpu().numpy()
                scalar_raw = CoverageAccumulator(len(raw_weights), raw_weights)
                scalar_reach = CoverageAccumulator(len(reach_weights), reach_weights)

                def scalar_update(frame_id: int):
                    center_w = runtime.camera.data.pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
                    center = center_w - runtime.env_origins[0].detach().cpu().numpy()
                    quat_t = runtime.camera.data.quat_w_ros.torch[0:1].to(dtype=torch.float64)
                    local_z = torch.tensor([[0.0, 0.0, 1.0]], device=env.device, dtype=torch.float64)
                    axis = math_utils.quat_apply(quat_t, local_z)[0].detach().cpu().numpy()
                    ids, distances = candidate_vertices(runtime.reference_local.vertices_world, center, axis)
                    candidate_mask = torch.zeros(
                        (1, len(runtime.vertices_local)), dtype=torch.bool, device=env.device
                    )
                    candidate_mask[0, torch.as_tensor(ids, device=env.device)] = True
                    all_hit_d, all_hit_f = runtime.raycaster.query(
                        torch.as_tensor(center, device=env.device, dtype=torch.float64).reshape(1, 3),
                        candidate_mask,
                        runtime.vertices_local,
                    )
                    hit_d = all_hit_d[0, torch.as_tensor(ids, device=env.device)].detach().cpu().numpy()
                    hit_f = all_hit_f[0, torch.as_tensor(ids, device=env.device)].detach().cpu().numpy()
                    visible = visible_from_first_hits(ids, distances, hit_d, hit_f, runtime.reference_local.incident_triangles)
                    visible &= camera_facing_first_hits(
                        center,
                        runtime.reference_local.vertices_world[ids],
                        hit_f,
                        runtime.reference_local,
                        normal_sign=-1,
                    )
                    raw_mask = np.zeros(len(raw_weights), dtype=np.bool_)
                    raw_mask[ids[visible]] = True
                    reach_mask = raw_mask & (reach_weights > 0.0)
                    raw_result = scalar_raw.update(frame_id, np.flatnonzero(raw_mask))
                    reach_result = scalar_reach.update(frame_id, np.flatnonzero(reach_mask))
                    return raw_mask, reach_mask, raw_result, reach_result, axis, {
                        "center": center,
                        "candidate_ids": ids,
                        "target_distances": distances,
                        "hit_distances": hit_d,
                        "hit_faces": hit_f,
                    }

                initial_frame = int(runtime.latest_update.frame_ids[0].item())
                _, initial_reach, _, initial_result, _, initial_debug = scalar_update(initial_frame)
                vector_initial = runtime.latest_reachable_visible_mask[0].cpu().numpy()
                if not np.array_equal(initial_reach, vector_initial):
                    differing = np.flatnonzero(initial_reach != vector_initial)
                    from robotarm_magnetic_lab.coverage.batched_visibility import batched_candidate_mask
                    centers_t = torch.as_tensor(initial_debug["center"], device=env.device, dtype=torch.float64).reshape(1, 3)
                    axes_t = torch.as_tensor(
                        math_utils.quat_apply(
                            runtime.camera.data.quat_w_ros.torch[0:1].to(dtype=torch.float64),
                            torch.tensor([[0.0, 0.0, 1.0]], device=env.device, dtype=torch.float64),
                        ),
                        device=env.device,
                    )
                    batched_candidates, batched_distances = batched_candidate_mask(
                        runtime.vertices_local, centers_t, axes_t
                    )
                    candidate_ids = initial_debug["candidate_ids"]
                    scalar_lookup = {int(vertex): ray for ray, vertex in enumerate(candidate_ids)}
                    detail = []
                    for vertex in differing[:20]:
                        ray = scalar_lookup.get(int(vertex))
                        detail.append({
                            "vertex": int(vertex),
                            "scalar_candidate": ray is not None,
                            "vector_candidate": bool(batched_candidates[0, vertex].item()),
                            "target_distance_scalar": None if ray is None else float(initial_debug["target_distances"][ray]),
                            "target_distance_vector": float(batched_distances[0, vertex].item()),
                            "scalar_hit_distance": None if ray is None else float(initial_debug["hit_distances"][ray]),
                            "scalar_hit_face": None if ray is None else int(initial_debug["hit_faces"][ray]),
                        })
                    raise RuntimeError(
                        f"{pose_id}: C0 current mask parity failed: "
                        f"scalar={np.count_nonzero(initial_reach)}, "
                        f"vector={np.count_nonzero(vector_initial)}, "
                        f"differing={len(differing)}, detail={json.dumps(detail, sort_keys=True)}"
                    )
                if not np.array_equal(scalar_reach.mask, runtime.reachable_accumulator.mask[0].cpu().numpy()):
                    raise RuntimeError(f"{pose_id}: C0 cumulative mask parity failed")
                previous_frame = initial_frame
                pose_start = len(records)
                for boundary in range(60):
                    mode, alpha = sequence[boundary % len(sequence)]
                    action = torch.tensor([[float(mode), alpha]], device=env.device)
                    observation, reward, terminated, truncated, extras = env.step(action)
                    if bool(terminated[0].item()) or bool(truncated[0].item()):
                        raise RuntimeError(f"{pose_id}: early termination")
                    vector = runtime.latest_update
                    frame = int(vector.frame_ids[0].item())
                    if frame != previous_frame + 1:
                        raise RuntimeError(f"{pose_id}: RGB did not advance exactly once")
                    previous_frame = frame
                    raw_mask, reach_mask, scalar_raw_result, scalar_result, axis, _ = scalar_update(frame)
                    term = env.action_manager.get_term("parameterized_force")
                    scalar_force = parameterized_endpoint_forces(
                        mode,
                        alpha,
                        mass_kg=float(term.mass_kg[0].item()),
                        camera_axis_world=term.last_camera_positions_world[0].cpu().numpy()
                        - term.last_other_positions_world[0].cpu().numpy(),
                        config=term.config,
                    )
                    vector_current = runtime.latest_reachable_visible_mask[0].cpu().numpy()
                    vector_cumulative = runtime.reachable_accumulator.mask[0].cpu().numpy()
                    area_error = abs(
                        scalar_result.cumulative_area_m2
                        - float(vector.reachable.cumulative_area_m2[0].item())
                    )
                    state = env.scene["capsule"].data.root_state_w.torch[0]
                    records.append({
                        "pose_id": pose_id,
                        "boundary": boundary + 1,
                        "physics_substeps": int(env.cfg.decimation),
                        "scalar_frame_id": frame,
                        "vector_frame_id": frame,
                        "scalar_camera_force": scalar_force.camera_force_world.tolist(),
                        "vector_camera_force": term.last_camera_forces_world[0].cpu().tolist(),
                        "scalar_other_force": scalar_force.other_force_world.tolist(),
                        "vector_other_force": term.last_other_forces_world[0].cpu().tolist(),
                        "scalar_current_mask_sha256": _mask_hash(reach_mask),
                        "vector_current_mask_sha256": _mask_hash(vector_current),
                        "scalar_cumulative_mask_sha256": _mask_hash(scalar_reach.mask),
                        "vector_cumulative_mask_sha256": _mask_hash(vector_cumulative),
                        "area_error_m2": area_error,
                        "finite": bool(torch.isfinite(state).all().item()),
                    })
                pose_records = records[pose_start:]
                validate_parity_records(pose_records, 60)
                pose_summaries.append({
                    "pose_id": pose_id,
                    "boundaries": 60,
                    "C0": float(initial_result.coverage_fraction),
                    "status": "pass",
                })
            summary = validate_parity_records(records, 300)
            manifest = {
                "status": "pass",
                "task_id": TASK_ID,
                "branch": "feature/TASK-009D0-vectorized-training-infrastructure",
                "commit": _git_head(),
                "device": str(env.device),
                "poses": pose_summaries,
                "summary": summary,
                "records": records,
            }
            target = args.output_directory / "task009d0_gate2_single_env_parity.json"
            target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("TASK009D0_GATE2", json.dumps({"status": "pass", "path": str(target.resolve()), **summary}, sort_keys=True))
            env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
