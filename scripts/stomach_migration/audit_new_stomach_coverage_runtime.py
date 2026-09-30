#!/usr/bin/env python3
"""Verify a review-only tube mask against the actual Isaac Lab Stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robotarm_magnetic_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
args.enable_cameras = False
launcher = AppLauncher(args)

import numpy as np
import omni.usd
from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv

from robotarm_magnetic_lab.coverage.new_stomach_tubes import (
    candidate_right_tube_mask,
    candidate_tube_mask,
)
from robotarm_magnetic_lab.geometry.new_stomach_runtime import NewStomachRuntimeGeometry
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_env_cfg import (
    NEW_STOMACH_GEOMETRY,
    RobotarmMagneticNewStomachLabEnvCfg,
)


def main() -> None:
    candidate_dir = args.candidate_dir.resolve()
    record = json.loads((candidate_dir / "candidate_summary.json").read_text())
    if (
        record.get("schema") != "robotarm_magnetic_lab.new_stomach_tube_candidate.v3"
        or record.get("status") != "needs_input"
        or not record.get("not_active_coverage_target")
    ):
        raise ValueError("only an unapproved, review-only tube candidate may be audited")
    mask_path = candidate_dir / "candidate_face_mask.npz"
    mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    if mask_sha != record["mask_file"]["sha256"]:
        raise ValueError("candidate mask SHA-256 mismatch")
    saved = np.load(mask_path)
    cfg = RobotarmMagneticNewStomachLabEnvCfg()
    cfg.sim.device = args.device
    cfg.observations.vision = None
    cfg.scene.capsule_camera = None
    cfg.scene.capsule_camera_preview = None
    env = None
    try:
        with launch_simulation(cfg, args):
            env = ManagerBasedRLEnv(cfg=cfg)
            geometry = NewStomachRuntimeGeometry.from_stage(
                omni.usd.get_context().get_stage(), NEW_STOMACH_GEOMETRY
            )
            candidate = candidate_tube_mask(
                geometry.reference,
                NEW_STOMACH_GEOMETRY.opening_centers_world_m,
                NEW_STOMACH_GEOMETRY.opening_axes_world,
                tube_length_m=float(record["tube_length_m"]),
                radial_limit_m=float(record["radial_limit_m"]),
                mouth_buffer_m=float(record["mouth_buffer_m"]),
            )
            right_tube = candidate_right_tube_mask(
                geometry.reference,
                cut_x_from_min_m=float(record["right_tube"]["cut_x_from_min_m"]),
                max_y_from_min_m=float(record["right_tube"]["max_y_from_min_m"]),
            )
            right_mask = np.zeros(len(candidate.labels), dtype=bool)
            right_mask[right_tube.face_indices] = True
            if not np.all(right_mask[candidate.per_tube_face_indices[0]]):
                raise RuntimeError("runtime right tube does not contain Tube A")
            if np.any(right_mask[candidate.per_tube_face_indices[1]]):
                raise RuntimeError("runtime right tube overlaps Tube B")
            labels = candidate.labels.copy()
            labels[right_mask & (labels == 0)] = 3
            different = int(np.count_nonzero(labels != saved["face_labels"]))
            if different or len(labels) != int(record["triangle_count"]):
                raise RuntimeError(f"candidate mask differs from actual Isaac Lab mesh: {different} faces")
            excluded_area = right_tube.area_m2 + candidate.per_tube_area_m2[1]
            if not np.isclose(excluded_area, record["excluded_area_m2"], atol=1.0e-9, rtol=0):
                raise RuntimeError("runtime excluded surface area differs from candidate")
            report = {
                "status": "passed",
                "candidate_status": "needs_input",
                "coverage_target_activated": False,
                "runtime_geometry_sha256": geometry.geometry_sha256,
                "collision_mesh_path": geometry.collision_mesh_path,
                "candidate_mask_sha256": mask_sha,
                "runtime_triangle_count": len(labels),
                "differing_face_labels": different,
                "excluded_face_count": int(np.count_nonzero(labels)),
                "excluded_area_m2": excluded_area,
                "tube_connected_components": candidate.per_tube_component_counts,
                "right_tube_full_face_count": len(right_tube.face_indices),
                "right_tube_added_face_count": int(np.count_nonzero(labels == 3)),
                "right_tube_connected_components": right_tube.connected_components,
                "right_tube_cut_x_from_min_m": right_tube.cut_x_from_min_m,
            }
            (candidate_dir / "runtime_mask_audit.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )
            print("NEW_STOMACH_RUNTIME_MASK_AUDIT", json.dumps(report, sort_keys=True), flush=True)
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
