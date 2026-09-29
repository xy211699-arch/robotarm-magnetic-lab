#!/usr/bin/env python3
"""Calibrate a review-only two-tube exclusion and draw the new stomach ROI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robotarm_magnetic_lab"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

from robotarm_magnetic_lab.coverage.area_weights import target_vertex_area_weights, weights_sha256
from robotarm_magnetic_lab.coverage.new_stomach_tubes import candidate_tube_mask
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh
from robotarm_magnetic_lab.geometry.new_stomach_runtime import NEW_STOMACH_WELD_TOLERANCE_M
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_env_cfg import (
    NEW_STOMACH_ASSET_USD_PATH,
    NEW_STOMACH_GEOMETRY,
)


COLLIDER_IN_ASSET = "/NewStomach/geometry/CollisionMesh"
COLLIDER_IN_SCENE = "/World/envs/env_0/Stomach/geometry/CollisionMesh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_confirmed_reference():
    asset_path = Path(NEW_STOMACH_ASSET_USD_PATH)
    if _sha256(asset_path) != NEW_STOMACH_GEOMETRY.asset_usd_sha256:
        raise RuntimeError("new-stomach asset differs from the confirmed scene")
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError("new-stomach USD did not open")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(COLLIDER_IN_ASSET))
    if not mesh or str(mesh.GetOrientationAttr().Get()) != "leftHanded":
        raise RuntimeError("confirmed inward-facing collision mesh is unavailable")
    local_to_asset = np.asarray(
        UsdGeom.Xformable(mesh.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    )
    asset_to_world = np.eye(4, dtype=np.float64)
    asset_to_world[:3, :3] = Rotation.from_quat(
        NEW_STOMACH_GEOMETRY.rotation_xyzw
    ).as_matrix().T
    asset_to_world[3, :3] = NEW_STOMACH_GEOMETRY.position_world_m
    mesh_input = MeshInput(
        prim_path=COLLIDER_IN_SCENE,
        vertices=np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64),
        face_vertex_counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64),
        face_vertex_indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64),
        world_transform=local_to_asset @ asset_to_world,
        orientation="leftHanded",
    )
    reference = preprocess_reference_mesh(
        [mesh_input], [COLLIDER_IN_SCENE],
        weld_tolerance_m=NEW_STOMACH_WELD_TOLERANCE_M,
    )
    actual_min = reference.vertices_world.min(axis=0)
    actual_max = reference.vertices_world.max(axis=0)
    if not (
        np.allclose(actual_min, NEW_STOMACH_GEOMETRY.world_bounds_min_m, atol=1.0e-6, rtol=0)
        and np.allclose(actual_max, NEW_STOMACH_GEOMETRY.world_bounds_max_m, atol=1.0e-6, rtol=0)
    ):
        raise RuntimeError("offline coverage geometry does not align with the accepted scene")
    return reference


def render_candidate(reference, candidate, output: Path, tube_length_m: float, radial_limit_m: float):
    centers = reference.vertices_world[reference.triangles].mean(axis=1)
    origin = np.asarray(NEW_STOMACH_GEOMETRY.world_bounds_min_m)
    points = (centers - origin) * 1000.0
    mouths = (np.asarray(NEW_STOMACH_GEOMETRY.opening_centers_world_m) - origin) * 1000.0
    directions = candidate.inward_directions * tube_length_m * 1000.0
    reachable = np.flatnonzero(candidate.labels == 0)[::5]
    tube_a = candidate.per_tube_face_indices[0]
    tube_b = candidate.per_tube_face_indices[1]
    green, red, orange = "#16a68c", "#e34842", "#f39c35"
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax3.scatter(*points[reachable].T, s=0.12, c=green, alpha=0.18, depthshade=False)
    for selected, color in ((tube_a, red), (tube_b, orange)):
        ax3.scatter(*points[selected].T, s=0.8, c=color, alpha=0.9, depthshade=False)
    for mouth, direction in zip(mouths, directions, strict=True):
        ax3.plot(*np.stack((mouth, mouth + direction), axis=1), color="black", lw=2.0)
    ax3.view_init(elev=23, azim=-65)
    ax3.set_title("3D surface (candidate mask)")
    ax3.set_xlabel("X from min (mm)")
    ax3.set_ylabel("Y from min (mm)")
    ax3.set_zlabel("Z from min (mm)")
    span = (np.asarray(NEW_STOMACH_GEOMETRY.world_bounds_max_m) - origin) * 1000.0
    ax3.set_box_aspect(span.copy())

    for subplot, dims, name, labels in (
        (2, (0, 1), "Top / XY", ("X from min (mm)", "Y from min (mm)")),
        (3, (0, 2), "Front / XZ", ("X from min (mm)", "Z from min (mm)")),
        (4, (1, 2), "Side / YZ", ("Y from min (mm)", "Z from min (mm)")),
    ):
        ax = fig.add_subplot(2, 2, subplot)
        ax.scatter(points[reachable, dims[0]], points[reachable, dims[1]],
                   s=0.12, c=green, alpha=0.16, linewidths=0, rasterized=True)
        for selected, color in ((tube_a, red), (tube_b, orange)):
            ax.scatter(points[selected, dims[0]], points[selected, dims[1]],
                       s=0.55, c=color, alpha=0.85, linewidths=0, rasterized=True)
        for index, (mouth, direction) in enumerate(zip(mouths, directions, strict=True)):
            ax.scatter(mouth[dims[0]], mouth[dims[1]], s=18, c="black", zorder=4)
            ax.annotate(
                f"Tube {chr(65 + index)}",
                xy=(mouth[dims[0]], mouth[dims[1]]),
                xytext=(mouth[dims[0]] + direction[dims[0]], mouth[dims[1]] + direction[dims[1]]),
                fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "black"},
            )
        ax.set_title(name)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xlim(0, span[dims[0]])
        ax.set_ylim(0, span[dims[1]])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.17)
    fraction = candidate.excluded_area_m2 / (
        candidate.excluded_area_m2 + candidate.reachable_area_m2
    )
    fig.suptitle(
        "New stomach coverage ROI: proposed tube exclusion\n"
        f"Tube depth {tube_length_m*1000:.0f} mm, radial limit {radial_limit_m*1000:.0f} mm; "
        f"excluded {100*fraction:.2f}% of surface area (review only)",
        fontsize=14,
    )
    fig.legend(
        handles=(
            Patch(color=green, label="Reachable candidate"),
            Patch(color=red, label="Excluded Tube A"),
            Patch(color=orange, label="Excluded Tube B"),
        ),
        loc="lower center", ncol=3, frameon=False, fontsize=10,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.09, top=0.89, hspace=0.23)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tube-length-mm", type=float, default=25.0)
    parser.add_argument("--radial-limit-mm", type=float, default=12.0)
    args = parser.parse_args()
    if not (5 <= args.tube_length_mm <= 60 and 9.5 <= args.radial_limit_mm <= 25):
        parser.error("tube length must be 5..60 mm and radial limit 9.5..25 mm")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = load_confirmed_reference()
    tube_length_m = args.tube_length_mm / 1000.0
    radial_limit_m = args.radial_limit_mm / 1000.0
    candidate = candidate_tube_mask(
        reference,
        NEW_STOMACH_GEOMETRY.opening_centers_world_m,
        NEW_STOMACH_GEOMETRY.opening_axes_world,
        tube_length_m=tube_length_m,
        radial_limit_m=radial_limit_m,
    )
    raw_weights = target_vertex_area_weights(reference)
    reachable_weights = target_vertex_area_weights(reference, candidate.reachable_face_indices)
    mask_path = output / "candidate_face_mask.npz"
    np.savez_compressed(
        mask_path,
        face_labels=candidate.labels,
        excluded_face_indices=candidate.excluded_face_indices,
        reachable_face_indices=candidate.reachable_face_indices,
        tube_a_face_indices=candidate.per_tube_face_indices[0],
        tube_b_face_indices=candidate.per_tube_face_indices[1],
    )
    figure_path = output / "reachable_vs_excluded_tubes.png"
    render_candidate(reference, candidate, figure_path, tube_length_m, radial_limit_m)
    total_area = candidate.excluded_area_m2 + candidate.reachable_area_m2
    summary = {
        "schema": "robotarm_magnetic_lab.new_stomach_tube_candidate.v1",
        "status": "needs_input",
        "not_active_coverage_target": True,
        "asset_usd_sha256": NEW_STOMACH_GEOMETRY.asset_usd_sha256,
        "orientation_config_sha256": NEW_STOMACH_GEOMETRY.config_sha256,
        "offline_reference_geometry_sha256": reference.geometry_sha256,
        "collision_mesh_path": COLLIDER_IN_SCENE,
        "vertex_count": len(reference.vertices_world),
        "triangle_count": len(reference.triangles),
        "tube_length_m": tube_length_m,
        "radial_limit_m": radial_limit_m,
        "mouth_buffer_m": 0.001,
        "opening_centers_world_m": NEW_STOMACH_GEOMETRY.opening_centers_world_m,
        "inward_directions_world": candidate.inward_directions.tolist(),
        "directional_support_plus_minus": candidate.directional_support.tolist(),
        "tube_face_counts": [len(group) for group in candidate.per_tube_face_indices],
        "tube_connected_components": candidate.per_tube_component_counts,
        "tube_area_m2": candidate.per_tube_area_m2,
        "excluded_face_count": len(candidate.excluded_face_indices),
        "reachable_face_count": len(candidate.reachable_face_indices),
        "excluded_area_m2": candidate.excluded_area_m2,
        "reachable_area_m2": candidate.reachable_area_m2,
        "total_area_m2": total_area,
        "excluded_area_fraction": candidate.excluded_area_m2 / total_area,
        "raw_vertex_weights_sha256": weights_sha256(raw_weights),
        "candidate_reachable_vertex_weights_sha256": weights_sha256(reachable_weights),
        "mask_file": {"path": str(mask_path), "bytes": mask_path.stat().st_size, "sha256": _sha256(mask_path)},
        "figure_file": {"path": str(figure_path), "bytes": figure_path.stat().st_size, "sha256": _sha256(figure_path)},
        "operator_review": "Confirm both entire cylindrical ducts are red/orange and the gastric cavity remains green.",
    }
    (output / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("NEW_STOMACH_TUBE_CANDIDATE", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
