"""TASK-009B anchor/entry surface geometry contract tests."""

from __future__ import annotations

import numpy as np

from robotarm_magnetic_lab.coverage.entry_surface_region import (
    anchor_record,
    closest_point_on_triangle,
    geodesic_face_distances,
    nearest_surface_point,
    region_record,
    shared_edge_adjacency,
    surface_region_from_distances,
)
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh


def _reference():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.1, 0.1, 0.01], [0.2, 0.1, 0.01], [0.1, 0.2, 0.01],
        ]
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
    source = MeshInput(
        "/World/Stomach",
        vertices,
        np.full(3, 3),
        triangles.reshape(-1),
        np.eye(4),
    )
    return preprocess_reference_mesh([source], [source.prim_path])


def test_closest_point_uses_triangle_interior_not_nearest_vertex():
    triangle = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    closest = closest_point_on_triangle(np.asarray([0.25, 0.25, 0.2]), triangle)
    assert np.allclose(closest, [0.25, 0.25, 0.0])


def test_nearest_surface_point_and_geodesic_region_do_not_jump_spatial_gap():
    reference = _reference()
    closest = nearest_surface_point(reference, np.asarray([0.75, 0.75, 0.1]))
    assert closest.triangle_index in (0, 1)
    distances = geodesic_face_distances(shared_edge_adjacency(reference), closest.triangle_index)
    region = surface_region_from_distances(reference, distances, 10.0)
    assert set(region.triangle_indices.tolist()) == {0, 1}
    assert region.connected_components == 1
    assert np.isinf(distances[2])


def test_anchor_and_region_hashes_change_with_geometry_inputs():
    reference = _reference()
    pose = np.asarray([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    stable = {"duration_s": 0.25, "linear_speed_m_s": 0.001, "angular_speed_rad_s": 0.01}
    anchor = anchor_record(
        default_pose_xyzw=pose,
        release_pose_xyzw=pose,
        settled_pose_xyzw=pose,
        stable_detection=stable,
        stomach_geometry_sha256=reference.geometry_sha256,
        capsule_asset_identifier="scene.usda#target_magnet",
    )
    closest = nearest_surface_point(reference, pose[:3])
    distances = geodesic_face_distances(shared_edge_adjacency(reference), closest.triangle_index)
    first_region = surface_region_from_distances(reference, distances, 0.5)
    second_region = surface_region_from_distances(reference, distances, 2.0)
    first = region_record(
        anchor_config_sha256=anchor["config_sha256"],
        settled_pose_xyzw=pose,
        closest=closest,
        region=first_region,
        stomach_geometry_sha256=reference.geometry_sha256,
    )
    second = region_record(
        anchor_config_sha256=anchor["config_sha256"],
        settled_pose_xyzw=pose,
        closest=closest,
        region=second_region,
        stomach_geometry_sha256=reference.geometry_sha256,
    )
    assert first["config_sha256"] != second["config_sha256"]

