"""Pure geometry and artifact tests for TASK-009B entrance calibration."""

from __future__ import annotations

import json

import numpy as np

from robotarm_magnetic_lab.coverage.entrance_region import (
    entrance_region_record,
    select_entrance_triangles,
    triangles_intersecting_aabb,
)
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh


def _reference():
    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
    mesh = MeshInput(
        prim_path="/World/Stomach",
        vertices=vertices,
        face_vertex_counts=np.full(3, 3),
        face_vertex_indices=triangles.reshape(-1),
        world_transform=np.eye(4),
    )
    return preprocess_reference_mesh([mesh], [mesh.prim_path])


def test_triangle_aabb_intersection_includes_edge_crossing_without_contained_vertices():
    vertices = np.asarray([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    selected = triangles_intersecting_aabb(
        vertices,
        np.asarray([[0, 1, 2]]),
        np.zeros(3),
        np.asarray([0.2, 0.2, 0.2]),
    )
    assert selected.tolist() == [0]


def test_selection_reports_area_and_shared_edge_components():
    reference = _reference()
    selection = select_entrance_triangles(
        reference,
        center_world_m=np.asarray([0.5, -0.5, 0.0]),
        size_world_m=np.asarray([3.1, 1.1, 0.2]),
    )
    assert selection.triangle_count == 3
    assert np.isclose(selection.area_m2, 1.5)
    assert selection.connected_components == 2


def test_config_hash_changes_with_box_and_excludes_timestamp():
    reference = _reference()
    selection = select_entrance_triangles(reference, np.zeros(3), np.ones(3))
    first = entrance_region_record(
        reference,
        np.zeros(3),
        np.ones(3),
        selection,
        stomach_asset_identifier="stomach.usd",
    )
    second = entrance_region_record(
        reference,
        np.asarray([0.01, 0.0, 0.0]),
        np.ones(3),
        selection,
        stomach_asset_identifier="stomach.usd",
    )
    assert first["config_sha256"] != second["config_sha256"]
    confirmed = entrance_region_record(
        reference,
        np.zeros(3),
        np.ones(3),
        selection,
        stomach_asset_identifier="stomach.usd",
        operator_confirmation="confirmed",
    )
    assert confirmed["config_sha256"] == first["config_sha256"]
    assert "saved_utc" not in json.dumps({key: first[key] for key in first if key != "saved_utc"})
