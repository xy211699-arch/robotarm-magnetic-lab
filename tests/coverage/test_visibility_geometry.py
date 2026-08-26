"""Coverage candidate and first-hit visibility geometry tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh
from robotarm_magnetic_lab.coverage.visibility import (
    ScalarFirstHitRaycaster,
    camera_facing_first_hits,
    candidate_vertices,
    triangle_normals,
    visible_from_first_hits,
)


def test_exact_distance_and_cone_boundaries_are_inclusive():
    angle = math.radians(60.0)
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.07],
            [0.07 * math.sin(angle), 0.0, 0.07 * math.cos(angle)],
            [0.0, 0.0, 0.070001],
            [0.07 * math.sin(angle + 1.0e-4), 0.0, 0.07 * math.cos(angle + 1.0e-4)],
        ]
    )
    indices, distances = candidate_vertices(
        vertices,
        optical_center_world=np.zeros(3),
        optical_axis_world=np.asarray([0.0, 0.0, 1.0]),
    )
    assert indices.tolist() == [0, 1]
    np.testing.assert_allclose(distances, [0.07, 0.07], atol=1.0e-12)


def test_incident_first_hit_and_distance_tolerance():
    incident = ((0, 1), (1,), (2,))
    visible = visible_from_first_hits(
        candidate_indices=np.asarray([0, 1, 2]),
        vertex_distances_m=np.asarray([0.03, 0.03, 0.03]),
        hit_distances_m=np.asarray([0.0301, 0.02, 0.0301001]),
        hit_face_ids=np.asarray([1, 0, 2]),
        incident_triangles=incident,
    )
    assert visible.tolist() == [True, False, False]


def test_scalar_oracle_returns_nearest_face():
    # Near triangle z=0.02 is face 0, farther target triangle z=0.03 is face 1.
    points = np.asarray(
        [
            [-0.01, -0.01, 0.02],
            [0.01, -0.01, 0.02],
            [0.0, 0.01, 0.02],
            [-0.01, -0.01, 0.03],
            [0.01, -0.01, 0.03],
            [0.0, 0.01, 0.03],
        ]
    )
    mesh = MeshInput(
        "/Inner",
        points,
        np.asarray([3, 3]),
        np.asarray([0, 1, 2, 3, 4, 5]),
        np.eye(4),
    )
    reference = preprocess_reference_mesh([mesh], ["/Inner"])
    oracle = ScalarFirstHitRaycaster(reference)
    distances, face_ids = oracle.query(
        np.zeros(3),
        np.asarray([[0.0, 0.0, 0.03], [0.05, 0.0, 0.03]]),
    )
    assert math.isclose(distances[0], 0.02, abs_tol=1.0e-12)
    assert face_ids.tolist() == [0, -1]
    assert math.isinf(distances[1])


def test_camera_facing_normal_gate_and_winding_correction():
    vertices = np.asarray([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]])
    common = dict(
        prim_path="/Face",
        vertices=vertices,
        face_vertex_counts=np.asarray([3]),
        face_vertex_indices=np.asarray([0, 1, 2]),
        world_transform=np.eye(4),
    )
    back = preprocess_reference_mesh([MeshInput(**common)], ["/Face"])
    front = preprocess_reference_mesh(
        [MeshInput(**common, orientation="leftHanded")], ["/Face"]
    )
    origin = np.asarray([0.0, 0.0, 0.0])
    target = np.asarray([[0.0, 0.0, 1.0]])
    assert triangle_normals(back)[0, 2] > 0.0
    assert camera_facing_first_hits(origin, target, np.asarray([0]), back).tolist() == [False]
    assert camera_facing_first_hits(origin, target, np.asarray([0]), front).tolist() == [True]
    assert camera_facing_first_hits(
        origin, target, np.asarray([0]), back, normal_sign=-1
    ).tolist() == [True]


def test_normal_sign_must_be_binary():
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mesh = MeshInput(
        "/Face", vertices, np.asarray([3]), np.asarray([0, 1, 2]), np.eye(4)
    )
    reference = preprocess_reference_mesh([mesh], ["/Face"])
    with pytest.raises(ValueError, match="normal_sign"):
        triangle_normals(reference, normal_sign=0)


def test_camera_facing_rejects_grazing_with_tolerance():
    mesh = MeshInput(
        "/Face",
        np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.asarray([3]),
        np.asarray([0, 1, 2]),
        np.eye(4),
    )
    reference = preprocess_reference_mesh([mesh], ["/Face"])
    accepted = camera_facing_first_hits(
        np.zeros(3), np.asarray([[0.0, 1.0, 0.0]]), np.asarray([0]), reference
    )
    assert accepted.tolist() == [False]
