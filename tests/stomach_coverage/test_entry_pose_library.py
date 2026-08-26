"""Pure contracts for TASK-009B entry pose-library geometry."""

from __future__ import annotations

import numpy as np

from robotarm_magnetic_lab.coverage.entry_pose_library import (
    MIN_UNORIENTED_AXIS_ANGLE_DEG,
    SPLIT_BASE_SEEDS,
    SPLIT_COUNTS,
    deterministic_candidate_seed,
    pose_fingerprint,
    sample_surface_pose,
    stable_record_is_valid,
    triangle_areas,
    unoriented_axis_angle_deg,
)
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh
from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw


def _horizontal_reference():
    source = MeshInput(
        "/World/Stomach",
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([3]),
        np.asarray([0, 1, 2]),
        np.eye(4),
    )
    return preprocess_reference_mesh([source], [source.prim_path])


def test_frozen_split_counts_and_independent_seed_namespaces():
    assert SPLIT_COUNTS == {"train": 1000, "validation": 100, "test": 100}
    assert len(set(SPLIT_BASE_SEEDS.values())) == 3
    seeds = {
        deterministic_candidate_seed(split, attempt)
        for split in SPLIT_COUNTS
        for attempt in range(20)
    }
    assert len(seeds) == 60


def test_surface_pose_is_area_sampled_tangent_and_deterministic():
    reference = _horizontal_reference()
    first = sample_surface_pose(reference, np.asarray([0]), 1234)
    second = sample_surface_pose(reference, np.asarray([0]), 1234)
    assert np.allclose(first.pose_world_xyzw, second.pose_world_xyzw)
    assert np.allclose(first.barycentric.sum(), 1.0)
    assert np.all(first.barycentric >= 0.0)
    rotation = rotation_matrix_from_xyzw(first.pose_world_xyzw[3:])
    assert abs(float(np.dot(rotation[:, 2], first.surface_normal_world))) < 1.0e-10
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
    assert np.linalg.det(rotation) > 0.0
    assert np.isclose(triangle_areas(reference, np.asarray([0]))[0], 1.0)


def test_unoriented_axis_angle_and_record_gate():
    assert unoriented_axis_angle_deg([1.0, 0.0, 0.0]) == 90.0
    assert unoriented_axis_angle_deg([0.0, 0.0, -1.0]) == 0.0
    record = {
        "pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "stable": True,
        "camera_inside_lumen": True,
        "unoriented_axis_angle_deg": MIN_UNORIENTED_AXIS_ANGLE_DEG,
    }
    assert stable_record_is_valid(record)
    record["camera_inside_lumen"] = False
    assert not stable_record_is_valid(record)


def test_pose_fingerprint_rejects_exact_duplicate_but_ignores_subnanometre_noise():
    pose = np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    assert pose_fingerprint(pose) == pose_fingerprint(pose.copy())
    assert pose_fingerprint(pose) == pose_fingerprint(pose + 1.0e-11)
    changed = pose.copy()
    changed[0] += 1.0e-6
    assert pose_fingerprint(pose) != pose_fingerprint(changed)
