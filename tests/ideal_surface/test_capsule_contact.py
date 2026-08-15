from __future__ import annotations

import math

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface import (
    CapsulePose,
    ContactClassifier,
    IdealSurfaceConfig,
    Spherocylinder,
    SurfaceNavigationMesh,
    assess_pose,
    assess_swept_target,
    select_active_anchor,
)


class _Reference:
    vertices_world = np.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]])


def plane_mesh():
    return SurfaceNavigationMesh.from_reference(_Reference(), inward_sign=1)


def capsule():
    return Spherocylinder(radius_m=0.005, cylinder_half_length_m=0.0075)


def pose(axis, center_z):
    return CapsulePose(
        center_world=np.asarray([0.0, 0.0, center_z]),
        axis_world=np.asarray(axis, dtype=float),
        image_up_world=np.asarray([0.0, 1.0, 0.0]),
    )


def test_any_contact_is_not_side_contact():
    assessment = assess_pose(
        plane_mesh(), capsule(), pose([0, 0, 1], 0.0125), active_triangle=0, cfg=IdealSurfaceConfig()
    )
    assert assessment.support_valid
    assert not assessment.side_contact
    assert not assessment.contact_limited


def test_two_separated_barrel_samples_create_stable_side_contact():
    detector = ContactClassifier(IdealSurfaceConfig(), capsule())
    result = None
    for _ in range(math.ceil(0.1 / (1 / 240))):
        result = detector.observe(
            pose([1, 0, 0], 0.005),
            barrel_clearances=np.array([0.0, 0.0]),
            barrel_axial_parameters=np.array([-0.5, 0.5]),
            dt=1 / 240,
        )
    assert result is not None and result.side_contact


def test_side_contact_stability_resets_when_separation_or_clearance_is_invalid():
    detector = ContactClassifier(IdealSurfaceConfig(), capsule())
    for _ in range(23):
        detector.observe(
            pose([1, 0, 0], 0.005), np.array([0.0, 0.0]), np.array([-0.5, 0.5]), 1 / 240
        )
    result = detector.observe(
        pose([1, 0, 0], 0.005), np.array([0.0, 0.001]), np.array([-0.05, 0.05]), 1 / 240
    )
    assert not result.side_contact


def test_penetrating_next_pose_clips_without_hard_failure():
    cfg = IdealSurfaceConfig()
    current = pose([1, 0, 0], 0.005)
    proposed = pose([1, 0, 0], 0.005 - 0.02 * capsule().radius_m)
    result = assess_swept_target(
        mesh=plane_mesh(), capsule=capsule(), current=current, proposed=proposed,
        active_triangle=0, cfg=cfg
    )
    assert result.assessment.contact_limited
    assert not result.assessment.hard_failure
    assert result.pose.center_world[2] == pytest.approx(
        capsule().radius_m - cfg.planned_penetration_radius_fraction * capsule().radius_m
    )


def test_penetration_over_hard_threshold_is_terminal():
    cfg = IdealSurfaceConfig()
    result = assess_pose(
        plane_mesh(), capsule(), pose([1, 0, 0], 0.004), active_triangle=0, cfg=cfg
    )
    assert result.hard_failure
    assert result.maximum_penetration_m > cfg.hard_penetration_radius_fraction * capsule().radius_m


def test_side_contact_anchor_is_extreme_opposite_tilt_ray():
    contacts = np.array([[-0.0075, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0075, 0.0, 0.0]])
    anchor = select_active_anchor(
        contacts_world=contacts,
        center_world=np.zeros(3),
        tilt_direction_world=np.array([1.0, 0.0, 0.0]),
        triangle_ids=np.array([3, 2, 1]),
    )
    np.testing.assert_allclose(anchor.point_world, [-0.0075, 0.0, 0.0])
    assert anchor.triangle_id == 3


def test_anchor_tie_breaks_by_triangle_then_input_index():
    anchor = select_active_anchor(
        contacts_world=np.asarray([[0, 1, 0], [0, -1, 0], [0, 1, 0]], dtype=float),
        center_world=np.zeros(3),
        tilt_direction_world=np.array([1.0, 0.0, 0.0]),
        triangle_ids=np.array([5, 2, 2]),
    )
    assert anchor.triangle_id == 2
    assert anchor.contact_index == 1
