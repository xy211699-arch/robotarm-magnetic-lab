from __future__ import annotations

from dataclasses import fields
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from virtual_magnet import (
    ActionId,
    ActionResult,
    ClosedLoopProfile,
    camera_image_axes_from_ros_rotation,
    load_profile,
    move_direction,
    profile_sha256,
    quintic_progress,
    unsigned_axis_tilt,
    view_target_axis,
)


def test_public_action_contract_is_exact():
    assert [(a.value, a.name) for a in ActionId] == [
        (0, "HOLD_VIEW"),
        (1, "VIEW_UP"),
        (2, "VIEW_UP_RIGHT"),
        (3, "VIEW_RIGHT"),
        (4, "VIEW_DOWN_RIGHT"),
        (5, "VIEW_DOWN"),
        (6, "VIEW_DOWN_LEFT"),
        (7, "VIEW_LEFT"),
        (8, "VIEW_UP_LEFT"),
        (9, "MOVE_SIDE_POS"),
        (10, "MOVE_SIDE_NEG"),
    ]
    assert {result.value for result in ActionResult} == {"completed", "rejected", "fault"}


def test_profile_freezes_all_timing_and_acceptance_contracts():
    profile = load_profile()
    assert profile.schema_version == "task007_virtual_magnet_closed_loop_v1"
    assert profile.physics_hz == 240
    assert profile.feedback_hz == 60
    assert profile.action_hz == 1
    assert profile.action_substeps == 240
    assert profile.feedback_stride == 4
    assert profile.motion_substeps == 192
    assert profile.stabilization_substeps == 48
    assert profile.stability_window_substeps == 24
    assert profile.contact_window_substeps == 12
    assert profile.view_cone_deg == pytest.approx(15.0)
    assert profile.move_target_m == pytest.approx(0.005)
    assert profile.move_acceptance_min_m == pytest.approx(0.004)
    assert profile.move_acceptance_max_m == pytest.approx(0.006)
    assert profile.move_tilt_min_deg == pytest.approx(45.0)
    assert profile.boundary_linear_speed_m_s == pytest.approx(0.002)
    assert profile.boundary_angular_speed_rad_s == pytest.approx(0.1)
    assert len(profile_sha256()) == 64
    assert not any("stomach" in item.name or "flat" in item.name for item in fields(ClosedLoopProfile))


def test_view_geometry_uses_frozen_camera_image_frame():
    optical = np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 1.0, 0.0])
    right = np.array([1.0, 0.0, 0.0])
    up_target = view_target_axis(optical, up, right, ActionId.VIEW_UP, 15.0)
    right_target = view_target_axis(optical, up, right, ActionId.VIEW_RIGHT, 15.0)
    diagonal = view_target_axis(optical, up, right, ActionId.VIEW_UP_RIGHT, 15.0)
    assert math.degrees(math.acos(np.dot(optical, up_target))) == pytest.approx(15.0)
    assert up_target[1] > 0.0 and abs(up_target[0]) < 1.0e-12
    assert right_target[0] > 0.0 and abs(right_target[1]) < 1.0e-12
    assert diagonal[0] == pytest.approx(diagonal[1])


def test_real_camera_mount_maps_ros_image_axes_without_sign_reversal():
    # Both quaternions are xyzw. The camera is mounted at the capsule -Z end
    # with a 180-degree rotation about capsule Y.
    capsule_rotation = Rotation.from_quat(
        [-0.2565394892869615, 0.6589290481048662, 0.2565394892869615, 0.6589290481048662]
    ).as_matrix()
    camera_mount_rotation = Rotation.from_quat([0.0, 1.0, 0.0, 0.0]).as_matrix()
    camera_rotation = capsule_rotation @ camera_mount_rotation
    optical, up, right = camera_image_axes_from_ros_rotation(camera_rotation)

    np.testing.assert_allclose(optical, -capsule_rotation[:, 2], atol=1.0e-12)
    np.testing.assert_allclose(up, -capsule_rotation[:, 1], atol=1.0e-12)
    np.testing.assert_allclose(right, -capsule_rotation[:, 0], atol=1.0e-12)

    up_target = view_target_axis(optical, up, right, ActionId.VIEW_UP, 15.0)
    right_target = view_target_axis(optical, up, right, ActionId.VIEW_RIGHT, 15.0)
    assert np.dot(up_target - optical, up) > 0.0
    assert abs(float(np.dot(up_target - optical, right))) < 1.0e-12
    assert np.dot(right_target - optical, right) > 0.0
    assert abs(float(np.dot(right_target - optical, up))) < 1.0e-12


def test_move_uses_unsigned_tilt_and_frozen_opposite_tangents():
    assert unsigned_axis_tilt((0, 0, -1), (0, 0, 1)) == pytest.approx(0.0)
    assert math.degrees(unsigned_axis_tilt((1, 0, 0), (0, 0, 1))) == pytest.approx(90.0)
    pos = move_direction(axis=(1, 0, 0), normal=(0, 0, 1), sign=1)
    neg = move_direction(axis=(1, 0, 0), normal=(0, 0, 1), sign=-1)
    np.testing.assert_allclose(pos, -neg)
    np.testing.assert_allclose(pos, [0.0, 1.0, 0.0])


def test_quintic_progress_has_zero_end_velocity_and_clamps():
    assert quintic_progress(-1.0) == pytest.approx(0.0)
    assert quintic_progress(0.0) == pytest.approx(0.0)
    assert quintic_progress(0.5) == pytest.approx(0.5)
    assert quintic_progress(1.0) == pytest.approx(1.0)
    assert quintic_progress(2.0) == pytest.approx(1.0)
    eps = 1.0e-5
    assert quintic_progress(eps) / eps < 1.0e-3
    assert (1.0 - quintic_progress(1.0 - eps)) / eps < 1.0e-3


def test_profile_rejects_unknown_or_nonfinite_values(tmp_path):
    profile_path = tmp_path / "bad.json"
    profile_path.write_text('{"schema_version":"bad","unknown":1}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(profile_path)
