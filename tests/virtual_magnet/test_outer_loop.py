from __future__ import annotations

import numpy as np

from virtual_magnet import ActionId, load_profile
from virtual_magnet.outer_loop import desired_hold_wrench, desired_move_wrench, desired_view_wrench


def test_hold_has_no_normal_force_or_long_axis_twist_objective():
    profile = load_profile()
    wrench = desired_hold_wrench(
        optical_axis=np.array([0.0, 0.1, 0.995]),
        target_optical_axis=np.array([0.0, 0.0, 1.0]),
        position=np.array([0.002, -0.001, 0.02]),
        tangent_anchor=np.array([0.0, 0.0, 0.0]),
        inward_normal=np.array([0.0, 0.0, 1.0]),
        linear_velocity=np.array([0.01, 0.0, 0.02]),
        angular_velocity=np.array([0.0, 0.0, 1.0]),
        profile=profile,
    )
    assert wrench.shape == (6,)
    assert abs(wrench[2]) < 1.0e-12
    optical = np.array([0.0, 0.1, 0.995])
    optical /= np.linalg.norm(optical)
    assert abs(np.dot(wrench[3:], optical)) < 1.0e-12


def test_view_prioritizes_minimal_swing_and_tangent_anchor():
    profile = load_profile()
    wrench = desired_view_wrench(
        optical_axis=np.array([0.0, 0.0, 1.0]),
        target_optical_axis=np.array([0.0, 0.258819, 0.965926]),
        position=np.array([0.001, 0.002, 0.02]),
        tangent_anchor=np.zeros(3),
        inward_normal=np.array([0.0, 0.0, 1.0]),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        profile=profile,
    )
    assert wrench[3] < 0.0
    assert abs(wrench[5]) < 1.0e-12
    assert abs(wrench[2]) < 1.0e-12


def test_move_outputs_tangent_force_and_no_active_tilt_or_roll_torque():
    profile = load_profile()
    direction = np.array([0.0, 1.0, 0.0])
    wrench = desired_move_wrench(
        position=np.zeros(3),
        target_position=np.array([0.0, 0.005, 0.0]),
        start_position=np.zeros(3),
        move_direction=direction,
        inward_normal=np.array([0.0, 0.0, 1.0]),
        linear_velocity=np.zeros(3),
        profile=profile,
    )
    assert wrench[1] > 0.0
    assert abs(wrench[2]) < 1.0e-12
    np.testing.assert_allclose(wrench[3:], 0.0)
    assert np.linalg.norm(wrench[:3]) <= profile.max_desired_force_n + 1.0e-12
