import math

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.trajectory import (
    move_direction,
    quintic_progress,
    quintic_progress_rate,
    swing_angular_velocity,
    swing_axis,
)


def test_quintic_progress_has_exact_192_substep_swing_boundary():
    assert quintic_progress(0, 192) == pytest.approx(0.0)
    assert 0.0 < quintic_progress(1, 192) < 1.0
    assert quintic_progress(191, 192) < 1.0
    assert quintic_progress(192, 192) == pytest.approx(1.0)
    assert quintic_progress(239, 192) == pytest.approx(1.0)


def test_swing_axis_is_minimal_and_hits_fifteen_degree_target():
    start = np.asarray([0.0, 0.0, 1.0])
    target = np.asarray([math.sin(math.radians(15.0)), 0.0, math.cos(math.radians(15.0))])
    np.testing.assert_allclose(swing_axis(start, target, 0.0), start, atol=1.0e-12)
    np.testing.assert_allclose(swing_axis(start, target, 1.0), target, atol=1.0e-12)
    midpoint = swing_axis(start, target, 0.5)
    assert math.degrees(math.acos(float(start @ midpoint))) == pytest.approx(7.5, abs=1.0e-10)


def test_quintic_swing_rate_is_zero_at_boundaries_and_tangent_to_axis():
    start = np.asarray([0.0, 0.0, 1.0])
    target = np.asarray([math.sin(math.radians(15.0)), 0.0, math.cos(math.radians(15.0))])
    assert quintic_progress_rate(0, 192, 240) == pytest.approx(0.0)
    assert quintic_progress_rate(192, 192, 240) == pytest.approx(0.0)
    rate = quintic_progress_rate(96, 192, 240)
    omega = swing_angular_velocity(start, target, rate)
    assert np.linalg.norm(omega) > 0.0
    assert float(omega @ start) == pytest.approx(0.0, abs=1.0e-12)


def test_move_directions_are_opposite_and_tangent_to_normal_and_axis_projection():
    axis = np.asarray([0.8, 0.1, 0.3])
    normal = np.asarray([0.0, 0.0, 1.0])
    positive, degenerate = move_direction(axis, normal, positive=True)
    negative, negative_degenerate = move_direction(axis, normal, positive=False)
    tangent_axis = axis - float(axis @ normal) * normal
    assert not degenerate and not negative_degenerate
    np.testing.assert_allclose(negative, -positive)
    assert float(positive @ normal) == pytest.approx(0.0, abs=1.0e-12)
    assert float(positive @ tangent_axis) == pytest.approx(0.0, abs=1.0e-12)


def test_move_direction_reports_degenerate_for_axis_normal_to_surface():
    direction, degenerate = move_direction([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], positive=True)
    assert degenerate
    np.testing.assert_allclose(direction, np.zeros(3))
