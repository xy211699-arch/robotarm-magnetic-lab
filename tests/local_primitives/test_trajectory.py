import math

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    WORLD_UP, axis_at_tilt, cone_axis, directed_axis_from_quaternion_wxyz,
    quintic_progress, slerp_axis,
)


def test_quintic_profile_has_stationary_endpoints():
    assert quintic_progress(0.0, 2.0) == pytest.approx((0.0, 0.0))
    assert quintic_progress(2.0, 2.0) == pytest.approx((1.0, 0.0))
    assert quintic_progress(1.0, 2.0)[0] == pytest.approx(0.5)


def test_slerp_axis_is_unit_length_and_reaches_target():
    start = np.array([1.0, 0.0, 0.0])
    middle = slerp_axis(start, WORLD_UP, 1.0, 2.0)
    assert np.linalg.norm(middle.axis_world) == pytest.approx(1.0)
    assert middle.axis_world[0] == pytest.approx(math.sqrt(0.5))
    final = slerp_axis(start, WORLD_UP, 2.0, 2.0)
    np.testing.assert_allclose(final.axis_world, WORLD_UP, atol=1e-12)
    np.testing.assert_allclose(final.axis_dot_world, 0.0, atol=1e-12)


def test_cone_completes_positive_revolution_at_constant_tilt():
    first = cone_axis(math.radians(30.0), 0.25, 0.0, 8.0)
    half = cone_axis(math.radians(30.0), 0.25, 4.0, 8.0)
    final = cone_axis(math.radians(30.0), 0.25, 8.0, 8.0)
    assert math.acos(first.axis_world[2]) == pytest.approx(math.radians(30.0))
    assert half.phase_rad == pytest.approx(0.25 + math.pi)
    assert final.phase_rad == pytest.approx(0.25 + 2.0 * math.pi)
    np.testing.assert_allclose(first.axis_world, final.axis_world, atol=1e-12)


def test_directed_axis_uses_local_negative_z():
    np.testing.assert_allclose(directed_axis_from_quaternion_wxyz([1, 0, 0, 0]), [0, 0, -1])
    np.testing.assert_allclose(directed_axis_from_quaternion_wxyz([0, 1, 0, 0]), [0, 0, 1])
    expected = axis_at_tilt(math.radians(30), [1, 0])
    assert expected.tolist() == pytest.approx([0.5, 0.0, math.sqrt(3) / 2])
