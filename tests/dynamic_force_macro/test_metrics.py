import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force_macro import (
    move_projected_displacement_m,
    up_elevation_and_crossing,
    view_signed_angle_deg,
)


def test_move_metric_is_frozen_projection():
    assert move_projected_displacement_m([0, 0, 0], [0, 0.006, 0.002], [0, 1, 0]) == pytest.approx(0.006)


def test_view_metric_signed_in_requested_plane():
    angle = np.deg2rad(20.0)
    assert view_signed_angle_deg([1, 0, 0], [np.cos(angle), np.sin(angle), 0], [0, 1, 0]) == pytest.approx(20.0)


def test_up_metric_detects_elevation_and_crossing():
    angle = np.deg2rad(50.0)
    end = [np.cos(angle), 0, np.sin(angle)]
    elevation, crossed = up_elevation_and_crossing(end, [1, 0, 0], ([1, 0, 0], end))
    assert elevation == pytest.approx(50.0)
    assert not crossed
    _, crossed = up_elevation_and_crossing(end, [1, 0, 0], ([1, 0, 0], [-0.1, 0, 0.995]))
    assert crossed
