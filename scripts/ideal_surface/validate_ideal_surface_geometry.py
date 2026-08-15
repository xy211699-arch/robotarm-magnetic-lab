#!/usr/bin/env python3
"""Deterministically validate the pure ideal-surface controller on a plane."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface import (  # noqa: E402
    ContactAssessment,
    ControllerSnapshot,
    IdealActionStatus,
    IdealSurfaceAction,
    IdealSurfaceConfig,
    IdealSurfaceController,
    SurfaceFlags,
    SurfaceNavigationMesh,
    orientation_from_axis_and_image_up,
    quaternion_wxyz_to_matrix,
)


class Plane:
    vertices_world = np.asarray([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], float)
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], int)


def snapshot(theta_deg=0.0, phi_deg=0.0, side=False, surface=None):
    cfg = IdealSurfaceConfig()
    theta, phi = math.radians(theta_deg), math.radians(phi_deg)
    direction = np.asarray([-math.sin(phi), math.cos(phi), 0.0])
    normal = np.asarray([0.0, 0.0, 1.0])
    axis = math.sin(theta) * direction + math.cos(theta) * normal
    image_up = math.cos(theta) * direction - math.sin(theta) * normal
    quaternion = orientation_from_axis_and_image_up(axis, image_up)
    image_up = quaternion_wxyz_to_matrix(quaternion)[:, 1]
    surface = np.zeros(3) if surface is None else np.asarray(surface, dtype=float)
    height = cfg.capsule_radius_m + cfg.capsule_cylinder_half_length_m * abs(axis @ normal)
    return ControllerSnapshot(
        0.0,
        surface + height * normal,
        quaternion,
        axis,
        image_up,
        surface,
        normal,
        0,
        theta,
        phi,
        SurfaceFlags(theta_deg <= 5.0, side),
    )


def make_controller(initial, assessor=None):
    value = IdealSurfaceController(
        SurfaceNavigationMesh.from_reference(Plane(), inward_sign=1),
        pose_assessor=assessor,
    )
    value.reset(initial)
    return value


def execute(value, action, request=1):
    assert value.submit(action, value.snapshot, request)
    output = None
    for _ in range(240):
        output = value.step(1 / 240)
    assert output is not None and output.result is not None
    return output.result


def validate_plane_controller() -> dict[str, bool]:
    axes = []
    direction_errors = []
    for action_id in range(1, 9):
        result = execute(make_controller(snapshot()), action_id, action_id)
        axes.append(tuple(np.round(result.final_axis_world, 8)))
        direction_errors.append(abs(math.degrees(result.final_tilt_rad) - 15.0))

    tilt = execute(
        make_controller(snapshot(30.0)), IdealSurfaceAction.TILT_MORE, 20
    )
    rise = execute(make_controller(snapshot(30.0)), IdealSurfaceAction.RISE, 21)
    precess = execute(
        make_controller(snapshot(45.0, 10.0)), IdealSurfaceAction.PRECESS_POS, 22
    )
    side = snapshot(90.0, 0.0, side=True)
    roll = execute(make_controller(side), IdealSurfaceAction.ROLL_POS, 23)
    roll_distance = float(np.linalg.norm(roll.final_position_world - side.position_world))

    residual_ok = True
    for residual in np.linspace(0.0, 5.0, 6):
        result = execute(make_controller(snapshot(float(residual))), 1, 100 + int(residual))
        residual_ok &= abs(math.degrees(result.final_tilt_rad) - 15.0) <= 0.2

    calls = 0

    def limiter(pose, triangle, cfg):
        nonlocal calls
        calls += 1
        limited = calls >= 145
        return ContactAssessment(
            True, False, limited, False, False, 0.0,
            np.zeros(3), np.asarray([0, 0, 1]), triangle,
            np.ones(2), np.asarray([-1.0, 1.0]),
        )

    saturated = execute(
        make_controller(snapshot(15.0), limiter), IdealSurfaceAction.TILT_MORE, 30
    )
    boundary_start = snapshot(90.0, 0.0, side=True, surface=[1.999, 0, 0])
    boundary = execute(
        make_controller(boundary_start), IdealSurfaceAction.ROLL_POS, 31
    )
    checks = {
        "eight_unique_start_tilts": len(set(axes)) == 8,
        "start_tilt_absolute_15_deg": max(direction_errors) <= 0.2,
        "tilt_increment_deg": abs(math.degrees(tilt.final_tilt_rad) - 45.0) <= 0.2,
        "rise_increment_deg": abs(math.degrees(rise.final_tilt_rad) - 45.0) <= 0.2,
        "precession_increment_deg": (
            abs(math.degrees(precess.final_tilt_rad) - 45.0) <= 0.2
            and abs(math.degrees(precess.final_azimuth_rad) - 25.0) <= 0.2
        ),
        "roll_arc_length_m": abs(roll_distance - 0.010) <= 0.0001,
        "upright_residual_range_deg": bool(residual_ok),
        "contact_limited_done": (
            saturated.status is IdealActionStatus.DONE and saturated.contact_limited
        ),
        "boundary_limited_done": (
            boundary.status is IdealActionStatus.DONE and boundary.boundary_limited
        ),
    }
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks, sort_keys=True))
    return checks


if __name__ == "__main__":
    result = validate_plane_controller()
    print("IDEAL_SURFACE_GEOMETRY_PASS " + json.dumps(result, sort_keys=True), flush=True)
