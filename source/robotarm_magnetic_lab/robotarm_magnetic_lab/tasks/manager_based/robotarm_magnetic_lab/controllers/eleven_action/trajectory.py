"""Integer-substep trajectories for the TASK-005 action controller."""

from __future__ import annotations

import math

import numpy as np

from .geometry import normalized, tangent_projection


def quintic_progress(substep_index: int, motion_substeps: int) -> float:
    if substep_index < 0 or motion_substeps < 1:
        raise ValueError("trajectory substeps must be valid")
    tau = float(np.clip(float(substep_index) / float(motion_substeps), 0.0, 1.0))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def quintic_progress_rate(substep_index: int, motion_substeps: int, physics_hz: int) -> float:
    """Return time derivative of quintic progress in inverse seconds."""
    if substep_index < 0 or motion_substeps < 1 or physics_hz < 1:
        raise ValueError("trajectory rate arguments must be valid")
    tau = float(np.clip(float(substep_index) / float(motion_substeps), 0.0, 1.0))
    derivative_tau = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    return derivative_tau * float(physics_hz) / float(motion_substeps)


def swing_angular_velocity(start_axis_world, target_axis_world, progress_rate: float) -> np.ndarray:
    """Minimal-swing desired angular velocity for a progress derivative."""
    start = normalized(start_axis_world, name="swing velocity start")
    target = normalized(target_axis_world, name="swing velocity target")
    cross = np.cross(start, target)
    cross_norm = float(np.linalg.norm(cross))
    angle = math.atan2(cross_norm, float(np.clip(start @ target, -1.0, 1.0)))
    if cross_norm <= 1.0e-12 or angle <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    return (cross / cross_norm) * angle * float(progress_rate)


def swing_axis(start_axis_world, target_axis_world, progress: float) -> np.ndarray:
    start = normalized(start_axis_world, name="swing start")
    target = normalized(target_axis_world, name="swing target")
    amount = float(np.clip(progress, 0.0, 1.0))
    cross = np.cross(start, target)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(start @ target, -1.0, 1.0))
    angle = math.atan2(cross_norm, dot)
    if cross_norm <= 1.0e-12 or angle <= 1.0e-12:
        return start.copy() if amount < 1.0 else target.copy()
    rotation_axis = cross / cross_norm
    theta = amount * angle
    # Rodrigues rotation on the directed optical axis.
    value = (
        math.cos(theta) * start
        + math.sin(theta) * np.cross(rotation_axis, start)
        + (1.0 - math.cos(theta)) * float(rotation_axis @ start) * rotation_axis
    )
    return normalized(value, name="swing interpolation")


def move_direction(axis_world, normal_world, *, positive: bool) -> tuple[np.ndarray, bool]:
    normal = normalized(normal_world, name="move surface normal")
    tangent_axis = tangent_projection(axis_world, normal)
    length = float(np.linalg.norm(tangent_axis))
    if not np.isfinite(length) or length <= 1.0e-10:
        return np.zeros(3, dtype=np.float64), True
    heading = tangent_axis / length
    lateral = normalized(np.cross(normal, heading), name="lateral move direction")
    return (lateral if positive else -lateral), False
