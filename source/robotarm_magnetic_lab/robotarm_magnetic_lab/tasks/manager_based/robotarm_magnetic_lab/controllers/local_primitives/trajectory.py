"""Smooth directed-axis trajectories for the four primitive motions."""

from __future__ import annotations

import math

import numpy as np

from .types import AxisTarget


WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def quintic_progress(elapsed_s: float, duration_s: float) -> tuple[float, float]:
    """Return minimum-jerk progress and its derivative."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    ratio = float(np.clip(elapsed_s / duration_s, 0.0, 1.0))
    progress = 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5
    derivative = (30.0 * ratio**2 - 60.0 * ratio**3 + 30.0 * ratio**4) / duration_s
    if ratio >= 1.0:
        derivative = 0.0
    return progress, derivative


def axis_at_tilt(tilt_rad: float, direction_xy: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction_xy, dtype=np.float64).reshape(2)
    direction /= np.linalg.norm(direction)
    return np.array(
        [math.sin(tilt_rad) * direction[0], math.sin(tilt_rad) * direction[1], math.cos(tilt_rad)],
        dtype=np.float64,
    )


def slerp_axis(start: np.ndarray, end: np.ndarray, elapsed_s: float, duration_s: float) -> AxisTarget:
    """Interpolate two directed unit axes along their shortest great-circle arc."""

    start_axis = _unit(start)
    end_axis = _unit(end)
    progress, progress_dot = quintic_progress(elapsed_s, duration_s)
    dot = float(np.clip(np.dot(start_axis, end_axis), -1.0, 1.0))
    angle = math.acos(dot)
    if angle < 1.0e-9:
        return AxisTarget(end_axis, np.zeros(3, dtype=np.float64))
    if math.pi - angle < 1.0e-7:
        raise ValueError("antipodal directed-axis interpolation is undefined")
    sin_angle = math.sin(angle)
    a = math.sin((1.0 - progress) * angle) / sin_angle
    b = math.sin(progress * angle) / sin_angle
    axis = a * start_axis + b * end_axis
    derivative_progress = (
        -angle * math.cos((1.0 - progress) * angle) * start_axis
        + angle * math.cos(progress * angle) * end_axis
    ) / sin_angle
    return AxisTarget(axis, derivative_progress * progress_dot)


def cone_axis(tilt_rad: float, initial_phase_rad: float, elapsed_s: float, duration_s: float) -> AxisTarget:
    """Trace one positive world-Z cone revolution with a minimum-jerk phase."""

    progress, progress_dot = quintic_progress(elapsed_s, duration_s)
    phase = initial_phase_rad + 2.0 * math.pi * progress
    phase_dot = 2.0 * math.pi * progress_dot
    sin_tilt = math.sin(tilt_rad)
    axis = np.array(
        [sin_tilt * math.cos(phase), sin_tilt * math.sin(phase), math.cos(tilt_rad)],
        dtype=np.float64,
    )
    axis_dot = np.array(
        [-sin_tilt * math.sin(phase) * phase_dot, sin_tilt * math.cos(phase) * phase_dot, 0.0],
        dtype=np.float64,
    )
    return AxisTarget(axis, axis_dot, phase)


def directed_axis_from_quaternion_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate capsule local ``-Z`` into world coordinates."""

    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion must be finite and nonzero")
    w, x, y, z = q / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float64)


def tilt_from_axis(axis_world: np.ndarray) -> float:
    return math.acos(float(np.clip(_unit(axis_world)[2], -1.0, 1.0)))


def azimuth_from_axis(axis_world: np.ndarray, fallback: float = 0.0) -> float:
    axis = _unit(axis_world)
    if float(np.linalg.norm(axis[:2])) < 1.0e-9:
        return float(fallback)
    return math.atan2(float(axis[1]), float(axis[0]))


def wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("axis must be finite and nonzero")
    return vector / norm
