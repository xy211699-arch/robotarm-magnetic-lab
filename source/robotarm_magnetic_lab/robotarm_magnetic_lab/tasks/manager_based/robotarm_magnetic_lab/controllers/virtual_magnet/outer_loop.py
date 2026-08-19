"""Outer-loop desired magnetic-wrench laws for HOLD, VIEW, and MOVE."""

from __future__ import annotations

import numpy as np

from .config import ClosedLoopProfile
from .geometry import normalize


def _clip(vector: np.ndarray, limit: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > limit and norm > 1.0e-12:
        value = value * (limit / norm)
    return value


def _tangent(vector, normal) -> np.ndarray:
    normal_value = normalize(normal)
    value = np.asarray(vector, dtype=np.float64)
    return value - normal_value * np.dot(value, normal_value)


def _minimal_swing_torque(optical_axis, target_optical_axis, angular_velocity, kp, kd) -> np.ndarray:
    optical = normalize(optical_axis)
    target = normalize(target_optical_axis)
    swing_error = np.cross(optical, target)
    angular = np.asarray(angular_velocity, dtype=np.float64)
    angular_without_twist = angular - optical * np.dot(angular, optical)
    torque = float(kp) * swing_error - float(kd) * angular_without_twist
    return torque - optical * np.dot(torque, optical)


def desired_hold_wrench(
    *,
    optical_axis,
    target_optical_axis,
    position,
    tangent_anchor,
    inward_normal,
    linear_velocity,
    angular_velocity,
    profile: ClosedLoopProfile,
) -> np.ndarray:
    position_error = _tangent(np.asarray(tangent_anchor) - np.asarray(position), inward_normal)
    tangent_velocity = _tangent(linear_velocity, inward_normal)
    force = profile.hold_anchor_kp_n_m * position_error - profile.hold_anchor_kd_n_s_m * tangent_velocity
    torque = _minimal_swing_torque(
        optical_axis,
        target_optical_axis,
        angular_velocity,
        profile.hold_axis_kp_nm,
        profile.hold_axis_kd_nm_s,
    )
    return np.concatenate(
        (_clip(force, profile.max_desired_force_n), _clip(torque, profile.max_desired_torque_nm))
    )


def desired_view_wrench(
    *,
    optical_axis,
    target_optical_axis,
    position,
    tangent_anchor,
    inward_normal,
    linear_velocity,
    angular_velocity,
    profile: ClosedLoopProfile,
) -> np.ndarray:
    position_error = _tangent(np.asarray(tangent_anchor) - np.asarray(position), inward_normal)
    tangent_velocity = _tangent(linear_velocity, inward_normal)
    force = profile.view_anchor_kp_n_m * position_error - profile.view_anchor_kd_n_s_m * tangent_velocity
    torque = _minimal_swing_torque(
        optical_axis,
        target_optical_axis,
        angular_velocity,
        profile.view_axis_kp_nm,
        profile.view_axis_kd_nm_s,
    )
    return np.concatenate(
        (_clip(force, profile.max_desired_force_n), _clip(torque, profile.max_desired_torque_nm))
    )


def desired_move_wrench(
    *,
    position,
    target_position,
    start_position,
    move_direction,
    inward_normal,
    linear_velocity,
    profile: ClosedLoopProfile,
) -> np.ndarray:
    direction = normalize(move_direction)
    normal = normalize(inward_normal)
    position_error = _tangent(np.asarray(target_position) - np.asarray(position), normal)
    velocity = _tangent(linear_velocity, normal)
    signed_error = float(np.dot(position_error, direction))
    signed_velocity = float(np.dot(velocity, direction))
    along_force = (
        profile.move_tangent_kp_n_m * signed_error
        - profile.move_tangent_kd_n_s_m * signed_velocity
    ) * direction
    cross_error = position_error - signed_error * direction
    cross_velocity = velocity - signed_velocity * direction
    cross_force = profile.move_cross_kp_n_m * cross_error - profile.move_cross_kd_n_s_m * cross_velocity
    force = _clip(_tangent(along_force + cross_force, normal), profile.max_desired_force_n)
    return np.concatenate((force, np.zeros(3)))
