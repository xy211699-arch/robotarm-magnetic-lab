"""Bounded numerical inverse from desired finite-magnet wrench to magnet pose."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation


WrenchModel = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PoseInverseState:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    capsule_magnet_position: np.ndarray
    capsule_magnet_rotation: np.ndarray
    nominal_position: np.ndarray
    nominal_quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class PoseInverseResult:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    pose_increment: np.ndarray
    current_wrench: np.ndarray
    residual: np.ndarray
    condition_number: float
    solver_saturated: bool


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(size)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} is non-finite")
    return array


def _unit_quaternion_xyzw(value) -> np.ndarray:
    quaternion = _finite_vector(value, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    return quaternion / norm


def numerical_pose_jacobian(
    model: WrenchModel,
    position,
    rotation,
    translation_step_m: float,
    rotation_step_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference a local-axis SE(3) pose-to-wrench Jacobian."""
    position_value = _finite_vector(position, 3, "position")
    rotation_value = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(rotation_value).all():
        raise ValueError("rotation is non-finite")
    if translation_step_m <= 0.0 or rotation_step_rad <= 0.0:
        raise ValueError("finite-difference steps must be positive")
    current = _finite_vector(model(position_value, rotation_value), 6, "model wrench")
    jacobian = np.empty((6, 6), dtype=np.float64)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = translation_step_m
        plus = _finite_vector(model(position_value + delta, rotation_value), 6, "translation plus wrench")
        minus = _finite_vector(model(position_value - delta, rotation_value), 6, "translation minus wrench")
        jacobian[:, axis] = (plus - minus) / (2.0 * translation_step_m)
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = rotation_step_rad
        plus_rotation = rotation_value @ Rotation.from_rotvec(delta).as_matrix()
        minus_rotation = rotation_value @ Rotation.from_rotvec(-delta).as_matrix()
        plus = _finite_vector(model(position_value, plus_rotation), 6, "rotation plus wrench")
        minus = _finite_vector(model(position_value, minus_rotation), 6, "rotation minus wrench")
        jacobian[:, 3 + axis] = (plus - minus) / (2.0 * rotation_step_rad)
    if not np.isfinite(jacobian).all():
        raise ValueError("pose Jacobian is non-finite")
    return current, jacobian


def integrate_pose_increment(
    position,
    quaternion_xyzw,
    pose_increment,
    *,
    translation_trust_m: float,
    rotation_trust_rad: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply a clipped local-axis SE(3) increment with quaternion continuity."""
    position_value = _finite_vector(position, 3, "position")
    quaternion = _unit_quaternion_xyzw(quaternion_xyzw)
    increment = _finite_vector(pose_increment, 6, "pose increment").copy()
    clipped = False
    translation_norm = float(np.linalg.norm(increment[:3]))
    if translation_norm > translation_trust_m:
        increment[:3] *= translation_trust_m / translation_norm
        clipped = True
    rotation_norm = float(np.linalg.norm(increment[3:]))
    if rotation_norm > rotation_trust_rad:
        increment[3:] *= rotation_trust_rad / rotation_norm
        clipped = True
    current_rotation = Rotation.from_quat(quaternion)
    next_rotation = current_rotation * Rotation.from_rotvec(increment[3:])
    next_quaternion = _unit_quaternion_xyzw(next_rotation.as_quat())
    if float(np.dot(next_quaternion, quaternion)) < 0.0:
        next_quaternion = -next_quaternion
    return position_value + increment[:3], next_quaternion, clipped


def _hold_result(state: PoseInverseState, current, residual, condition) -> PoseInverseResult:
    return PoseInverseResult(
        position=_finite_vector(state.position, 3, "position").copy(),
        quaternion_xyzw=_unit_quaternion_xyzw(state.quaternion_xyzw).copy(),
        pose_increment=np.zeros(6),
        current_wrench=np.asarray(current, dtype=np.float64).reshape(6),
        residual=np.asarray(residual, dtype=np.float64).reshape(6),
        condition_number=float(condition),
        solver_saturated=True,
    )


def solve_pose_increment(
    model: WrenchModel,
    state: PoseInverseState,
    desired_wrench,
    *,
    weights,
    translation_step_m: float,
    rotation_step_rad: float,
    damping: float,
    relative_regularization: float,
    translation_trust_m: float,
    rotation_trust_rad: float,
    minimum_separation_m: float,
    maximum_separation_m: float,
    maximum_relative_angle_rad: float,
    condition_limit: float,
) -> PoseInverseResult:
    """Solve one weighted regularized update and enforce finite pose limits."""
    rotation = Rotation.from_quat(_unit_quaternion_xyzw(state.quaternion_xyzw)).as_matrix()
    current, jacobian = numerical_pose_jacobian(
        model,
        state.position,
        rotation,
        translation_step_m,
        rotation_step_rad,
    )
    desired = _finite_vector(desired_wrench, 6, "desired wrench")
    residual = desired - current
    weight_values = _finite_vector(weights, 6, "wrench weights")
    weighted_jacobian = weight_values[:, None] * jacobian
    weighted_residual = weight_values * residual
    singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
    condition = math.inf if singular_values[-1] <= 1.0e-18 else float(singular_values[0] / singular_values[-1])
    if not math.isfinite(condition) or condition > condition_limit:
        return _hold_result(state, current, residual, condition)

    nominal_rotation = Rotation.from_quat(_unit_quaternion_xyzw(state.nominal_quaternion_xyzw)).as_matrix()
    relative_target = np.concatenate(
        (
            _finite_vector(state.nominal_position, 3, "nominal position") - _finite_vector(state.position, 3, "position"),
            Rotation.from_matrix(rotation.T @ nominal_rotation).as_rotvec(),
        )
    )
    identity = np.eye(6)
    hessian = weighted_jacobian.T @ weighted_jacobian
    hessian += (float(damping) ** 2 + float(relative_regularization) ** 2) * identity
    right_hand = weighted_jacobian.T @ weighted_residual
    right_hand += float(relative_regularization) ** 2 * relative_target
    try:
        raw_increment = np.linalg.solve(hessian, right_hand)
    except np.linalg.LinAlgError:
        return _hold_result(state, current, residual, condition)
    if not np.isfinite(raw_increment).all():
        return _hold_result(state, current, residual, condition)

    next_position, next_quaternion, clipped = integrate_pose_increment(
        state.position,
        state.quaternion_xyzw,
        raw_increment,
        translation_trust_m=translation_trust_m,
        rotation_trust_rad=rotation_trust_rad,
    )
    separation = float(np.linalg.norm(next_position - _finite_vector(state.capsule_magnet_position, 3, "capsule magnet position")))
    capsule_rotation = np.asarray(state.capsule_magnet_rotation, dtype=np.float64).reshape(3, 3)
    relative_angle = Rotation.from_matrix(
        capsule_rotation.T @ Rotation.from_quat(next_quaternion).as_matrix()
    ).magnitude()
    if (
        not minimum_separation_m <= separation <= maximum_separation_m
        or relative_angle > maximum_relative_angle_rad
        or not np.isfinite(next_position).all()
        or not np.isfinite(next_quaternion).all()
    ):
        return _hold_result(state, current, residual, condition)
    applied_increment = np.concatenate(
        (
            next_position - np.asarray(state.position, dtype=np.float64),
            Rotation.from_matrix(rotation.T @ Rotation.from_quat(next_quaternion).as_matrix()).as_rotvec(),
        )
    )
    return PoseInverseResult(
        position=next_position,
        quaternion_xyzw=next_quaternion,
        pose_increment=applied_increment,
        current_wrench=current,
        residual=residual,
        condition_number=condition,
        solver_saturated=clipped,
    )
