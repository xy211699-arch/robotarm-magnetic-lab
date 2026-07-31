# Copyright (c) 2026, robotarm magnetic simulation contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop magnetic trajectories for the flat-table benchmark.

The capsule is never actuated by this module.  It produces only robot-arm and
ball-joint references from nominal magnetic-field trajectories.  Measured
capsule state is intentionally excluded from every control calculation; test
scripts may still record it for offline acceptance checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


CAPSULE_RADIUS_M = 0.0065
CAPSULE_LENGTH_M = 0.025


class MotionMode(str, Enum):
    """Supported open-loop magnetic motion modes."""

    TILT_AZIMUTH = "tilt_azimuth"
    UPRIGHT_TO_SIDE = "upright_to_side"
    LONG_AXIS_ROLL = "long_axis_roll"


def quintic_smoothstep(value: float) -> float:
    """Return a zero-velocity/zero-acceleration interpolation parameter."""
    value = min(max(float(value), 0.0), 1.0)
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def axis_from_tilt_azimuth(tilt_rad: float, azimuth_rad: float) -> np.ndarray:
    """Return a unit axis for tilt from world +Z and azimuth in world XY."""
    return np.array(
        [
            math.sin(tilt_rad) * math.cos(azimuth_rad),
            math.sin(tilt_rad) * math.sin(azimuth_rad),
            math.cos(tilt_rad),
        ],
        dtype=np.float64,
    )


def quaternion_from_y_rotation(angle_rad: float) -> np.ndarray:
    """Return an xyzw quaternion rotating local +Z toward world +X."""
    half = 0.5 * float(angle_rad)
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float64)


def quaternion_from_axis(axis_world: np.ndarray) -> np.ndarray:
    """Return an xyzw quaternion aligning local +Z with a world axis."""
    axis = np.asarray(axis_world, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    dot = float(np.clip(axis[2], -1.0, 1.0))
    if dot > 1.0 - 1.0e-10:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if dot < -1.0 + 1.0e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quaternion = np.array([-axis[1], axis[0], 0.0, 1.0 + dot])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def capsule_support_height(axis_world: np.ndarray) -> float:
    """Return capsule center height [m] for ideal contact with a Z-up plane."""
    axis = np.asarray(axis_world, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    half_cylinder = 0.5 * CAPSULE_LENGTH_M - CAPSULE_RADIUS_M
    return CAPSULE_RADIUS_M + half_cylinder * abs(float(axis[2]))


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ball_chain_rotation(ball_joint_positions_rad: np.ndarray) -> np.ndarray:
    """Return ball-chain rotation for Y, X, Z joint axes."""
    qx, qy, qz = np.asarray(ball_joint_positions_rad, dtype=np.float64)
    return _rot_y(float(qx)) @ _rot_x(float(qy)) @ _rot_z(float(qz))


class BallFieldPlanner:
    """Map desired field directions to continuous ball-joint references.

    The main magnet is a cube whose magnetic polarization is local +Z.
    Rotation about that axis therefore does not alter the finite-magnet field.
    The solver searches ballx/​bally and holds ballz continuous.
    """

    def __init__(
        self,
        model,
        main_position_world: np.ndarray,
        main_rotation_world: np.ndarray,
        ball_joint_positions_rad: np.ndarray,
        nominal_target_position_world: np.ndarray,
        ball_default_positions_rad: np.ndarray,
        action_scale_rad: float = math.pi / 2.0,
    ):
        self.model = model
        self.main_position_world = np.asarray(main_position_world, dtype=np.float64)
        self.nominal_target_position_world = np.asarray(
            nominal_target_position_world, dtype=np.float64
        )
        self.default = np.asarray(ball_default_positions_rad, dtype=np.float64)
        self.action_scale_rad = float(action_scale_rad)
        current = np.asarray(ball_joint_positions_rad, dtype=np.float64)
        self.mount_rotation_world = np.asarray(
            main_rotation_world, dtype=np.float64
        ) @ ball_chain_rotation(current).T
        self.current = current.copy()

    def rotation_world(self, ball_joint_positions_rad: np.ndarray) -> np.ndarray:
        """Return main-magnet world rotation for candidate ball joints."""
        return self.mount_rotation_world @ ball_chain_rotation(
            ball_joint_positions_rad
        )

    def field_world(self, ball_joint_positions_rad: np.ndarray) -> np.ndarray:
        """Return field [T] at the fixed nominal capsule-magnet center."""
        values = self.model.field_tesla(
            self.nominal_target_position_world,
            self.main_position_world,
            self.rotation_world(ball_joint_positions_rad),
        )
        return np.asarray(values, dtype=np.float64).reshape(-1, 3)[0]

    def solve(
        self,
        desired_field_direction_world: np.ndarray,
        *,
        global_search: bool = False,
    ) -> tuple[np.ndarray, dict[str, float | list[float]]]:
        """Solve ball joints for one desired field direction.

        Args:
            desired_field_direction_world: Unit or non-unit world direction.
            global_search: Whether to seed the local refinement with a coarse
                full-range search.

        Returns:
            Ball positions [rad] and finite-field diagnostics.
        """
        desired = np.asarray(desired_field_direction_world, dtype=np.float64)
        desired /= max(float(np.linalg.norm(desired)), 1.0e-12)
        qz = float(self.current[2])
        # Respect the action envelope about the reset pose.  The original
        # hard-coded qx=[0, pi] range covers only one magnetic hemisphere:
        # with the Y-X-Z chain the magnet's local +Z direction then has only
        # one sign of its mount-frame X component.  A full 360-degree field
        # revolution therefore stalled at the missing hemisphere.  Composite
        # trajectories widen qx to +/-pi about the reset pose, while the
        # ordinary +/-pi/2 tasks retain their previous range.
        qx_half_range = min(self.action_scale_rad, math.pi)
        qy_half_range = min(self.action_scale_rad, math.pi / 2.0)
        bounds = (
            (
                float(self.default[0] - qx_half_range + 0.02),
                float(self.default[0] + qx_half_range - 0.02),
            ),
            (
                float(self.default[1] - qy_half_range + 0.02),
                float(self.default[1] + qy_half_range - 0.02),
            ),
        )

        def evaluate(qx: float, qy: float) -> tuple[float, np.ndarray]:
            q = np.array([qx, qy, qz], dtype=np.float64)
            field = self.field_world(q)
            magnitude = float(np.linalg.norm(field))
            if magnitude <= 1.0e-12:
                return math.inf, field
            direction = field / magnitude
            direction_error = 1.0 - float(np.clip(np.dot(direction, desired), -1.0, 1.0))
            continuity = 2.0e-4 * float(np.sum((q[:2] - self.current[:2]) ** 2))
            return direction_error + continuity, field

        best = self.current[:2].copy()
        best_cost, best_field = evaluate(float(best[0]), float(best[1]))
        if global_search:
            for qx in np.linspace(bounds[0][0], bounds[0][1], 13):
                for qy in np.linspace(bounds[1][0], bounds[1][1], 13):
                    cost, field = evaluate(float(qx), float(qy))
                    if cost < best_cost:
                        best = np.array([qx, qy])
                        best_cost, best_field = cost, field

        step = 0.35
        for _ in range(18):
            improved = False
            for axis in range(2):
                for sign in (-1.0, 1.0):
                    candidate = best.copy()
                    candidate[axis] = np.clip(
                        candidate[axis] + sign * step,
                        bounds[axis][0],
                        bounds[axis][1],
                    )
                    cost, field = evaluate(float(candidate[0]), float(candidate[1]))
                    if cost + 1.0e-12 < best_cost:
                        best = candidate
                        best_cost, best_field = cost, field
                        improved = True
            if not improved:
                step *= 0.5
            if step < 5.0e-4:
                break

        result = np.array([best[0], best[1], qz], dtype=np.float64)
        self.current = result
        magnitude = float(np.linalg.norm(best_field))
        direction = best_field / max(magnitude, 1.0e-12)
        angle_error = math.degrees(
            math.acos(float(np.clip(np.dot(direction, desired), -1.0, 1.0)))
        )
        diagnostics: dict[str, float | list[float]] = {
            "field_magnitude_T": magnitude,
            "field_direction_world": direction.tolist(),
            "direction_error_deg": angle_error,
        }
        return result, diagnostics

    def action_from_positions(self, positions_rad: np.ndarray) -> np.ndarray:
        """Convert absolute ball positions [rad] into normalized task actions."""
        action = (
            np.asarray(positions_rad, dtype=np.float64) - self.default
        ) / self.action_scale_rad
        return np.clip(action, -1.0, 1.0)


@dataclass(frozen=True)
class ArmGradientPlan:
    """One small Cartesian displacement mapped to arm action space."""

    desired_displacement_world_m: np.ndarray
    joint_delta_rad: np.ndarray
    normalized_action: np.ndarray
    predicted_displacement_world_m: np.ndarray


def arm_gradient_plan(
    jacobian_world: np.ndarray,
    desired_displacement_world_m: np.ndarray,
    *,
    action_scale_rad: float = 0.05,
    max_joint_delta_rad: float = 0.045,
    damping: float = 2.0e-3,
    locked_joint_indices: tuple[int, ...] = (),
    orientation_weight: float = 1.0,
) -> ArmGradientPlan:
    """Map a small magnet translation to a bounded six-axis arm action.

    The six-dimensional target twist uses zero desired orientation change so
    the ball-field solution remains valid during the gradient-force sweep.
    """
    jacobian = np.asarray(jacobian_world, dtype=np.float64)
    if jacobian.shape != (6, 6):
        raise ValueError(f"Expected a 6x6 arm Jacobian, got {jacobian.shape}")
    displacement = np.asarray(desired_displacement_world_m, dtype=np.float64)
    target = np.concatenate((displacement, np.zeros(3, dtype=np.float64)))
    locked = set(int(index) for index in locked_joint_indices)
    if any(index < 0 or index >= 6 for index in locked):
        raise ValueError(f"Invalid locked joint indices: {locked_joint_indices}")
    free = [index for index in range(6) if index not in locked]
    orientation_weight = float(orientation_weight)
    if orientation_weight < 0.0:
        raise ValueError("orientation_weight must be non-negative")
    row_weights = np.array(
        [1.0, 1.0, 1.0, orientation_weight, orientation_weight, orientation_weight],
        dtype=np.float64,
    )
    reduced_jacobian = jacobian[:, free] * row_weights[:, None]
    weighted_target = target * row_weights
    # Use damped least squares in joint space so the solve remains well posed
    # when one wrist redundancy is reserved for mounted-ASM collision
    # clearance. Locked joints receive exactly zero command.
    regularized = (
        reduced_jacobian.T @ reduced_jacobian
        + damping**2 * np.eye(len(free))
    )
    reduced_delta = np.linalg.solve(
        regularized, reduced_jacobian.T @ weighted_target
    )
    joint_delta = np.zeros(6, dtype=np.float64)
    joint_delta[free] = reduced_delta
    peak = float(np.max(np.abs(joint_delta)))
    if peak > max_joint_delta_rad:
        joint_delta *= max_joint_delta_rad / peak
    predicted = jacobian[:3] @ joint_delta
    normalized = np.clip(joint_delta / action_scale_rad, -1.0, 1.0)
    return ArmGradientPlan(displacement, joint_delta, normalized, predicted)
