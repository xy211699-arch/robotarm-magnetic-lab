"""Coordinate helpers and deployment-safe command-state initialization."""

from __future__ import annotations

import math

import numpy as np

from .types import DeviceSnapshot, MagnetCommandState


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def field_direction(theta_rad: float, phi_rad: float) -> np.ndarray:
    """Return the commanded field direction in the world frame."""
    theta = float(theta_rad)
    phi = float(phi_rad)
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=np.float64,
    )

def angles_from_field(direction_world: np.ndarray) -> tuple[float, float]:
    """Return polar/azimuth command angles for a non-zero world vector."""
    direction = np.asarray(direction_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("field direction must be non-zero")
    direction /= norm
    theta = math.acos(float(np.clip(direction[2], -1.0, 1.0)))
    phi = wrap_angle(math.atan2(float(direction[1]), float(direction[0])))
    return theta, phi


def roll_direction_from_field(direction_world: np.ndarray) -> np.ndarray:
    """Return a command-frame rolling direction without capsule feedback."""
    direction = np.asarray(direction_world, dtype=np.float64).reshape(3)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    roll = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    norm = float(np.linalg.norm(roll))
    if norm <= 1.0e-8:
        # At a command pole, azimuth is singular. Preserve a deterministic
        # reference; ROLL remains masked until theta enters its valid band.
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return roll / norm


def initial_command_state(
    snapshot: DeviceSnapshot,
    field_direction_world: np.ndarray,
    *,
    arm_joint_count: int = 6,
) -> MagnetCommandState:
    """Build the cumulative target from encoder/FK signals only."""
    direction = np.asarray(field_direction_world, dtype=np.float64).reshape(3)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    theta, phi = angles_from_field(direction)
    return MagnetCommandState(
        theta_rad=theta,
        phi_rad=phi,
        field_direction_world=direction,
        arm_joint_target_rad=snapshot.joint_position_rad[:arm_joint_count],
        ball_joint_target_rad=snapshot.joint_position_rad[arm_joint_count:arm_joint_count + 3],
        magnet_position_target_world_m=snapshot.magnet_position_world_m,
        magnet_rotation_target_world=snapshot.magnet_rotation_world,
        roll_direction_world=roll_direction_from_field(direction),
    )
