"""Pure data contracts for local capsule dynamics primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class PrimitiveCode(IntEnum):
    """Externally visible primitive identifiers."""

    SIDE_TO_UPRIGHT = 0
    UPRIGHT_TO_SIDE = 1
    UPRIGHT_TO_TILT = 2
    TILT_CONE_REVOLUTION = 3


class PrimitiveStatus(IntEnum):
    """Lifecycle state of one primitive request."""

    IDLE = 0
    RUNNING = 1
    HOLDING = 2
    COMPLETE = 3
    INVALID_START = 4
    TIMED_OUT = 5
    NONFINITE = 6


@dataclass(frozen=True)
class CapsuleState:
    """World-frame center-of-mass state using a WXYZ quaternion."""

    position_world: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_world", _vector(self.position_world, 3))
        object.__setattr__(self, "quaternion_wxyz", _vector(self.quaternion_wxyz, 4))
        object.__setattr__(self, "linear_velocity_world", _vector(self.linear_velocity_world, 3))
        object.__setattr__(self, "angular_velocity_world", _vector(self.angular_velocity_world, 3))

    @property
    def is_finite(self) -> bool:
        return all(
            np.isfinite(value).all()
            for value in (
                self.position_world,
                self.quaternion_wxyz,
                self.linear_velocity_world,
                self.angular_velocity_world,
            )
        )


@dataclass(frozen=True)
class PrimitiveRequest:
    """One edge-triggered command and its desired world azimuth."""

    code: PrimitiveCode
    direction_xy: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", PrimitiveCode(int(self.code)))
        direction = _vector(self.direction_xy, 2)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError("direction_xy must be finite and nonzero")
        object.__setattr__(self, "direction_xy", direction / norm)


@dataclass(frozen=True)
class AxisTarget:
    """Desired directed capsule axis and its time derivative."""

    axis_world: np.ndarray
    axis_dot_world: np.ndarray
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        axis = _vector(self.axis_world, 3)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError("axis_world must be finite and nonzero")
        object.__setattr__(self, "axis_world", axis / norm)
        object.__setattr__(self, "axis_dot_world", _vector(self.axis_dot_world, 3))
        object.__setattr__(self, "phase_rad", float(self.phase_rad))


@dataclass(frozen=True)
class WrenchCommand:
    """Bounded world-frame force and torque command."""

    force_world: np.ndarray
    torque_world: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_world", _vector(self.force_world, 3))
        object.__setattr__(self, "torque_world", _vector(self.torque_world, 3))


@dataclass(frozen=True)
class PrimitiveTelemetry:
    """Controller output and status for logging and quantitative validation."""

    status: PrimitiveStatus
    active_code: PrimitiveCode | None
    elapsed_s: float
    desired_axis_world: np.ndarray
    actual_axis_world: np.ndarray
    tilt_error_rad: float
    azimuth_error_rad: float
    stable_time_s: float
    cone_phase_rad: float
    cone_tilt_rmse_rad: float
    last_request_result: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PrimitiveStatus(int(self.status)))
        if self.active_code is not None:
            object.__setattr__(self, "active_code", PrimitiveCode(int(self.active_code)))
        object.__setattr__(self, "desired_axis_world", _vector(self.desired_axis_world, 3))
        object.__setattr__(self, "actual_axis_world", _vector(self.actual_axis_world, 3))


def _vector(value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(size).copy()
    vector.setflags(write=False)
    return vector
