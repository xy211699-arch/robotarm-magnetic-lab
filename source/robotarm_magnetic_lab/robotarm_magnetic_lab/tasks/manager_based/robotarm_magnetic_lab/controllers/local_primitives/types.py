"""Pure data contracts for local capsule dynamics primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


class PrimitiveId(IntEnum):
    """Externally visible primitive identifiers."""

    SIDE_TO_UPRIGHT = 0
    UPRIGHT_TO_SIDE = 1
    UPRIGHT_TO_30_DEG = 2
    CONE_30_DEG_ONE_REVOLUTION = 3


class PrimitiveStatus(str, Enum):
    """Lifecycle state of one primitive request."""

    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED_HOLDING = "succeeded_holding"
    INVALID_START = "invalid_start"
    TIMED_OUT = "timed_out"
    NONFINITE = "nonfinite"


@dataclass(frozen=True)
class CapsuleState:
    """World-frame center-of-mass state using a WXYZ quaternion."""

    sim_time_s: float
    position_world_m: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_world_m_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "sim_time_s", float(self.sim_time_s))
        object.__setattr__(self, "position_world_m", _vector(self.position_world_m, 3))
        object.__setattr__(self, "quaternion_wxyz", _vector(self.quaternion_wxyz, 4))
        object.__setattr__(self, "linear_velocity_world_m_s", _vector(self.linear_velocity_world_m_s, 3))
        object.__setattr__(self, "angular_velocity_world_rad_s", _vector(self.angular_velocity_world_rad_s, 3))

    @property
    def is_finite(self) -> bool:
        return all(
            np.isfinite(value).all()
            for value in (
                np.array([self.sim_time_s]),
                self.position_world_m,
                self.quaternion_wxyz,
                self.linear_velocity_world_m_s,
                self.angular_velocity_world_rad_s,
            )
        )


@dataclass(frozen=True)
class PrimitiveRequest:
    """One edge-triggered command and its desired world azimuth."""

    primitive_id: PrimitiveId
    azimuth_rad: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "primitive_id", PrimitiveId(int(self.primitive_id)))
        azimuth = float(self.azimuth_rad)
        if not np.isfinite(azimuth):
            raise ValueError("azimuth_rad must be finite")
        object.__setattr__(self, "azimuth_rad", azimuth)


@dataclass(frozen=True)
class AxisTarget:
    """Desired directed capsule axis and its time derivative."""

    axis_world: np.ndarray
    axis_dot_world_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray
    unwrapped_target_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        axis = _vector(self.axis_world, 3)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise ValueError("axis_world must be finite and nonzero")
        object.__setattr__(self, "axis_world", axis / norm)
        object.__setattr__(self, "axis_dot_world_s", _vector(self.axis_dot_world_s, 3))
        object.__setattr__(self, "angular_velocity_world_rad_s", _vector(self.angular_velocity_world_rad_s, 3))
        object.__setattr__(self, "unwrapped_target_phase_rad", float(self.unwrapped_target_phase_rad))


@dataclass(frozen=True)
class WrenchCommand:
    """Bounded world-frame force and torque command."""

    force_world_n: np.ndarray
    torque_world_nm: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_world_n", _vector(self.force_world_n, 3))
        object.__setattr__(self, "torque_world_nm", _vector(self.torque_world_nm, 3))


@dataclass(frozen=True)
class PrimitiveTelemetry:
    """Controller output and status for logging and quantitative validation."""

    status: PrimitiveStatus
    active_primitive: PrimitiveId | None
    elapsed_s: float
    desired_axis_world: np.ndarray
    actual_axis_world: np.ndarray
    tilt_error_rad: float
    azimuth_error_rad: float
    stable_time_s: float
    cone_phase_rad: float
    cone_tilt_rmse_rad: float
    last_request_result: str
    completion_time_s: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", PrimitiveStatus(self.status))
        if self.active_primitive is not None:
            object.__setattr__(self, "active_primitive", PrimitiveId(int(self.active_primitive)))
        object.__setattr__(self, "desired_axis_world", _vector(self.desired_axis_world, 3))
        object.__setattr__(self, "actual_axis_world", _vector(self.actual_axis_world, 3))


def _vector(value: np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(size).copy()
    vector.setflags(write=False)
    return vector


# Compatibility alias for the short-lived development spelling; public code uses PrimitiveId.
PrimitiveCode = PrimitiveId
