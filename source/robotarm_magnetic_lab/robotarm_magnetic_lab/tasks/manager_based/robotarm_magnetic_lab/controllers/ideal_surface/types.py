"""Frozen data contracts for the privileged ideal-surface controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

import numpy as np


class IdealSurfaceAction(IntEnum):
    HOLD = 0
    START_TILT_000 = 1
    START_TILT_045 = 2
    START_TILT_090 = 3
    START_TILT_135 = 4
    START_TILT_180 = 5
    START_TILT_225 = 6
    START_TILT_270 = 7
    START_TILT_315 = 8
    TILT_MORE = 9
    RISE = 10
    PRECESS_POS = 11
    PRECESS_NEG = 12
    ROLL_POS = 13
    ROLL_NEG = 14


START_TILT_ACTIONS = tuple(IdealSurfaceAction(value) for value in range(1, 9))


class ControllerState(str, Enum):
    READY = "READY"
    EXECUTING = "EXECUTING"
    TERMINAL_FAULT = "TERMINAL_FAULT"


class IdealActionStatus(str, Enum):
    DONE = "DONE"
    HARD_FAILURE = "HARD_FAILURE"


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return result.copy()


@dataclass(frozen=True)
class SurfaceFlags:
    upright: bool
    side_contact: bool
    contact_limited: bool = False
    boundary_limited: bool = False
    no_effect: bool = False


@dataclass(frozen=True)
class ControllerSnapshot:
    sim_time_s: float
    position_world: np.ndarray
    quaternion_for_sim: np.ndarray
    axis_world: np.ndarray
    image_up_world: np.ndarray
    surface_point_world: np.ndarray
    surface_normal_world: np.ndarray
    surface_triangle_id: int
    theta_rad: float
    phi_rad: float
    flags: SurfaceFlags

    def __post_init__(self) -> None:
        for name, size in (
            ("position_world", 3),
            ("quaternion_for_sim", 4),
            ("axis_world", 3),
            ("image_up_world", 3),
            ("surface_point_world", 3),
            ("surface_normal_world", 3),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name))
        for name in ("axis_world", "surface_normal_world"):
            value = getattr(self, name)
            norm = float(np.linalg.norm(value))
            if norm <= 1.0e-12:
                raise ValueError(f"{name} must be non-zero")
            object.__setattr__(self, name, value / norm)
        quaternion = self.quaternion_for_sim
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 1.0e-12:
            raise ValueError("quaternion_for_sim must be non-zero")
        object.__setattr__(self, "quaternion_for_sim", quaternion / quaternion_norm)
        object.__setattr__(self, "sim_time_s", float(self.sim_time_s))
        object.__setattr__(self, "surface_triangle_id", int(self.surface_triangle_id))
        object.__setattr__(self, "theta_rad", float(self.theta_rad))
        object.__setattr__(self, "phi_rad", float(self.phi_rad))

    @property
    def axis_tangent_world(self) -> np.ndarray:
        tangent = self.axis_world - float(self.axis_world @ self.surface_normal_world) * self.surface_normal_world
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-12:
            return np.zeros(3, dtype=np.float64)
        return tangent / norm


@dataclass(frozen=True)
class IdealActionResult:
    request_id: int
    action: IdealSurfaceAction
    status: IdealActionStatus
    started_at_s: float
    ended_at_s: float
    contact_limited: bool
    boundary_limited: bool
    no_effect: bool
    hard_failure_detail: str | None
    final_position_world: np.ndarray
    final_quaternion_for_sim: np.ndarray
    final_axis_world: np.ndarray
    final_tilt_rad: float
    final_azimuth_rad: float
    maximum_penetration_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", IdealSurfaceAction(self.action))
        object.__setattr__(self, "status", IdealActionStatus(self.status))
        for name, size in (
            ("final_position_world", 3),
            ("final_quaternion_for_sim", 4),
            ("final_axis_world", 3),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name))

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.ended_at_s) - float(self.started_at_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": int(self.request_id),
            "action_id": int(self.action),
            "action": self.action.name,
            "status": self.status.value,
            "started_at_s": float(self.started_at_s),
            "ended_at_s": float(self.ended_at_s),
            "duration_s": self.duration_s,
            "contact_limited": bool(self.contact_limited),
            "boundary_limited": bool(self.boundary_limited),
            "no_effect": bool(self.no_effect),
            "hard_failure_detail": self.hard_failure_detail,
            "final_position_world": self.final_position_world.tolist(),
            "final_quaternion_for_sim": self.final_quaternion_for_sim.tolist(),
            "final_axis_world": self.final_axis_world.tolist(),
            "final_tilt_rad": float(self.final_tilt_rad),
            "final_azimuth_rad": float(self.final_azimuth_rad),
            "maximum_penetration_m": float(self.maximum_penetration_m),
        }

