"""Dependency-light data contracts for the TASK-005 controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


class ElevenActionId(IntEnum):
    """Only IDs exposed to a future actor."""

    HOLD_VIEW = 0
    VIEW_UP = 1
    VIEW_UP_RIGHT = 2
    VIEW_RIGHT = 3
    VIEW_DOWN_RIGHT = 4
    VIEW_DOWN = 5
    VIEW_DOWN_LEFT = 6
    VIEW_LEFT = 7
    VIEW_UP_LEFT = 8
    MOVE_SIDE_POS = 9
    MOVE_SIDE_NEG = 10

    @property
    def is_view(self) -> bool:
        return 1 <= int(self) <= 8

    @property
    def is_move(self) -> bool:
        return int(self) in (9, 10)


class ActionResult(str, Enum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAULT = "fault"


class Lifecycle(str, Enum):
    READY_HOLD = "ready_hold"
    EXECUTING = "executing"
    FAULTED = "faulted"


def immutable_vector(value, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(size).copy()
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class CapsuleState:
    """World-frame COM state plus the geometry-link WXYZ orientation."""

    position_world_m: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_world_m_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_world_m", immutable_vector(self.position_world_m, 3))
        object.__setattr__(self, "quaternion_wxyz", immutable_vector(self.quaternion_wxyz, 4))
        object.__setattr__(
            self, "linear_velocity_world_m_s", immutable_vector(self.linear_velocity_world_m_s, 3)
        )
        object.__setattr__(
            self, "angular_velocity_world_rad_s", immutable_vector(self.angular_velocity_world_rad_s, 3)
        )

    @property
    def is_finite(self) -> bool:
        return all(
            np.isfinite(value).all()
            for value in (
                self.position_world_m,
                self.quaternion_wxyz,
                self.linear_velocity_world_m_s,
                self.angular_velocity_world_rad_s,
            )
        )


@dataclass(frozen=True)
class WrenchCommand:
    force_world_n: np.ndarray
    torque_world_nm: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_world_n", immutable_vector(self.force_world_n, 3))
        object.__setattr__(self, "torque_world_nm", immutable_vector(self.torque_world_nm, 3))

    @classmethod
    def zero(cls) -> "WrenchCommand":
        return cls(np.zeros(3), np.zeros(3))


@dataclass(frozen=True)
class ActionTelemetry:
    """One physics-substep record and eventual action result."""

    lifecycle: Lifecycle
    action_id: ElevenActionId | None
    request_id: int
    substep_index: int
    result: ActionResult | None
    constrained: bool
    direction_degenerate: bool
    start_axis_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    end_axis_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    desired_axis_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    surface_normal_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    support_anchor_world_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    support_drift_m: float = 0.0
    move_direction_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    move_signed_displacement_m: float = 0.0
    any_contact: bool = False
    camera_contact: bool = False
    sidewall_contact: bool = False
    contact_cancel_delay_substeps: int | None = None
    force_world_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_world_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_slew_limited: bool = False
    torque_slew_limited: bool = False
    profile_sha256: str = ""
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "lifecycle", Lifecycle(self.lifecycle))
        if self.action_id is not None:
            object.__setattr__(self, "action_id", ElevenActionId(int(self.action_id)))
        if self.result is not None:
            object.__setattr__(self, "result", ActionResult(self.result))
        for name in (
            "start_axis_world", "end_axis_world", "desired_axis_world", "surface_normal_world",
            "support_anchor_world_m", "move_direction_world", "force_world_n", "torque_world_nm",
        ):
            object.__setattr__(self, name, immutable_vector(getattr(self, name), 3))
        finite_scalars = (self.support_drift_m, self.move_signed_displacement_m)
        if not all(np.isfinite(value) for value in finite_scalars):
            raise ValueError("telemetry scalars must be finite")
        if self.request_id < 0 or self.substep_index < 0:
            raise ValueError("request and substep indices must be nonnegative")
        if self.profile_sha256 and len(self.profile_sha256) != 64:
            raise ValueError("profile_sha256 must be empty or 64 characters")

    @classmethod
    def empty(cls, profile_sha256: str = "") -> "ActionTelemetry":
        return cls(
            lifecycle=Lifecycle.READY_HOLD,
            action_id=None,
            request_id=0,
            substep_index=0,
            result=None,
            constrained=False,
            direction_degenerate=False,
            profile_sha256=profile_sha256,
        )
