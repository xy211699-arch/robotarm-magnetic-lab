"""Dependency-light public and privileged contracts for TASK-007."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


class ActionId(IntEnum):
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


class ActionResult(Enum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAULT = "fault"


class Lifecycle(Enum):
    READY = "ready"
    EXECUTING = "executing"
    TERMINAL = "terminal"
    FAULT = "fault"


@dataclass(frozen=True)
class ControllerState:
    capsule_position: np.ndarray
    capsule_rotation: np.ndarray
    capsule_magnet_position: np.ndarray
    capsule_magnet_rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    optical_axis: np.ndarray
    camera_up: np.ndarray
    camera_right: np.ndarray
    long_axis: np.ndarray
    inward_normal: np.ndarray
    contact_point: np.ndarray
    camera_contact: bool = False
    sidewall_contact: bool = False
    last_sidewall_contact_substep: int = -1_000_000


@dataclass(frozen=True)
class FrozenActionTarget:
    action_id: ActionId
    start_optical_axis: np.ndarray
    target_optical_axis: np.ndarray
    camera_up: np.ndarray
    camera_right: np.ndarray
    inward_normal: np.ndarray
    tangent_anchor: np.ndarray
    start_position: np.ndarray
    move_direction: np.ndarray
    move_eligible: bool


@dataclass(frozen=True)
class ControllerCommand:
    virtual_magnet_position: np.ndarray
    virtual_magnet_quaternion_xyzw: np.ndarray
    desired_wrench: np.ndarray
    model_wrench: np.ndarray
    solver_saturated: bool = False
    inverse_condition_number: float = 0.0


@dataclass
class ControllerTelemetry:
    lifecycle: Lifecycle = Lifecycle.READY
    action_id: ActionId | None = None
    result: ActionResult | None = None
    substep: int = 0
    feedback_updates: int = 0
    constrained: bool = False
    low_effect: bool = False
    solver_saturated: bool = False
    inverse_condition_number: float = 0.0
    optical_axis_error_deg: float = 0.0
    tangent_drift_m: float = 0.0
    move_signed_displacement_m: float = 0.0
    passive_roll_rad: float = 0.0
    linear_speed_m_s: float = 0.0
    angular_speed_rad_s: float = 0.0
    magnetic_wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))
    virtual_magnet_relative_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
