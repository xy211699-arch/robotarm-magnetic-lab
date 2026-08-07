"""Deployment-safe data contracts for the magnetic short-action layer.

The executor types deliberately contain no capsule pose, velocity, contact,
force, torque, depth, stomach mesh, or coverage fields.  Capsule truth belongs
to training-only evaluators and must never leak through this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import numpy as np


class AtomicAction(IntEnum):
    """Frozen Actor action IDs from the 2026-08-08 design specification."""

    HOLD = 0
    TILT_POS = 1
    TILT_NEG = 2
    AZIMUTH_POS = 3
    AZIMUTH_NEG = 4
    ROLL_POS = 5
    ROLL_NEG = 6
    TURN_POS = 7
    TURN_NEG = 8
    APPROACH = 9
    RETREAT = 10


class ExecutionState(str, Enum):
    """Externally visible states of one atomic action."""

    IDLE = "IDLE"
    PRECHECK = "PRECHECK"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    DONE = "DONE"
    HARD_FAILURE = "HARD_FAILURE"
    SAFE_RECOVER = "SAFE_RECOVER"


class ActionStatus(str, Enum):
    """Only the two outcomes allowed in the revised specification."""

    DONE = "DONE"
    HARD_FAILURE = "HARD_FAILURE"


class HardFailureCode(str, Enum):
    """Device-level reasons for a hard action-layer failure."""

    INVALID_ACTION = "INVALID_ACTION"
    ACTION_MASKED = "ACTION_MASKED"
    BUSY = "BUSY"
    NONFINITE_STATE = "NONFINITE_STATE"
    ILLEGAL_TARGET = "ILLEGAL_TARGET"
    JOINT_LIMIT = "JOINT_LIMIT"
    JOINT_VELOCITY = "JOINT_VELOCITY"
    JOINT_ACCELERATION = "JOINT_ACCELERATION"
    WORKSPACE_LIMIT = "WORKSPACE_LIMIT"
    SELF_COLLISION = "SELF_COLLISION"
    ENVIRONMENT_COLLISION = "ENVIRONMENT_COLLISION"
    ASM_CLEARANCE = "ASM_CLEARANCE"
    FIELD_INVERSE_FAILED = "FIELD_INVERSE_FAILED"
    PLANNING_FAILED = "PLANNING_FAILED"
    TRACKING_ERROR = "TRACKING_ERROR"
    CONTROLLER_DISCONNECTED = "CONTROLLER_DISCONNECTED"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    ENV_TERMINATED = "ENV_TERMINATED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result.copy()


@dataclass
class DeviceSnapshot:
    """Signals available to a deployed robot/external-magnet controller."""

    sim_time_s: float
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    joint_acceleration_rad_s2: np.ndarray
    joint_position_limits_rad: np.ndarray
    joint_velocity_limits_rad_s: np.ndarray
    joint_acceleration_limits_rad_s2: np.ndarray
    magnet_position_world_m: np.ndarray
    magnet_rotation_world: np.ndarray
    asm_clearance_m: float = np.inf
    controller_connected: bool = True
    environment_terminated: bool = False

    def __post_init__(self) -> None:
        joint_count = np.asarray(self.joint_position_rad).size
        self.joint_position_rad = _array(
            self.joint_position_rad, (joint_count,), "joint_position_rad"
        )
        self.joint_velocity_rad_s = _array(
            self.joint_velocity_rad_s, (joint_count,), "joint_velocity_rad_s"
        )
        self.joint_acceleration_rad_s2 = _array(
            self.joint_acceleration_rad_s2,
            (joint_count,),
            "joint_acceleration_rad_s2",
        )
        self.joint_position_limits_rad = _array(
            self.joint_position_limits_rad,
            (joint_count, 2),
            "joint_position_limits_rad",
        )
        self.joint_velocity_limits_rad_s = _array(
            self.joint_velocity_limits_rad_s,
            (joint_count,),
            "joint_velocity_limits_rad_s",
        )
        self.joint_acceleration_limits_rad_s2 = _array(
            self.joint_acceleration_limits_rad_s2,
            (joint_count,),
            "joint_acceleration_limits_rad_s2",
        )
        self.magnet_position_world_m = _array(
            self.magnet_position_world_m, (3,), "magnet_position_world_m"
        )
        self.magnet_rotation_world = _array(
            self.magnet_rotation_world, (3, 3), "magnet_rotation_world"
        )
        self.sim_time_s = float(self.sim_time_s)
        self.asm_clearance_m = float(self.asm_clearance_m)


@dataclass
class MagnetCommandState:
    """Cumulative command state, expressed without capsule truth."""

    theta_rad: float
    phi_rad: float
    field_direction_world: np.ndarray
    arm_joint_target_rad: np.ndarray
    ball_joint_target_rad: np.ndarray
    magnet_position_target_world_m: np.ndarray
    magnet_rotation_target_world: np.ndarray
    roll_direction_world: np.ndarray

    def __post_init__(self) -> None:
        self.theta_rad = float(self.theta_rad)
        self.phi_rad = float(self.phi_rad)
        self.field_direction_world = _array(
            self.field_direction_world, (3,), "field_direction_world"
        )
        norm = float(np.linalg.norm(self.field_direction_world))
        if norm <= 1.0e-12:
            raise ValueError("field_direction_world must be non-zero")
        self.field_direction_world /= norm
        self.arm_joint_target_rad = np.asarray(
            self.arm_joint_target_rad, dtype=np.float64
        ).reshape(-1).copy()
        self.ball_joint_target_rad = _array(
            self.ball_joint_target_rad, (3,), "ball_joint_target_rad"
        )
        self.magnet_position_target_world_m = _array(
            self.magnet_position_target_world_m,
            (3,),
            "magnet_position_target_world_m",
        )
        self.magnet_rotation_target_world = _array(
            self.magnet_rotation_target_world,
            (3, 3),
            "magnet_rotation_target_world",
        )
        self.roll_direction_world = _array(
            self.roll_direction_world, (3,), "roll_direction_world"
        )
        roll_norm = float(np.linalg.norm(self.roll_direction_world))
        if roll_norm > 1.0e-12:
            self.roll_direction_world /= roll_norm

    def copy(self) -> "MagnetCommandState":
        """Return a deep copy suitable for cumulative target updates."""
        return MagnetCommandState(
            theta_rad=self.theta_rad,
            phi_rad=self.phi_rad,
            field_direction_world=self.field_direction_world.copy(),
            arm_joint_target_rad=self.arm_joint_target_rad.copy(),
            ball_joint_target_rad=self.ball_joint_target_rad.copy(),
            magnet_position_target_world_m=self.magnet_position_target_world_m.copy(),
            magnet_rotation_target_world=self.magnet_rotation_target_world.copy(),
            roll_direction_world=self.roll_direction_world.copy(),
        )


@dataclass(frozen=True)
class ActionRequest:
    """One high-level decision submitted at an action boundary."""

    request_id: int
    action: AtomicAction
    requested_at_s: float


@dataclass
class TrajectoryPlan:
    """A finite 20 Hz joint-target trajectory for one action."""

    request: ActionRequest
    joint_targets_rad: np.ndarray
    magnet_targets_world_m: np.ndarray
    duration_s: float
    final_command_state: MagnetCommandState
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joint_targets_rad = np.asarray(
            self.joint_targets_rad, dtype=np.float64
        ).copy()
        self.magnet_targets_world_m = np.asarray(
            self.magnet_targets_world_m, dtype=np.float64
        ).copy()
        if self.joint_targets_rad.ndim != 2:
            raise ValueError("joint_targets_rad must be a 2-D array")
        if self.magnet_targets_world_m.shape != (
            self.joint_targets_rad.shape[0],
            3,
        ):
            raise ValueError("magnet target count must match joint target count")
        self.duration_s = float(self.duration_s)


@dataclass
class ActionResult:
    """Device execution result; deliberately contains no effect grade."""

    request_id: int
    action: AtomicAction
    status: ActionStatus
    requested_at_s: float
    started_at_s: float
    ended_at_s: float
    control_steps: int
    final_joint_position_rad: np.ndarray
    final_ball_position_rad: np.ndarray
    final_magnet_position_world_m: np.ndarray
    final_magnet_rotation_world: np.ndarray
    minimum_asm_clearance_m: float
    state_timestamps_s: dict[str, float]
    hard_failure_code: HardFailureCode | None = None
    hard_failure_detail: str | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.ended_at_s - self.started_at_s))

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON-compatible history entry for logging/training."""
        return {
            "request_id": int(self.request_id),
            "action_id": int(self.action),
            "action": self.action.name,
            "status": self.status.value,
            "requested_at_s": float(self.requested_at_s),
            "started_at_s": float(self.started_at_s),
            "ended_at_s": float(self.ended_at_s),
            "duration_s": self.duration_s,
            "control_steps": int(self.control_steps),
            "final_joint_position_rad": np.asarray(
                self.final_joint_position_rad
            ).tolist(),
            "final_ball_position_rad": np.asarray(
                self.final_ball_position_rad
            ).tolist(),
            "final_magnet_position_world_m": np.asarray(
                self.final_magnet_position_world_m
            ).tolist(),
            "final_magnet_rotation_world": np.asarray(
                self.final_magnet_rotation_world
            ).tolist(),
            "minimum_asm_clearance_m": float(self.minimum_asm_clearance_m),
            "state_timestamps_s": dict(self.state_timestamps_s),
            "hard_failure_code": (
                None if self.hard_failure_code is None else self.hard_failure_code.value
            ),
            "hard_failure_detail": self.hard_failure_detail,
        }


@dataclass
class ExecutorStep:
    """One 20 Hz output from the non-blocking executor."""

    state: ExecutionState
    joint_target_rad: np.ndarray
    accepted_request: bool = False
    result: ActionResult | None = None
