"""Configuration for the first short-action implementation stage."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class ActionLayerConfig:
    """Command and hard-safety limits shared by pure and Isaac Lab tests.

    Motion increments are provisional values inside the ranges approved by the
    revised design.  They are command magnitudes, not promised capsule motion.
    Stage two must measure and calibrate the physical response independently.
    """

    control_dt_s: float = 1.0 / 20.0
    nominal_duration_s: float = 0.90
    hold_duration_s: float = 1.00
    maximum_duration_s: float = 2.50
    tilt_increment_rad: float = math.radians(9.0)
    azimuth_increment_rad: float = math.radians(12.0)
    roll_displacement_m: float = 0.0075
    approach_displacement_m: float = 0.0030
    theta_min_rad: float = math.radians(5.0)
    theta_max_rad: float = math.radians(175.0)
    azimuth_min_sine: float = math.sin(math.radians(12.0))
    roll_theta_min_rad: float = math.radians(65.0)
    roll_theta_max_rad: float = math.radians(115.0)
    approach_direction_world: tuple[float, float, float] = (0.0, 0.0, -1.0)
    turn_ball_delta_rad: tuple[float, float, float] = (0.0, 0.12, 0.18)
    workspace_min_world_m: tuple[float, float, float] = (0.72, -0.20, 0.04)
    workspace_max_world_m: tuple[float, float, float] = (1.40, 0.55, 0.85)
    joint_limit_margin_rad: float = math.radians(0.25)
    velocity_tolerance_ratio: float = 1.05
    acceleration_tolerance_ratio: float = 1.10
    tracking_error_hard_rad: float = 0.35
    settle_position_tolerance_rad: float = 0.020
    settle_velocity_tolerance_rad_s: float = 0.08
    settle_required_steps: int = 2
    asm_min_clearance_m: float = 0.0
    ground_collision_margin_m: float = 0.0
    trajectory_collision_samples: int = 21
    arm_joint_count: int = 6
    ball_joint_count: int = 3
    # Conservative acceleration limits for planned targets. The first six are
    # inherited from the existing XRDF; the Ball values remain provisional.
    fallback_acceleration_limits_rad_s2: tuple[float, ...] = (
        0.4,
        0.4,
        0.4,
        0.6,
        0.6,
        0.8,
        2.0,
        2.0,
        2.0,
    )
    registered_field_point_world_m: tuple[float, float, float] = (
        1.0608155,
        0.1145374,
        0.0065,
    )
    metadata: dict[str, str] = field(
        default_factory=lambda: {
            "schema": "robotarm_magnetic_action_layer",
            "version": "1.0.0",
            "field_reference_semantics": "registered_workspace_point_not_capsule_truth",
        }
    )

    def __post_init__(self) -> None:
        if self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if not 0.8 <= self.nominal_duration_s <= 1.2:
            raise ValueError("nominal_duration_s must remain in the approved 0.8-1.2 s range")
        if self.maximum_duration_s < self.nominal_duration_s:
            raise ValueError("maximum_duration_s must not be shorter than nominal_duration_s")
        if not self.theta_min_rad < self.theta_max_rad:
            raise ValueError("theta command bounds are invalid")
        if self.trajectory_collision_samples < 2:
            raise ValueError("trajectory_collision_samples must be at least two")
        if len(self.fallback_acceleration_limits_rad_s2) != self.arm_joint_count + self.ball_joint_count:
            raise ValueError("fallback acceleration limits must cover all nine joints")
        minimum = np.asarray(self.workspace_min_world_m, dtype=np.float64)
        maximum = np.asarray(self.workspace_max_world_m, dtype=np.float64)
        if not np.all(minimum < maximum):
            raise ValueError("workspace minimum must be below maximum on every axis")

    @property
    def joint_count(self) -> int:
        return self.arm_joint_count + self.ball_joint_count
