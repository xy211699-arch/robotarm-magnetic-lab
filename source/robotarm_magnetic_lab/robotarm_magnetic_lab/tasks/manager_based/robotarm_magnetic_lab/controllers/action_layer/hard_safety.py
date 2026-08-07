"""Hard device-safety checks for atomic trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ActionLayerConfig
from .kinematics import UrdfXrdfSafetyModel
from .types import DeviceSnapshot, HardFailureCode, TrajectoryPlan


@dataclass(frozen=True)
class SafetyCheck:
    ok: bool
    code: HardFailureCode | None = None
    detail: str = ""
    minimum_asm_clearance_m: float = np.inf


class HardSafetyMonitor:
    """Apply only hardware/environment constraints, never effect grading."""

    def __init__(
        self,
        cfg: ActionLayerConfig,
        kinematics: UrdfXrdfSafetyModel | None = None,
        *,
        validate_ground: bool = False,
    ) -> None:
        self.cfg = cfg
        self.kinematics = kinematics
        self.validate_ground = validate_ground

    def check_snapshot(self, snapshot: DeviceSnapshot) -> SafetyCheck:
        arrays = (
            snapshot.joint_position_rad,
            snapshot.joint_velocity_rad_s,
            snapshot.joint_acceleration_rad_s2,
            snapshot.magnet_position_world_m,
            snapshot.magnet_rotation_world,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            return SafetyCheck(False, HardFailureCode.NONFINITE_STATE, "non-finite device state")
        if not snapshot.controller_connected:
            return SafetyCheck(
                False, HardFailureCode.CONTROLLER_DISCONNECTED, "controller unavailable"
            )
        if snapshot.environment_terminated:
            return SafetyCheck(
                False, HardFailureCode.ENV_TERMINATED, "environment terminated"
            )
        limits = snapshot.joint_position_limits_rad
        if np.any(snapshot.joint_position_rad < limits[:, 0]) or np.any(
            snapshot.joint_position_rad > limits[:, 1]
        ):
            return SafetyCheck(False, HardFailureCode.JOINT_LIMIT, "current joint outside hard limit")
        if np.any(
            np.abs(snapshot.joint_velocity_rad_s)
            > snapshot.joint_velocity_limits_rad_s * self.cfg.velocity_tolerance_ratio
        ):
            return SafetyCheck(False, HardFailureCode.JOINT_VELOCITY, "joint velocity limit")
        # Acceleration remains a hard constraint on the planned command below.
        # The current simulator/device acceleration is retained in the
        # snapshot for logging, but is not a hard-stop signal until a measured
        # actuator threshold and filter have been calibrated. A raw 20 Hz
        # finite difference produced false trips even while HOLD was stable.
        if snapshot.asm_clearance_m < self.cfg.asm_min_clearance_m:
            return SafetyCheck(
                False,
                HardFailureCode.ASM_CLEARANCE,
                f"live ASM clearance {snapshot.asm_clearance_m:.6f} m",
                snapshot.asm_clearance_m,
            )
        return SafetyCheck(True, minimum_asm_clearance_m=snapshot.asm_clearance_m)

    def check_plan(self, plan: TrajectoryPlan, snapshot: DeviceSnapshot) -> SafetyCheck:
        targets = plan.joint_targets_rad
        if not np.isfinite(targets).all() or not np.isfinite(
            plan.magnet_targets_world_m
        ).all():
            return SafetyCheck(False, HardFailureCode.ILLEGAL_TARGET, "non-finite target")
        lower = snapshot.joint_position_limits_rad[:, 0] + self.cfg.joint_limit_margin_rad
        upper = snapshot.joint_position_limits_rad[:, 1] - self.cfg.joint_limit_margin_rad
        if np.any(targets < lower[None, :]) or np.any(targets > upper[None, :]):
            return SafetyCheck(False, HardFailureCode.JOINT_LIMIT, "planned joint limit")
        velocity = np.diff(targets, axis=0) / self.cfg.control_dt_s
        if velocity.size and np.any(
            np.abs(velocity)
            > snapshot.joint_velocity_limits_rad_s[None, :]
            * self.cfg.velocity_tolerance_ratio
        ):
            return SafetyCheck(False, HardFailureCode.JOINT_VELOCITY, "planned velocity limit")
        acceleration = np.diff(velocity, axis=0) / self.cfg.control_dt_s
        if acceleration.size and np.any(
            np.abs(acceleration)
            > snapshot.joint_acceleration_limits_rad_s2[None, :]
            * self.cfg.acceleration_tolerance_ratio
        ):
            return SafetyCheck(False, HardFailureCode.JOINT_ACCELERATION, "planned acceleration limit")
        minimum = np.asarray(self.cfg.workspace_min_world_m)
        maximum = np.asarray(self.cfg.workspace_max_world_m)
        if np.any(plan.magnet_targets_world_m < minimum[None, :]) or np.any(
            plan.magnet_targets_world_m > maximum[None, :]
        ):
            return SafetyCheck(False, HardFailureCode.WORKSPACE_LIMIT, "planned magnet workspace")
        if self.kinematics is not None:
            result = self.kinematics.validate_path(
                targets[:, : self.cfg.arm_joint_count],
                required_asm_clearance_m=self.cfg.asm_min_clearance_m,
                ground_height_m=(
                    self.cfg.ground_collision_margin_m if self.validate_ground else None
                ),
            )
            if not bool(result["ok"]):
                code = HardFailureCode(str(result["kind"]))
                return SafetyCheck(False, code, str(result), float(result.get("minimum_asm_clearance_m", np.inf)))
            return SafetyCheck(
                True,
                minimum_asm_clearance_m=float(result["minimum_asm_clearance_m"]),
            )
        return SafetyCheck(True, minimum_asm_clearance_m=snapshot.asm_clearance_m)
