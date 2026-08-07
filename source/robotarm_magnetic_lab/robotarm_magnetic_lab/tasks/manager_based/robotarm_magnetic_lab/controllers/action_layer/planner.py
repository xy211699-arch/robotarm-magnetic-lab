"""Open-loop atomic command planning in device space."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .command_state import field_direction, roll_direction_from_field, wrap_angle
from .config import ActionLayerConfig
from .types import (
    ActionRequest,
    AtomicAction,
    DeviceSnapshot,
    HardFailureCode,
    MagnetCommandState,
    TrajectoryPlan,
)


class PlannerError(RuntimeError):
    def __init__(self, code: HardFailureCode, message: str):
        super().__init__(message)
        self.code = code


BallFieldSolver = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict]]
FieldEvaluator = Callable[[np.ndarray], np.ndarray]
ArmDisplacementSolver = Callable[[DeviceSnapshot, np.ndarray], np.ndarray]


class AtomicCommandPlanner:
    """Translate one atomic ID into a smooth, bounded joint trajectory."""

    def __init__(
        self,
        cfg: ActionLayerConfig,
        *,
        solve_ball_field: BallFieldSolver,
        solve_arm_displacement: ArmDisplacementSolver,
        field_for_ball: FieldEvaluator | None = None,
    ) -> None:
        self.cfg = cfg
        self.solve_ball_field = solve_ball_field
        self.solve_arm_displacement = solve_arm_displacement
        self.field_for_ball = field_for_ball

    def plan(
        self,
        request: ActionRequest,
        snapshot: DeviceSnapshot,
        command: MagnetCommandState,
    ) -> TrajectoryPlan:
        target = command.copy()
        action = request.action
        diagnostics: dict[str, object] = {"action": action.name}
        arm_count = self.cfg.arm_joint_count
        current_q = np.asarray(snapshot.joint_position_rad, dtype=np.float64)
        target_q = current_q.copy()

        if action in (AtomicAction.TILT_POS, AtomicAction.TILT_NEG):
            sign = 1.0 if action is AtomicAction.TILT_POS else -1.0
            target.theta_rad += sign * self.cfg.tilt_increment_rad
            self._solve_field_target(target, target_q, diagnostics)
        elif action in (AtomicAction.AZIMUTH_POS, AtomicAction.AZIMUTH_NEG):
            sign = 1.0 if action is AtomicAction.AZIMUTH_POS else -1.0
            target.phi_rad = wrap_angle(
                target.phi_rad + sign * self.cfg.azimuth_increment_rad
            )
            self._solve_field_target(target, target_q, diagnostics)
        elif action in (AtomicAction.ROLL_POS, AtomicAction.ROLL_NEG):
            sign = 1.0 if action is AtomicAction.ROLL_POS else -1.0
            displacement = (
                sign
                * self.cfg.roll_displacement_m
                * roll_direction_from_field(target.field_direction_world)
            )
            self._solve_arm_target(target, target_q, snapshot, displacement)
            diagnostics["displacement_world_m"] = displacement.tolist()
        elif action in (AtomicAction.APPROACH, AtomicAction.RETREAT):
            sign = 1.0 if action is AtomicAction.APPROACH else -1.0
            direction = np.asarray(self.cfg.approach_direction_world, dtype=np.float64)
            direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
            displacement = sign * self.cfg.approach_displacement_m * direction
            self._solve_arm_target(target, target_q, snapshot, displacement)
            diagnostics["displacement_world_m"] = displacement.tolist()
        elif action in (AtomicAction.TURN_POS, AtomicAction.TURN_NEG):
            sign = 1.0 if action is AtomicAction.TURN_POS else -1.0
            delta = sign * np.asarray(self.cfg.turn_ball_delta_rad, dtype=np.float64)
            target.ball_joint_target_rad = target.ball_joint_target_rad + delta
            target_q[arm_count : arm_count + self.cfg.ball_joint_count] = (
                target.ball_joint_target_rad
            )
            if self.field_for_ball is not None:
                direction = np.asarray(
                    self.field_for_ball(target.ball_joint_target_rad), dtype=np.float64
                )
                direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
                target.field_direction_world = direction
            target.roll_direction_world = roll_direction_from_field(
                target.field_direction_world
            )
        elif action is not AtomicAction.HOLD:
            raise PlannerError(HardFailureCode.INVALID_ACTION, f"unknown action {action}")

        duration_s = self._required_duration(current_q, target_q, snapshot)
        count = max(2, int(math.ceil(duration_s / self.cfg.control_dt_s)) + 1)
        phase = np.linspace(0.0, 1.0, count)
        blend = phase**3 * (10.0 + phase * (-15.0 + 6.0 * phase))
        joint_targets = current_q + blend[:, None] * (target_q - current_q)
        start_magnet = np.asarray(snapshot.magnet_position_world_m, dtype=np.float64)
        magnet_targets = start_magnet + blend[:, None] * (
            target.magnet_position_target_world_m - start_magnet
        )
        diagnostics["duration_s"] = duration_s
        diagnostics["waypoints"] = count
        return TrajectoryPlan(
            request=request,
            joint_targets_rad=joint_targets,
            magnet_targets_world_m=magnet_targets,
            duration_s=duration_s,
            final_command_state=target,
            diagnostics=diagnostics,
        )

    def _solve_field_target(
        self, target: MagnetCommandState, target_q: np.ndarray, diagnostics: dict
    ) -> None:
        desired = field_direction(target.theta_rad, target.phi_rad)
        try:
            ball, field_diagnostics = self.solve_ball_field(
                desired, target.ball_joint_target_rad
            )
        except Exception as error:
            raise PlannerError(
                HardFailureCode.FIELD_INVERSE_FAILED, str(error)
            ) from error
        ball = np.asarray(ball, dtype=np.float64).reshape(self.cfg.ball_joint_count)
        if not np.isfinite(ball).all():
            raise PlannerError(
                HardFailureCode.FIELD_INVERSE_FAILED, "field inverse returned non-finite joints"
            )
        target.ball_joint_target_rad = ball
        target.field_direction_world = desired
        target_q[self.cfg.arm_joint_count :] = ball
        diagnostics["field_inverse"] = field_diagnostics

    def _solve_arm_target(
        self,
        target: MagnetCommandState,
        target_q: np.ndarray,
        snapshot: DeviceSnapshot,
        displacement: np.ndarray,
    ) -> None:
        try:
            delta = np.asarray(
                self.solve_arm_displacement(snapshot, displacement), dtype=np.float64
            ).reshape(self.cfg.arm_joint_count)
        except Exception as error:
            raise PlannerError(HardFailureCode.PLANNING_FAILED, str(error)) from error
        if not np.isfinite(delta).all():
            raise PlannerError(HardFailureCode.PLANNING_FAILED, "non-finite arm target")
        target.arm_joint_target_rad = target_q[: self.cfg.arm_joint_count] + delta
        target_q[: self.cfg.arm_joint_count] = target.arm_joint_target_rad
        target.magnet_position_target_world_m = (
            target.magnet_position_target_world_m + displacement
        )

    def _required_duration(
        self, current: np.ndarray, target: np.ndarray, snapshot: DeviceSnapshot
    ) -> float:
        delta = np.abs(target - current)
        velocity = np.maximum(np.asarray(snapshot.joint_velocity_limits_rad_s), 1.0e-6)
        acceleration = np.maximum(
            np.asarray(snapshot.joint_acceleration_limits_rad_s2), 1.0e-6
        )
        velocity_time = float(np.max(1.875 * delta / velocity))
        acceleration_time = float(np.max(np.sqrt(5.773503 * delta / acceleration)))
        duration = max(self.cfg.nominal_duration_s, velocity_time, acceleration_time)
        if duration > self.cfg.maximum_duration_s:
            raise PlannerError(
                HardFailureCode.PLANNING_FAILED,
                f"required duration {duration:.3f}s exceeds {self.cfg.maximum_duration_s:.3f}s",
            )
        return self.cfg.hold_duration_s if np.max(delta) < 1.0e-12 else duration
