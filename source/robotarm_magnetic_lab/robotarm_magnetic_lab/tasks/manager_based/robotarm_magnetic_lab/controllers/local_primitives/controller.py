"""Closed-loop world-wrench controller for local capsule primitives."""

from __future__ import annotations

import math

import numpy as np

from .config import LocalPrimitiveControllerCfg, make_local_primitive_controller_cfg
from .trajectory import (
    WORLD_UP,
    axis_at_tilt,
    azimuth_from_axis,
    desired_axis_sample,
    directed_axis_from_quaternion_wxyz,
    tilt_from_axis,
    wrap_angle,
)
from .types import (
    AxisTarget,
    CapsuleState,
    PrimitiveId,
    PrimitiveStatus,
    PrimitiveTelemetry,
    WrenchCommand,
)


class LocalPrimitiveController:
    """Generate bounded force and torque from live center-of-mass state only."""

    def __init__(self, cfg: LocalPrimitiveControllerCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else make_local_primitive_controller_cfg()
        self.reset()

    def reset(self) -> None:
        self.status = PrimitiveStatus.IDLE
        self.active_primitive: PrimitiveId | None = None
        self.elapsed_s = 0.0
        self.stable_time_s = 0.0
        self.last_request_result = "none"
        self._start_axis = WORLD_UP.copy()
        self._final_axis = WORLD_UP.copy()
        self._direction_xy = np.array([1.0, 0.0], dtype=np.float64)
        self._anchor_xy = np.zeros(2, dtype=np.float64)
        self._initial_cone_phase = 0.0
        self._actual_cone_phase = 0.0
        self._previous_actual_azimuth: float | None = None
        self._cone_tilt_squared_error_sum = 0.0
        self._cone_tilt_sample_count = 0
        self.completion_time_s: float | None = None
        zeros = np.zeros(3)
        self._last_desired = AxisTarget(WORLD_UP, zeros, zeros)
        self._last_actual_axis = WORLD_UP.copy()

    @property
    def anchor_xy(self) -> np.ndarray:
        return self._anchor_xy.copy()

    @staticmethod
    def directed_axis_world(state: CapsuleState) -> np.ndarray:
        return directed_axis_from_quaternion_wxyz(state.quaternion_wxyz)

    def start(self, primitive_id: PrimitiveId, azimuth_rad: float, state: CapsuleState) -> bool:
        """Accept one request if idle/holding and its measured posture is valid."""

        primitive = PrimitiveId(int(primitive_id))
        if self.status == PrimitiveStatus.RUNNING:
            self.last_request_result = "busy"
            return False
        if not state.is_finite:
            self.status = PrimitiveStatus.NONFINITE
            self.last_request_result = "nonfinite"
            return False
        actual_axis = directed_axis_from_quaternion_wxyz(state.quaternion_wxyz)
        tilt = tilt_from_axis(actual_axis)
        if not self._start_gate_allows(primitive, tilt):
            self.status = PrimitiveStatus.INVALID_START
            self.active_primitive = None
            self.last_request_result = "invalid_start"
            return False

        self.active_primitive = primitive
        self.status = PrimitiveStatus.RUNNING
        self.elapsed_s = 0.0
        self.stable_time_s = 0.0
        self.last_request_result = "accepted"
        self._start_axis = actual_axis.copy()
        self.completion_time_s = None
        self._direction_xy = np.array([math.cos(azimuth_rad), math.sin(azimuth_rad)], dtype=np.float64)
        self._final_axis = self._target_axis(primitive, self._direction_xy)
        self._anchor_xy = (
            state.position_world_m[:2]
            - self.cfg.capsule_half_total_length_m * actual_axis[:2]
        )
        self._initial_cone_phase = azimuth_from_axis(actual_axis, float(azimuth_rad))
        self._actual_cone_phase = 0.0
        self._previous_actual_azimuth = self._initial_cone_phase
        self._cone_tilt_squared_error_sum = 0.0
        self._cone_tilt_sample_count = 0
        self._last_actual_axis = actual_axis.copy()
        self._last_desired = self._sample_desired(0.0)
        return True

    def update(self, state: CapsuleState, dt_s: float) -> tuple[WrenchCommand, PrimitiveTelemetry]:
        """Advance one physics step and return a bounded world wrench."""

        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if not state.is_finite:
            self.status = PrimitiveStatus.NONFINITE
            self.active_primitive = None
            return self._zero_command(), self._telemetry(self._last_desired, self._last_actual_axis)

        actual_axis = directed_axis_from_quaternion_wxyz(state.quaternion_wxyz)
        self._last_actual_axis = actual_axis.copy()
        if self.active_primitive is None or self.status in (
            PrimitiveStatus.IDLE,
            PrimitiveStatus.INVALID_START,
            PrimitiveStatus.TIMED_OUT,
            PrimitiveStatus.NONFINITE,
        ):
            return self._zero_command(), self._telemetry(self._last_desired, actual_axis)

        if self.status == PrimitiveStatus.RUNNING:
            self.elapsed_s += dt
            if self.elapsed_s > self.cfg.hard_timeout_s[int(self.active_primitive)]:
                self.status = PrimitiveStatus.TIMED_OUT
                return self._zero_command(), self._telemetry(self._last_desired, actual_axis)

        desired = self._sample_desired(self.elapsed_s)
        self._last_desired = desired
        if self.active_primitive == PrimitiveId.CONE_30_DEG_ONE_REVOLUTION:
            self._observe_cone(actual_axis)

        command = self._compute_wrench(state, actual_axis, desired)
        if self.elapsed_s >= self.cfg.motion_duration_s[int(self.active_primitive)]:
            if self._is_stable(state, actual_axis):
                self.stable_time_s += dt
            else:
                self.stable_time_s = 0.0
            if self.stable_time_s >= self.cfg.stable_duration_s and self._completion_metrics_pass():
                self.status = PrimitiveStatus.SUCCEEDED_HOLDING
                if self.completion_time_s is None:
                    self.completion_time_s = self.elapsed_s

        return command, self._telemetry(desired, actual_axis)

    def _sample_desired(self, elapsed_s: float) -> AxisTarget:
        assert self.active_primitive is not None
        duration = self.cfg.motion_duration_s[int(self.active_primitive)]
        return desired_axis_sample(
            self.active_primitive, self._start_axis, self._initial_cone_phase,
            elapsed_s, duration,
        )

    def _compute_wrench(
        self,
        state: CapsuleState,
        actual_axis: np.ndarray,
        desired: AxisTarget,
    ) -> WrenchCommand:
        omega = state.angular_velocity_world_rad_s
        omega_perpendicular = omega - float(np.dot(omega, actual_axis)) * actual_axis
        desired_omega = desired.angular_velocity_world_rad_s
        torque = (
            self.cfg.axis_kp_nm_per_rad * np.cross(actual_axis, desired.axis_world)
            + self.cfg.axis_kd_nms_per_rad * (desired_omega - omega_perpendicular)
            - self.cfg.roll_damping_nms_per_rad * float(np.dot(omega, actual_axis)) * actual_axis
        )
        torque = _clip_norm(torque, self.cfg.torque_limit_nm)

        desired_position_xy = (
            self._anchor_xy
            + self.cfg.capsule_half_total_length_m * desired.axis_world[:2]
        )
        desired_velocity_xy = self.cfg.capsule_half_total_length_m * desired.axis_dot_world_s[:2]
        force_xy = (
            self.cfg.anchor_kp_n_per_m * (desired_position_xy - state.position_world_m[:2])
            + self.cfg.anchor_kd_ns_per_m * (desired_velocity_xy - state.linear_velocity_world_m_s[:2])
        )
        force_xy = _clip_norm(force_xy, self.cfg.xy_force_limit_n)
        force = np.array([force_xy[0], force_xy[1], -self.cfg.downward_preload_n], dtype=np.float64)
        return WrenchCommand(force, torque)

    def _start_gate_allows(self, code: PrimitiveId, tilt_rad: float) -> bool:
        if code == PrimitiveId.SIDE_TO_UPRIGHT:
            return self.cfg.side_start_min_rad <= tilt_rad <= self.cfg.side_start_max_rad
        if code in (PrimitiveId.UPRIGHT_TO_SIDE, PrimitiveId.UPRIGHT_TO_30_DEG):
            return tilt_rad <= self.cfg.upright_start_max_rad
        return abs(tilt_rad - self.cfg.target_tilt_rad) <= self.cfg.cone_start_tolerance_rad

    def _target_axis(self, code: PrimitiveId, direction_xy: np.ndarray) -> np.ndarray:
        if code == PrimitiveId.SIDE_TO_UPRIGHT:
            return WORLD_UP.copy()
        if code == PrimitiveId.UPRIGHT_TO_SIDE:
            return axis_at_tilt(math.pi / 2.0, direction_xy)
        if code == PrimitiveId.UPRIGHT_TO_30_DEG:
            return axis_at_tilt(self.cfg.target_tilt_rad, direction_xy)
        return self._start_axis.copy()

    def _is_stable(self, state: CapsuleState, actual_axis: np.ndarray) -> bool:
        tolerance = (
            self.cfg.tilt_tolerance_rad
            if self.active_primitive in (PrimitiveId.UPRIGHT_TO_30_DEG, PrimitiveId.CONE_30_DEG_ONE_REVOLUTION)
            else self.cfg.transition_tolerance_rad
        )
        axis_error = math.acos(float(np.clip(np.dot(actual_axis, self._last_desired.axis_world), -1.0, 1.0)))
        return (
            axis_error <= tolerance
            and float(np.linalg.norm(state.linear_velocity_world_m_s)) < self.cfg.max_stable_linear_speed_m_s
            and float(np.linalg.norm(state.angular_velocity_world_rad_s)) < self.cfg.max_stable_angular_speed_rad_s
        )

    def _observe_cone(self, actual_axis: np.ndarray) -> None:
        actual_azimuth = azimuth_from_axis(actual_axis, self._initial_cone_phase)
        if self._previous_actual_azimuth is not None:
            self._actual_cone_phase += wrap_angle(actual_azimuth - self._previous_actual_azimuth)
        self._previous_actual_azimuth = actual_azimuth
        error = tilt_from_axis(actual_axis) - self.cfg.target_tilt_rad
        self._cone_tilt_squared_error_sum += error * error
        self._cone_tilt_sample_count += 1

    def _completion_metrics_pass(self) -> bool:
        if self.active_primitive != PrimitiveId.CONE_30_DEG_ONE_REVOLUTION:
            return True
        return (
            self._actual_cone_phase >= 2.0 * math.pi - self.cfg.cone_coverage_tolerance_rad
            and self._cone_rmse() <= self.cfg.cone_tilt_rmse_limit_rad
        )

    def _cone_rmse(self) -> float:
        if self._cone_tilt_sample_count == 0:
            return 0.0
        return math.sqrt(self._cone_tilt_squared_error_sum / self._cone_tilt_sample_count)

    def _telemetry(self, desired: AxisTarget, actual_axis: np.ndarray) -> PrimitiveTelemetry:
        desired_tilt = tilt_from_axis(desired.axis_world)
        actual_tilt = tilt_from_axis(actual_axis)
        desired_azimuth = azimuth_from_axis(desired.axis_world)
        actual_azimuth = azimuth_from_axis(actual_axis, desired_azimuth)
        return PrimitiveTelemetry(
            status=self.status,
            active_primitive=self.active_primitive,
            elapsed_s=self.elapsed_s,
            desired_axis_world=desired.axis_world,
            actual_axis_world=actual_axis,
            tilt_error_rad=actual_tilt - desired_tilt,
            azimuth_error_rad=wrap_angle(actual_azimuth - desired_azimuth),
            stable_time_s=self.stable_time_s,
            cone_phase_rad=self._actual_cone_phase,
            cone_tilt_rmse_rad=self._cone_rmse(),
            last_request_result=self.last_request_result,
            completion_time_s=self.completion_time_s,
        )

    @staticmethod
    def _zero_command() -> WrenchCommand:
        return WrenchCommand(np.zeros(3), np.zeros(3))


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm <= 1.0e-15:
        return np.asarray(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) * (limit / norm)
