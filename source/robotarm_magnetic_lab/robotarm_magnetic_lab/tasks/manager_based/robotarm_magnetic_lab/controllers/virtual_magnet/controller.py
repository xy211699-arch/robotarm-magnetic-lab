"""Pure one-second closed-loop virtual-magnet action controller."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from .config import ClosedLoopProfile
from .geometry import move_direction, normalize, quintic_progress, unsigned_axis_tilt, view_target_axis
from .outer_loop import desired_hold_wrench, desired_move_wrench, desired_view_wrench
from .pose_inverse import PoseInverseState, solve_pose_increment
from .types import (
    ActionId,
    ActionResult,
    ControllerCommand,
    ControllerState,
    ControllerTelemetry,
    FrozenActionTarget,
    Lifecycle,
)


def _unit_quaternion(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm <= 1.0e-12:
        raise ValueError("invalid quaternion")
    return quaternion / norm


def _swing_interpolate(start, target, progress: float) -> np.ndarray:
    start_axis = normalize(start)
    target_axis = normalize(target)
    cosine = float(np.clip(np.dot(start_axis, target_axis), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1.0e-10:
        return start_axis
    sine = math.sin(angle)
    fraction = float(np.clip(progress, 0.0, 1.0))
    return normalize(
        math.sin((1.0 - fraction) * angle) / sine * start_axis
        + math.sin(fraction * angle) / sine * target_axis
    )


def _pose_interpolate(position_a, quaternion_a, position_b, quaternion_b, fraction):
    alpha = float(np.clip(fraction, 0.0, 1.0))
    position = (1.0 - alpha) * np.asarray(position_a) + alpha * np.asarray(position_b)
    qa = _unit_quaternion(quaternion_a)
    qb = _unit_quaternion(quaternion_b)
    if np.dot(qa, qb) < 0.0:
        qb = -qb
    relative = Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)
    quaternion = (Rotation.from_quat(qa) * Rotation.from_rotvec(alpha * relative.as_rotvec())).as_quat()
    return np.asarray(position, dtype=np.float64), _unit_quaternion(quaternion)


class VirtualMagnetElevenActionController:
    """Freeze one request, close the loop at 60 Hz, and emit smooth 240 Hz poses."""

    def __init__(
        self,
        profile: ClosedLoopProfile,
        wrench_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        initial_magnet_position,
        initial_magnet_quaternion_xyzw,
    ):
        self.profile = profile
        self._wrench_model = wrench_model
        self._initial_position = np.asarray(initial_magnet_position, dtype=np.float64).reshape(3)
        self._initial_quaternion = _unit_quaternion(initial_magnet_quaternion_xyzw)
        self.total_substeps = 0
        self.reset()

    def reset(self) -> None:
        self.lifecycle = Lifecycle.READY
        self.telemetry = ControllerTelemetry()
        self.frozen_target: FrozenActionTarget | None = None
        self._requested_action: ActionId | None = None
        self._rejected_move = False
        self._action_substep = 0
        self._feedback_updates = 0
        self._current_position = self._initial_position.copy()
        self._current_quaternion = self._initial_quaternion.copy()
        self._interpolation_start_position = self._current_position.copy()
        self._interpolation_start_quaternion = self._current_quaternion.copy()
        self._interpolation_target_position = self._current_position.copy()
        self._interpolation_target_quaternion = self._current_quaternion.copy()
        self._command: ControllerCommand | None = None
        self._linear_speed_history: list[float] = []
        self._angular_speed_history: list[float] = []

    @property
    def command(self) -> ControllerCommand:
        if self._command is None:
            raise RuntimeError("controller has not produced a command")
        return self._command

    def acknowledge(self) -> None:
        if self.lifecycle == Lifecycle.TERMINAL:
            self.lifecycle = Lifecycle.READY
            self.telemetry.lifecycle = Lifecycle.READY

    def _state_finite(self, state: ControllerState) -> bool:
        arrays = (
            state.capsule_position,
            state.capsule_rotation,
            state.capsule_magnet_position,
            state.capsule_magnet_rotation,
            state.linear_velocity,
            state.angular_velocity,
            state.optical_axis,
            state.camera_up,
            state.camera_right,
            state.long_axis,
            state.inward_normal,
            state.contact_point,
        )
        return all(np.isfinite(np.asarray(item)).all() for item in arrays)

    def submit(self, action_id: int | ActionId, state: ControllerState) -> bool:
        if self.lifecycle != Lifecycle.READY or not self._state_finite(state):
            return False
        action = ActionId(int(action_id))
        target_axis = view_target_axis(
            state.optical_axis,
            state.camera_up,
            state.camera_right,
            action if action.value <= ActionId.VIEW_UP_LEFT else ActionId.HOLD_VIEW,
            self.profile.view_cone_deg,
        )
        eligible = True
        direction = np.zeros(3)
        if action in (ActionId.MOVE_SIDE_POS, ActionId.MOVE_SIDE_NEG):
            tilt = math.degrees(unsigned_axis_tilt(state.long_axis, state.inward_normal))
            recent = self.total_substeps - int(state.last_sidewall_contact_substep) <= self.profile.contact_window_substeps
            eligible = tilt >= self.profile.move_tilt_min_deg and bool(state.sidewall_contact) and recent
            if eligible:
                direction = move_direction(
                    state.long_axis,
                    state.inward_normal,
                    1 if action == ActionId.MOVE_SIDE_POS else -1,
                )
        self.frozen_target = FrozenActionTarget(
            action_id=action,
            start_optical_axis=normalize(state.optical_axis),
            target_optical_axis=target_axis,
            camera_up=normalize(state.camera_up),
            camera_right=normalize(state.camera_right),
            inward_normal=normalize(state.inward_normal),
            tangent_anchor=np.asarray(state.capsule_position, dtype=np.float64).copy(),
            start_position=np.asarray(state.capsule_position, dtype=np.float64).copy(),
            move_direction=direction,
            move_eligible=eligible,
        )
        self._requested_action = action
        self._rejected_move = action in (ActionId.MOVE_SIDE_POS, ActionId.MOVE_SIDE_NEG) and not eligible
        self._action_substep = 0
        self._feedback_updates = 0
        self._linear_speed_history.clear()
        self._angular_speed_history.clear()
        self.lifecycle = Lifecycle.EXECUTING
        self.telemetry = ControllerTelemetry(lifecycle=Lifecycle.EXECUTING, action_id=action)
        return True

    def _desired_wrench(self, state: ControllerState) -> np.ndarray:
        assert self.frozen_target is not None and self._requested_action is not None
        motion_fraction = min((self._action_substep + 1) / self.profile.motion_substeps, 1.0)
        progress = quintic_progress(motion_fraction)
        action = self._requested_action
        if self._rejected_move or action == ActionId.HOLD_VIEW:
            return desired_hold_wrench(
                optical_axis=state.optical_axis,
                target_optical_axis=self.frozen_target.start_optical_axis,
                position=state.capsule_position,
                tangent_anchor=self.frozen_target.tangent_anchor,
                inward_normal=self.frozen_target.inward_normal,
                linear_velocity=state.linear_velocity,
                angular_velocity=state.angular_velocity,
                profile=self.profile,
            )
        if action.value <= ActionId.VIEW_UP_LEFT:
            target_axis = _swing_interpolate(
                self.frozen_target.start_optical_axis,
                self.frozen_target.target_optical_axis,
                progress,
            )
            if state.camera_contact and np.dot(target_axis - state.optical_axis, self.frozen_target.inward_normal) < 0.0:
                target_axis = normalize(state.optical_axis)
                self.telemetry.constrained = True
            return desired_view_wrench(
                optical_axis=state.optical_axis,
                target_optical_axis=target_axis,
                position=state.capsule_position,
                tangent_anchor=self.frozen_target.tangent_anchor,
                inward_normal=self.frozen_target.inward_normal,
                linear_velocity=state.linear_velocity,
                angular_velocity=state.angular_velocity,
                profile=self.profile,
            )
        target_position = (
            self.frozen_target.start_position
            + progress * self.profile.move_target_m * self.frozen_target.move_direction
        )
        return desired_move_wrench(
            position=state.capsule_position,
            target_position=target_position,
            start_position=self.frozen_target.start_position,
            move_direction=self.frozen_target.move_direction,
            inward_normal=self.frozen_target.inward_normal,
            linear_velocity=state.linear_velocity,
            profile=self.profile,
        )

    def _feedback_update(self, state: ControllerState, desired_wrench: np.ndarray) -> None:
        if not self.profile.feedback_enabled:
            return
        capsule_rotation = np.asarray(state.capsule_magnet_rotation, dtype=np.float64).reshape(3, 3)
        nominal_position = np.asarray(state.capsule_magnet_position) + capsule_rotation @ np.asarray(
            self.profile.nominal_position_capsule_m
        )
        nominal_rotation = capsule_rotation @ Rotation.from_quat(
            self.profile.nominal_quaternion_capsule_xyzw
        ).as_matrix()
        inverse_state = PoseInverseState(
            position=self._current_position,
            quaternion_xyzw=self._current_quaternion,
            capsule_magnet_position=state.capsule_magnet_position,
            capsule_magnet_rotation=capsule_rotation,
            nominal_position=nominal_position,
            nominal_quaternion_xyzw=Rotation.from_matrix(nominal_rotation).as_quat(),
        )
        result = solve_pose_increment(
            self._wrench_model,
            inverse_state,
            desired_wrench,
            weights=np.asarray(self.profile.force_weights + self.profile.torque_weights),
            translation_step_m=self.profile.translation_fd_step_m,
            rotation_step_rad=self.profile.rotation_fd_step_rad,
            damping=self.profile.inverse_damping,
            relative_regularization=self.profile.relative_regularization,
            translation_trust_m=self.profile.translation_trust_m,
            rotation_trust_rad=self.profile.rotation_trust_rad,
            minimum_separation_m=self.profile.minimum_separation_m,
            maximum_separation_m=self.profile.maximum_separation_m,
            maximum_relative_angle_rad=self.profile.maximum_relative_angle_rad,
            condition_limit=self.profile.condition_limit,
        )
        self._interpolation_start_position = self._current_position.copy()
        self._interpolation_start_quaternion = self._current_quaternion.copy()
        self._interpolation_target_position = result.position.copy()
        self._interpolation_target_quaternion = result.quaternion_xyzw.copy()
        self.telemetry.solver_saturated |= result.solver_saturated
        self._feedback_updates += 1

    def _finish(self, state: ControllerState) -> None:
        assert self.frozen_target is not None and self._requested_action is not None
        target_axis = (
            self.frozen_target.start_optical_axis
            if self._rejected_move or self._requested_action == ActionId.HOLD_VIEW
            else self.frozen_target.target_optical_axis
        )
        axis_cosine = float(np.clip(np.dot(normalize(state.optical_axis), normalize(target_axis)), -1.0, 1.0))
        self.telemetry.optical_axis_error_deg = math.degrees(math.acos(axis_cosine))
        tangent_delta = np.asarray(state.capsule_position) - self.frozen_target.tangent_anchor
        tangent_delta -= self.frozen_target.inward_normal * np.dot(tangent_delta, self.frozen_target.inward_normal)
        self.telemetry.tangent_drift_m = float(np.linalg.norm(tangent_delta))
        if self._requested_action in (ActionId.MOVE_SIDE_POS, ActionId.MOVE_SIDE_NEG) and self.frozen_target.move_eligible:
            self.telemetry.move_signed_displacement_m = float(
                np.dot(np.asarray(state.capsule_position) - self.frozen_target.start_position, self.frozen_target.move_direction)
            )
        window = self.profile.stability_window_substeps
        self.telemetry.linear_speed_m_s = float(np.mean(self._linear_speed_history[-window:]))
        self.telemetry.angular_speed_rad_s = float(np.mean(self._angular_speed_history[-window:]))
        self.telemetry.low_effect = (
            self.telemetry.linear_speed_m_s > self.profile.boundary_linear_speed_m_s
            or self.telemetry.angular_speed_rad_s > self.profile.boundary_angular_speed_rad_s
        )
        self.telemetry.result = ActionResult.REJECTED if self._rejected_move else ActionResult.COMPLETED
        self.lifecycle = Lifecycle.TERMINAL
        self.telemetry.lifecycle = Lifecycle.TERMINAL

    def step(self, state: ControllerState) -> ControllerCommand | None:
        if self.lifecycle != Lifecycle.EXECUTING:
            return None
        if not self._state_finite(state):
            self.lifecycle = Lifecycle.FAULT
            self.telemetry.lifecycle = Lifecycle.FAULT
            self.telemetry.result = ActionResult.FAULT
            return None
        try:
            desired_wrench = self._desired_wrench(state)
            if self._action_substep % self.profile.feedback_stride == 0:
                self._feedback_update(state, desired_wrench)
            interpolation_phase = self._action_substep % self.profile.feedback_stride
            fraction = (interpolation_phase + 1) / self.profile.feedback_stride
            self._current_position, self._current_quaternion = _pose_interpolate(
                self._interpolation_start_position,
                self._interpolation_start_quaternion,
                self._interpolation_target_position,
                self._interpolation_target_quaternion,
                fraction,
            )
            rotation = Rotation.from_quat(self._current_quaternion).as_matrix()
            model_wrench = np.asarray(self._wrench_model(self._current_position, rotation), dtype=np.float64).reshape(6)
            if not np.isfinite(model_wrench).all() or not np.isfinite(desired_wrench).all():
                raise ValueError("non-finite wrench")
            self._command = ControllerCommand(
                virtual_magnet_position=self._current_position.copy(),
                virtual_magnet_quaternion_xyzw=self._current_quaternion.copy(),
                desired_wrench=desired_wrench.copy(),
                model_wrench=model_wrench.copy(),
                solver_saturated=self.telemetry.solver_saturated,
            )
        except Exception:
            self.lifecycle = Lifecycle.FAULT
            self.telemetry.lifecycle = Lifecycle.FAULT
            self.telemetry.result = ActionResult.FAULT
            return None

        self._linear_speed_history.append(float(np.linalg.norm(state.linear_velocity)))
        self._angular_speed_history.append(float(np.linalg.norm(state.angular_velocity)))
        self._action_substep += 1
        self.total_substeps += 1
        self.telemetry.substep = self._action_substep
        self.telemetry.feedback_updates = self._feedback_updates
        self.telemetry.magnetic_wrench = model_wrench.copy()
        self.telemetry.virtual_magnet_relative_position = (
            self._current_position - np.asarray(state.capsule_magnet_position)
        )
        if self._action_substep == self.profile.action_substeps:
            self._finish(state)
        return self._command
