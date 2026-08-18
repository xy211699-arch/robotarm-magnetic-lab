"""Pure NumPy one-second dynamic controller for the eleven public actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import DynamicProfile, dynamic_profile_sha256
from .contact_history import ContactSample, SideContactHistory
from .geometry import (
    FrozenSupportPoint,
    camera_frame,
    capsule_axis_world,
    freeze_support_material_point,
    grid_direction_world,
    normalized,
    reconstruct_material_point,
    support_tangent_error,
    tangent_projection,
    view_target_axis,
)
from .surface_query import TriangleMeshSurfaceQuery
from .trajectory import move_direction, quintic_progress, swing_axis
from .types import (
    ActionResult,
    ActionTelemetry,
    CapsuleState,
    ElevenActionId,
    Lifecycle,
    WrenchCommand,
)


@dataclass(frozen=True)
class ControllerStep:
    wrench: WrenchCommand
    telemetry: ActionTelemetry


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(value))
    if length <= float(limit) or length <= 1.0e-15:
        return value.copy()
    return value * (float(limit) / length)


def _slew(previous: np.ndarray, desired: np.ndarray, maximum_delta: float) -> tuple[np.ndarray, bool]:
    delta = np.asarray(desired, dtype=np.float64) - np.asarray(previous, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length <= maximum_delta or length <= 1.0e-15:
        return np.asarray(desired, dtype=np.float64).copy(), False
    return np.asarray(previous, dtype=np.float64) + delta * (maximum_delta / length), True


class ElevenActionController:
    """A deterministic state machine stepped once per 240 Hz physics substep."""

    def __init__(self, profile: DynamicProfile, surface_query: TriangleMeshSurfaceQuery) -> None:
        self.profile = profile
        self.surface_query = surface_query
        self.profile_digest = dynamic_profile_sha256()
        self.contact_history = SideContactHistory(capacity_substeps=profile.contact_history_substeps)
        self.lifecycle = Lifecycle.READY_HOLD
        self.last_result: ActionResult | None = None
        self.discarded_request_count = 0
        self._request_id = 0
        self._latest_physics_substep = 0
        self._action: ElevenActionId | None = None
        self._action_substep = 0
        self._start_axis = np.asarray([0.0, 0.0, 1.0])
        self._target_axis = self._start_axis.copy()
        self._surface_normal = np.asarray([0.0, 0.0, 1.0])
        self._support: FrozenSupportPoint | None = None
        self._rejected_move = False
        self._constrained = False
        self._direction_degenerate = False
        self._move_direction = np.zeros(3)
        self._move_start_position = np.zeros(3)
        self._first_camera_contact_substep: int | None = None
        self._contact_cancel_delay: int | None = None
        self._previous_force = np.zeros(3)
        self._previous_torque = np.zeros(3)
        self._hold_axis: np.ndarray | None = None
        self._hold_support: FrozenSupportPoint | None = None
        self._hold_normal = np.asarray([0.0, 0.0, 1.0])

    @property
    def ready(self) -> bool:
        return self.lifecycle is Lifecycle.READY_HOLD

    def observe_contact(self, sample: ContactSample) -> None:
        self.contact_history.append(sample)

    def reset(self) -> None:
        self.__init__(self.profile, self.surface_query)

    def _freeze_hold(self, state: CapsuleState) -> None:
        hit = self.surface_query.query(state.position_world_m)
        self._hold_axis = capsule_axis_world(state)
        self._hold_normal = hit.normal_world.copy()
        self._hold_support = freeze_support_material_point(
            state,
            hit.normal_world,
            radius_m=self.profile.capsule_radius_m,
            half_length_m=self.profile.capsule_cylinder_half_length_m,
        )

    def submit(self, action_id, state: CapsuleState, *, physics_substep: int) -> bool:
        if self.lifecycle is not Lifecycle.READY_HOLD:
            self.discarded_request_count += 1
            return False
        action = ElevenActionId(int(action_id))
        self._latest_physics_substep = int(physics_substep)
        if not state.is_finite:
            self._enter_fault("non-finite state at submit")
            return False
        hit = self.surface_query.query(state.position_world_m)
        optical, up, right = camera_frame(state)
        self._action = action
        self._action_substep = 0
        self._request_id += 1
        self._start_axis = optical.copy()
        self._target_axis = optical.copy()
        self._surface_normal = hit.normal_world.copy()
        self._support = freeze_support_material_point(
            state,
            hit.normal_world,
            radius_m=self.profile.capsule_radius_m,
            half_length_m=self.profile.capsule_cylinder_half_length_m,
        )
        self._rejected_move = False
        self._constrained = False
        self._direction_degenerate = False
        self._move_direction = np.zeros(3)
        self._move_start_position = state.position_world_m.copy()
        self._first_camera_contact_substep = None
        self._contact_cancel_delay = None

        if action.is_view:
            direction = grid_direction_world(action, up, right)
            candidate = view_target_axis(
                optical, direction, angle_rad=math.radians(self.profile.view_cone_half_angle_deg)
            )
            constraints = self.contact_history.camera_constraints(
                current_substep=physics_substep,
                last_n_substeps=self.profile.contact_history_substeps,
            )
            pushes_inward = any(float((candidate - optical) @ sample.normal_world) < -1.0e-10 for sample in constraints)
            if pushes_inward:
                self._constrained = True
            else:
                self._target_axis = candidate
        elif action.is_move:
            tilt_deg = math.degrees(
                math.acos(np.clip(float(optical @ hit.normal_world), -1.0, 1.0))
            )
            recent_sidewall = self.contact_history.had_sidewall_contact(
                current_substep=physics_substep,
                last_n_substeps=self.profile.contact_history_substeps,
            )
            self._rejected_move = tilt_deg + 1.0e-10 < self.profile.move_min_tilt_deg or not recent_sidewall
        self.lifecycle = Lifecycle.EXECUTING
        self.last_result = None
        return True

    def _enter_fault(self, message: str) -> ControllerStep:
        self.lifecycle = Lifecycle.FAULTED
        self.last_result = ActionResult.FAULT
        telemetry = self._telemetry(
            substep_index=self._action_substep,
            result=ActionResult.FAULT,
            desired_axis=self._target_axis,
            wrench=WrenchCommand.zero(),
            message=message,
        )
        return ControllerStep(WrenchCommand.zero(), telemetry)

    def _support_wrench(
        self,
        state: CapsuleState,
        support: FrozenSupportPoint,
        normal_world: np.ndarray,
        desired_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        support_world, support_velocity = reconstruct_material_point(state, support.local_offset_m)
        tangent_error = support_tangent_error(
            anchor_world_m=support.anchor_world_m,
            support_world_m=support_world,
            normal_world=normal_world,
        )
        tangent_velocity = tangent_projection(support_velocity, normal_world)
        force = (
            self.profile.support_kp_n_per_m * tangent_error
            - self.profile.support_kd_ns_per_m * tangent_velocity
            - self.profile.support_normal_preload_n * normal_world
        )
        lever = support_world - state.position_world_m
        support_torque = np.cross(lever, force)
        actual_axis = capsule_axis_world(state)
        cross = np.cross(actual_axis, desired_axis)
        cross_length = float(np.linalg.norm(cross))
        angle = math.atan2(cross_length, float(np.clip(actual_axis @ desired_axis, -1.0, 1.0)))
        rotation_error = np.zeros(3) if cross_length <= 1.0e-12 else cross * (angle / cross_length)
        swing_rate = state.angular_velocity_world_rad_s - float(
            state.angular_velocity_world_rad_s @ actual_axis
        ) * actual_axis
        axis_torque = (
            self.profile.axis_kp_nm_per_rad * rotation_error
            - self.profile.axis_kd_nms_per_rad * swing_rate
        )
        torque = support_torque + axis_torque
        return force, torque, float(np.linalg.norm(tangent_error))

    def _limited_wrench(self, force, torque) -> tuple[WrenchCommand, bool, bool]:
        force = _clip_norm(force, self.profile.total_force_limit_n)
        torque = _clip_norm(torque, self.profile.total_torque_limit_nm)
        dt = 1.0 / self.profile.physics_hz
        force, force_limited = _slew(
            self._previous_force, force, self.profile.force_slew_limit_n_per_s * dt
        )
        torque, torque_limited = _slew(
            self._previous_torque, torque, self.profile.torque_slew_limit_nm_per_s * dt
        )
        self._previous_force = force.copy()
        self._previous_torque = torque.copy()
        return WrenchCommand(force, torque), force_limited, torque_limited

    def _telemetry(
        self,
        *,
        substep_index: int,
        result: ActionResult | None,
        desired_axis: np.ndarray,
        wrench: WrenchCommand,
        state: CapsuleState | None = None,
        support_drift: float = 0.0,
        force_slew_limited: bool = False,
        torque_slew_limited: bool = False,
        message: str = "",
    ) -> ActionTelemetry:
        current_axis = self._start_axis if state is None or not state.is_finite else capsule_axis_world(state)
        recent_step = max(0, self._latest_physics_substep)
        recent = self.contact_history.recent_contacts(
            current_substep=recent_step + 1,
            last_n_substeps=self.profile.contact_history_substeps,
        )
        return ActionTelemetry(
            lifecycle=self.lifecycle,
            action_id=self._action,
            request_id=self._request_id,
            substep_index=int(substep_index),
            result=result,
            constrained=self._constrained,
            direction_degenerate=self._direction_degenerate,
            start_axis_world=self._start_axis,
            end_axis_world=current_axis,
            desired_axis_world=desired_axis,
            surface_normal_world=self._surface_normal,
            support_anchor_world_m=np.zeros(3) if self._support is None else self._support.anchor_world_m,
            support_drift_m=support_drift,
            move_direction_world=self._move_direction,
            move_signed_displacement_m=0.0 if state is None else float(
                (state.position_world_m - self._move_start_position) @ self._move_direction
            ),
            any_contact=bool(recent),
            camera_contact=bool(self.contact_history.camera_constraints(
                current_substep=recent_step + 1,
                last_n_substeps=self.profile.contact_history_substeps,
            )),
            sidewall_contact=self.contact_history.had_sidewall_contact(
                current_substep=recent_step + 1,
                last_n_substeps=self.profile.contact_history_substeps,
            ),
            contact_cancel_delay_substeps=self._contact_cancel_delay,
            force_world_n=wrench.force_world_n,
            torque_world_nm=wrench.torque_world_nm,
            force_slew_limited=force_slew_limited,
            torque_slew_limited=torque_slew_limited,
            profile_sha256=self.profile_digest,
            message=message,
        )

    def _ready_hold_step(self, state: CapsuleState) -> ControllerStep:
        if self._hold_support is None or self._hold_axis is None:
            self._freeze_hold(state)
        force, torque, drift = self._support_wrench(
            state, self._hold_support, self._hold_normal, self._hold_axis
        )
        wrench, force_slew, torque_slew = self._limited_wrench(force, torque)
        telemetry = self._telemetry(
            substep_index=0,
            result=None,
            desired_axis=self._hold_axis,
            wrench=wrench,
            state=state,
            support_drift=drift,
            force_slew_limited=force_slew,
            torque_slew_limited=torque_slew,
        )
        return ControllerStep(wrench, telemetry)

    def step(self, state: CapsuleState, *, physics_substep: int) -> ControllerStep:
        self._latest_physics_substep = int(physics_substep)
        if self.lifecycle is Lifecycle.FAULTED:
            return self._enter_fault("controller remains faulted")
        if not state.is_finite:
            return self._enter_fault("non-finite capsule state")
        if self.lifecycle is Lifecycle.READY_HOLD:
            return self._ready_hold_step(state)
        if self._action is None or not 0 <= self._action_substep < self.profile.action_substeps:
            return self._enter_fault("invalid action state or substep")

        index = self._action_substep
        desired_axis = self._start_axis.copy()
        support_drift = 0.0
        force_slew = torque_slew = False

        if self._action.is_move and not self._rejected_move:
            if index == 60:
                hit = self.surface_query.query(state.position_world_m)
                self._surface_normal = hit.normal_world.copy()
                self._move_direction, self._direction_degenerate = move_direction(
                    capsule_axis_world(state),
                    hit.normal_world,
                    positive=self._action is ElevenActionId.MOVE_SIDE_POS,
                )
            if 60 <= index < 180 and not self._direction_degenerate:
                force = self.profile.move_force_n * self._move_direction
            else:
                force = np.zeros(3)
            # Explicit free/force/free MOVE phases never carry torque or support wrench.
            wrench, force_slew, torque_slew = self._limited_wrench(force, np.zeros(3))
        else:
            if self._action.is_view and not self._constrained:
                contacts = self.contact_history.camera_constraints(
                    current_substep=physics_substep + 1,
                    last_n_substeps=self.profile.contact_history_substeps,
                )
                pushes_inward = any(
                    float((self._target_axis - capsule_axis_world(state)) @ sample.normal_world)
                    < -1.0e-10
                    for sample in contacts
                )
                if pushes_inward:
                    self._constrained = True
                    self._target_axis = capsule_axis_world(state)
                    self._first_camera_contact_substep = min(item.physics_substep for item in contacts)
                    self._contact_cancel_delay = max(
                        1, int(physics_substep) - self._first_camera_contact_substep + 1
                    )
            if self._action.is_view and not self._constrained:
                progress = quintic_progress(index, self.profile.view_motion_substeps)
                desired_axis = swing_axis(self._start_axis, self._target_axis, progress)
            else:
                desired_axis = self._target_axis.copy()
            force, torque, support_drift = self._support_wrench(
                state, self._support, self._surface_normal, desired_axis
            )
            wrench, force_slew, torque_slew = self._limited_wrench(force, torque)

        self._action_substep += 1
        result = None
        if self._action_substep == self.profile.action_substeps:
            if self._action.is_move and self._rejected_move:
                result = ActionResult.REJECTED
            else:
                result = ActionResult.COMPLETED
            self.last_result = result
            self.lifecycle = Lifecycle.READY_HOLD

        telemetry = self._telemetry(
            substep_index=self._action_substep,
            result=result,
            desired_axis=desired_axis,
            wrench=wrench,
            state=state,
            support_drift=support_drift,
            force_slew_limited=force_slew,
            torque_slew_limited=torque_slew,
        )
        if result is not None:
            # The next READY update holds the true finite terminal state with no
            # zero-wrench gap, while preserving the slew history above.
            self._freeze_hold(state)
        return ControllerStep(wrench, telemetry)
