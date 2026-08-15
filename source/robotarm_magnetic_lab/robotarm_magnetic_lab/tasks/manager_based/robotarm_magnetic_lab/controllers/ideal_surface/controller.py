"""Minimal READY/EXECUTING/TERMINAL_FAULT ideal-surface executor."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import numpy as np

from .action_mask import compute_action_mask
from .capsule_geometry import Spherocylinder
from .config import IdealSurfaceConfig
from .contact import CapsulePose, ContactAssessment, ContactClassifier, assess_pose
from .geometry import quaternion_wxyz_to_matrix
from .surface_mesh import SurfaceNavigationMesh
from .trajectory import TrajectoryEvaluation, evaluate_trajectory, target_for_action
from .types import (
    ControllerSnapshot,
    ControllerState,
    IdealActionResult,
    IdealActionStatus,
    IdealSurfaceAction,
    SurfaceFlags,
)


PoseAssessor = Callable[[CapsulePose, int, IdealSurfaceConfig], ContactAssessment]


@dataclass(frozen=True)
class ControllerOutput:
    position_world: np.ndarray
    quaternion_for_sim: np.ndarray
    linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray
    flags: SurfaceFlags
    result: IdealActionResult | None = None

    @property
    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.position_world, self.quaternion_for_sim


def _angular_velocity(previous: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
    relative = quaternion_wxyz_to_matrix(current) @ quaternion_wxyz_to_matrix(previous).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1.0e-10:
        return np.zeros(3)
    axis = np.asarray(
        [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]
    )
    sine = math.sin(angle)
    if abs(sine) <= 1.0e-10:
        return np.zeros(3)
    axis /= 2.0 * sine
    return axis * angle / dt


class IdealSurfaceController:
    def __init__(
        self,
        mesh: SurfaceNavigationMesh,
        *,
        cfg: IdealSurfaceConfig | None = None,
        capsule: Spherocylinder | None = None,
        pose_assessor: PoseAssessor | None = None,
    ) -> None:
        self.mesh = mesh
        self.cfg = cfg or IdealSurfaceConfig()
        self.capsule = capsule or Spherocylinder(
            self.cfg.capsule_radius_m, self.cfg.capsule_cylinder_half_length_m
        )
        self._pose_assessor = pose_assessor
        self.contact_classifier = ContactClassifier(self.cfg, self.capsule)
        self.state = ControllerState.READY
        self.snapshot: ControllerSnapshot | None = None
        self.last_result: IdealActionResult | None = None
        self._seen_request_ids: set[int] = set()
        self._active_action = IdealSurfaceAction.HOLD
        self._request_id = -1
        self._start: ControllerSnapshot | None = None
        self._target = None
        self._tilt_anchor_world: np.ndarray | None = None
        self._tilt_anchor_normal_world: np.ndarray | None = None
        self._tilt_anchor_triangle_id: int | None = None
        self._elapsed_s = 0.0
        self._last_safe: TrajectoryEvaluation | None = None
        self._motion_limited = False
        self._contact_limited = False
        self._boundary_limited = False
        self._maximum_penetration_m = 0.0
        self._no_effect = False
        self._static_pose_action = False
        self._held_assessment: ContactAssessment | None = None
        self._logical_upright = True
        self._upright_candidate: bool | None = None
        self._upright_candidate_time_s = 0.0

    @property
    def ready(self) -> bool:
        return self.state is ControllerState.READY and self.last_result is None and self.snapshot is not None

    def reset(self, snapshot: ControllerSnapshot) -> None:
        self.snapshot = snapshot
        self.state = ControllerState.READY
        self.last_result = None
        self._seen_request_ids.clear()
        self._elapsed_s = 0.0
        self._last_safe = None
        self._tilt_anchor_world = None
        self._tilt_anchor_normal_world = None
        self._tilt_anchor_triangle_id = None
        self._motion_limited = False
        self._contact_limited = False
        self._boundary_limited = False
        self._maximum_penetration_m = 0.0
        self._no_effect = False
        self._static_pose_action = False
        self._held_assessment = None
        self._logical_upright = bool(snapshot.flags.upright)
        self._upright_candidate = None
        self._upright_candidate_time_s = 0.0
        self.contact_classifier.reset()

    def action_mask(self) -> np.ndarray:
        if self.snapshot is None:
            return np.zeros(len(IdealSurfaceAction), dtype=np.bool_)
        return compute_action_mask(self.snapshot.flags, self.cfg)

    def submit(self, action_id: int, snapshot: ControllerSnapshot, request_id: int) -> bool:
        if not self.ready or int(request_id) in self._seen_request_ids:
            return False
        try:
            action = IdealSurfaceAction(int(action_id))
        except ValueError:
            return False
        self.snapshot = snapshot
        self._start = snapshot
        self._active_action = action
        self._request_id = int(request_id)
        self._seen_request_ids.add(int(request_id))
        self._elapsed_s = 0.0
        self._motion_limited = False
        self._contact_limited = False
        self._boundary_limited = False
        self._maximum_penetration_m = 0.0
        self._no_effect = not bool(self.action_mask()[int(action)])
        effective = IdealSurfaceAction.HOLD if self._no_effect else action
        self._target = target_for_action(effective, snapshot, self.cfg)
        self._static_pose_action = bool(
            abs(self._target.theta_rad - snapshot.theta_rad) <= 1.0e-15
            and abs(self._target.phi_rad - snapshot.phi_rad) <= 1.0e-15
            and float(np.linalg.norm(self._target.tangent_delta_world)) <= 1.0e-15
            and abs(self._target.axial_roll_rad) <= 1.0e-15
        )
        self._held_assessment = None
        self._tilt_anchor_world = None
        self._tilt_anchor_normal_world = None
        self._tilt_anchor_triangle_id = None
        if self._target.uses_tilt_anchor:
            # The submitted snapshot is the last accepted active contact frame.
            # Re-projecting the same pose at an action boundary can select the
            # neighbouring facet of a mesh edge and create a discontinuous
            # normal (and false penetration) before progress has advanced.
            self._tilt_anchor_world = snapshot.surface_point_world.copy()
            self._tilt_anchor_normal_world = snapshot.surface_normal_world.copy()
            self._tilt_anchor_triangle_id = int(snapshot.surface_triangle_id)
        self._last_safe = self._evaluate(0.0)
        self.state = ControllerState.EXECUTING
        return True

    def acknowledge_result(self) -> None:
        if self.last_result is not None and self.state is not ControllerState.TERMINAL_FAULT:
            self.last_result = None
            self.state = ControllerState.READY

    def _evaluate(self, progress: float) -> TrajectoryEvaluation:
        assert self._start is not None and self._target is not None
        return evaluate_trajectory(
            start=self._start,
            target=self._target,
            progress=progress,
            mesh=self.mesh,
            capsule=self.capsule,
            recovery_radius_m=self.cfg.recovery_query_radius_scale * self.capsule.radius_m,
            tilt_anchor_world=self._tilt_anchor_world,
            tilt_anchor_normal_world=self._tilt_anchor_normal_world,
            tilt_anchor_triangle_id=self._tilt_anchor_triangle_id,
        )

    def _assess(self, evaluation: TrajectoryEvaluation) -> ContactAssessment:
        if self._pose_assessor is not None:
            return self._pose_assessor(evaluation.pose, evaluation.surface_triangle_id, self.cfg)
        return assess_pose(
            self.mesh, self.capsule, evaluation.pose, evaluation.surface_triangle_id, self.cfg
        )

    def _update_logical_upright(self, theta_rad: float, dt: float) -> bool:
        desired = self._logical_upright
        if self._logical_upright and theta_rad > self.cfg.upright_exit_rad:
            desired = False
        elif not self._logical_upright and theta_rad <= self.cfg.upright_enter_rad:
            desired = True
        if desired == self._logical_upright:
            self._upright_candidate = None
            self._upright_candidate_time_s = 0.0
            return self._logical_upright
        if self._upright_candidate != desired:
            self._upright_candidate = desired
            self._upright_candidate_time_s = 0.0
        self._upright_candidate_time_s += float(dt)
        if self._upright_candidate_time_s + 1.0e-12 >= self.cfg.logical_stability_s:
            self._logical_upright = desired
            self._upright_candidate = None
            self._upright_candidate_time_s = 0.0
        return self._logical_upright

    def _flags(
        self, evaluation: TrajectoryEvaluation, assessment: ContactAssessment, dt: float
    ) -> SurfaceFlags:
        upright = self._update_logical_upright(evaluation.theta_rad, dt)
        side_state = self.contact_classifier.observe(
            evaluation.pose,
            assessment.barrel_clearances_m,
            assessment.barrel_axial_parameters,
            float(dt),
        )
        return SurfaceFlags(
            upright=bool(upright),
            side_contact=bool(side_state.side_contact),
            contact_limited=bool(self._contact_limited or assessment.contact_limited),
            boundary_limited=bool(
                self._boundary_limited
                or evaluation.boundary_limited
                or assessment.boundary_limited
            ),
            no_effect=bool(self._no_effect),
        )

    def _snapshot_from(self, evaluation: TrajectoryEvaluation, flags: SurfaceFlags) -> ControllerSnapshot:
        assert self._start is not None
        return ControllerSnapshot(
            sim_time_s=self._start.sim_time_s + self._elapsed_s,
            position_world=evaluation.pose.center_world,
            quaternion_for_sim=evaluation.quaternion_for_sim,
            axis_world=evaluation.pose.axis_world,
            image_up_world=evaluation.pose.image_up_world,
            surface_point_world=evaluation.surface_point_world,
            surface_normal_world=evaluation.surface_normal_world,
            surface_triangle_id=evaluation.surface_triangle_id,
            theta_rad=evaluation.theta_rad,
            phi_rad=evaluation.phi_rad,
            flags=flags,
        )

    def _make_result(self, status: IdealActionStatus, detail: str | None = None) -> IdealActionResult:
        assert self.snapshot is not None and self._start is not None
        end_time = (
            self._start.sim_time_s + self.cfg.action_duration_s
            if status is IdealActionStatus.DONE
            else self._start.sim_time_s + self._elapsed_s
        )
        return IdealActionResult(
            request_id=self._request_id,
            action=self._active_action,
            status=status,
            started_at_s=self._start.sim_time_s,
            ended_at_s=end_time,
            contact_limited=self.snapshot.flags.contact_limited,
            boundary_limited=self.snapshot.flags.boundary_limited,
            no_effect=self.snapshot.flags.no_effect,
            hard_failure_detail=detail,
            final_position_world=self.snapshot.position_world,
            final_quaternion_for_sim=self.snapshot.quaternion_for_sim,
            final_axis_world=self.snapshot.axis_world,
            final_tilt_rad=self.snapshot.theta_rad,
            final_azimuth_rad=self.snapshot.phi_rad,
            maximum_penetration_m=self._maximum_penetration_m,
        )

    def _held_output(self, result: IdealActionResult | None = None) -> ControllerOutput:
        assert self.snapshot is not None
        return ControllerOutput(
            self.snapshot.position_world.copy(),
            self.snapshot.quaternion_for_sim.copy(),
            np.zeros(3),
            np.zeros(3),
            self.snapshot.flags,
            result,
        )

    def step(self, dt: float) -> ControllerOutput:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.snapshot is None:
            raise RuntimeError("controller must be reset before stepping")
        if self.state is not ControllerState.EXECUTING:
            return self._held_output()
        previous = self.snapshot
        self._elapsed_s = min(self._elapsed_s + float(dt), self.cfg.action_duration_s)
        progress = self._elapsed_s / self.cfg.action_duration_s
        proposed = self._last_safe if self._motion_limited else self._evaluate(progress)
        assert proposed is not None
        assessment = (
            self._held_assessment
            if self._held_assessment is not None
            else self._assess(proposed)
        )
        if assessment.hard_failure:
            # A rejected future target is not actual penetration.  Stop at the
            # last accepted sub-target.  HARD_FAILURE is reserved for a current
            # pose that is already beyond the hard threshold.
            safe = self._last_safe
            assert safe is not None
            safe_assessment = self._assess(safe)
            if safe_assessment.hard_failure:
                flags = replace(previous.flags, contact_limited=True)
                self.snapshot = replace(
                    previous,
                    sim_time_s=(self._start.sim_time_s + self._elapsed_s),
                    flags=flags,
                )
                self.state = ControllerState.TERMINAL_FAULT
                self.last_result = self._make_result(
                    IdealActionStatus.HARD_FAILURE,
                    "current capsule penetration exceeded hard threshold",
                )
                return self._held_output(self.last_result)
            self._motion_limited = True
            self._contact_limited = True
            proposed = safe
            assessment = safe_assessment
        self._maximum_penetration_m = max(
            self._maximum_penetration_m, assessment.maximum_penetration_m
        )
        limited = bool(assessment.contact_limited or assessment.boundary_limited or proposed.boundary_limited)
        if limited and not self._motion_limited:
            self._motion_limited = True
            self._contact_limited = bool(assessment.contact_limited)
            self._boundary_limited = bool(
                assessment.boundary_limited or proposed.boundary_limited
            )
            proposed = self._last_safe
            assert proposed is not None
            assessment = self._assess(proposed)
        elif not limited:
            self._last_safe = proposed
        if self._motion_limited or self._static_pose_action:
            self._held_assessment = assessment
        flags = self._flags(proposed, assessment, float(dt))
        self.snapshot = self._snapshot_from(proposed, flags)
        linear = (self.snapshot.position_world - previous.position_world) / float(dt)
        angular = _angular_velocity(previous.quaternion_for_sim, self.snapshot.quaternion_for_sim, float(dt))
        result = None
        if self._elapsed_s + 1.0e-12 >= self.cfg.action_duration_s:
            self._elapsed_s = self.cfg.action_duration_s
            self.snapshot = replace(self.snapshot, sim_time_s=self._start.sim_time_s + self.cfg.action_duration_s)
            self.last_result = self._make_result(IdealActionStatus.DONE)
            result = self.last_result
            self.state = ControllerState.READY
        return ControllerOutput(
            self.snapshot.position_world.copy(), self.snapshot.quaternion_for_sim.copy(),
            linear, angular, self.snapshot.flags, result
        )
