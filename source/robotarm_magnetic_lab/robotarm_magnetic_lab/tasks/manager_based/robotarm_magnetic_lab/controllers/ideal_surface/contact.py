"""Deterministic geometric capsule/surface contact classification."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .capsule_geometry import Spherocylinder
from .config import IdealSurfaceConfig
from .geometry import normalized
from .surface_mesh import SurfaceNavigationMesh


@dataclass(frozen=True)
class CapsulePose:
    center_world: np.ndarray
    axis_world: np.ndarray
    image_up_world: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world, dtype=np.float64).reshape(3)
        if not np.isfinite(center).all():
            raise ValueError("capsule center must be finite")
        axis = normalized(self.axis_world, name="capsule axis")
        image_up = np.asarray(self.image_up_world, dtype=np.float64).reshape(3)
        image_up -= float(image_up @ axis) * axis
        image_up = normalized(image_up, name="capsule image up")
        object.__setattr__(self, "center_world", center.copy())
        object.__setattr__(self, "axis_world", axis)
        object.__setattr__(self, "image_up_world", image_up)

    def translated(self, delta_world: np.ndarray) -> "CapsulePose":
        return replace(self, center_world=self.center_world + np.asarray(delta_world, dtype=np.float64))


@dataclass(frozen=True)
class ContactAssessment:
    support_valid: bool
    side_contact: bool
    contact_limited: bool
    boundary_limited: bool
    hard_failure: bool
    maximum_penetration_m: float
    support_point_world: np.ndarray
    support_normal_world: np.ndarray
    active_triangle: int
    barrel_clearances_m: np.ndarray
    barrel_axial_parameters: np.ndarray


@dataclass(frozen=True)
class ActiveAnchor:
    point_world: np.ndarray
    triangle_id: int
    contact_index: int


@dataclass(frozen=True)
class SweptTargetAssessment:
    pose: CapsulePose
    assessment: ContactAssessment


@dataclass(frozen=True)
class ContactClassifierResult:
    side_contact: bool
    stable_time_s: float


def select_active_anchor(
    contacts_world: np.ndarray,
    center_world: np.ndarray,
    tilt_direction_world: np.ndarray,
    triangle_ids: np.ndarray,
) -> ActiveAnchor:
    """Choose the extreme contact opposite the intended tilt ray, deterministically."""
    contacts = np.asarray(contacts_world, dtype=np.float64)
    triangles = np.asarray(triangle_ids, dtype=np.int64).reshape(-1)
    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    direction = normalized(tilt_direction_world, name="tilt direction")
    if contacts.ndim != 2 or contacts.shape[1] != 3 or len(contacts) != len(triangles) or not len(contacts):
        raise ValueError("contacts and triangle_ids must be non-empty and aligned")
    projections = (contacts - center) @ direction
    index = min(
        range(len(contacts)),
        key=lambda value: (
            float(projections[value]),
            int(triangles[value]),
            int(value),
        ),
    )
    return ActiveAnchor(contacts[index].copy(), int(triangles[index]), int(index))


def _project_sample(
    mesh: SurfaceNavigationMesh,
    active_triangle: int,
    sample_world: np.ndarray,
    recovery_radius_m: float,
):
    candidates = mesh.local_candidate_triangles(active_triangle, sample_world, recovery_radius_m)
    ranked = mesh.rank_projected_candidates(sample_world, candidates)
    if not ranked:
        raise RuntimeError("no same-component contact projection")
    _, triangle_id, _, point, barycentric = ranked[0]
    normal = mesh.normals[triangle_id]
    clearance = float((sample_world - point) @ normal)
    boundary = mesh._is_boundary_projection(triangle_id, barycentric)
    return clearance, np.asarray(point), np.asarray(normal), int(triangle_id), bool(boundary)


def assess_pose(
    mesh: SurfaceNavigationMesh,
    capsule: Spherocylinder,
    pose: CapsulePose,
    active_triangle: int,
    cfg: IdealSurfaceConfig,
) -> ContactAssessment:
    """Assess deterministic longitudinal barrel samples against the local component."""
    active_triangle = int(active_triangle)
    base_normal = normalized(mesh.normals[active_triangle], name="active surface normal")
    # Five samples permit the required separated-barrel test on curved tissue;
    # checking only both cylinder ends incorrectly rejects valid side contact.
    axial_parameters = np.linspace(-0.5, 0.5, 5, dtype=np.float64)
    axial_offsets = axial_parameters * capsule.cylinder_length_m
    samples = (
        pose.center_world[None, :]
        + axial_offsets[:, None] * pose.axis_world[None, :]
        - capsule.radius_m * base_normal[None, :]
    )
    recovery_radius = cfg.recovery_query_radius_scale * capsule.radius_m
    projected = [
        _project_sample(mesh, active_triangle, sample, recovery_radius) for sample in samples
    ]
    clearances = np.asarray([item[0] for item in projected], dtype=np.float64)
    support_index = int(np.argmin(np.abs(clearances)))
    support_point = projected[support_index][1]
    support_normal = normalized(projected[support_index][2], name="support normal")
    support_triangle = int(projected[support_index][3])
    clearance_limit = cfg.contact_clearance_radius_fraction * capsule.radius_m
    planned_limit = cfg.planned_penetration_radius_fraction * capsule.radius_m
    hard_limit = cfg.hard_penetration_radius_fraction * capsule.radius_m
    maximum_penetration = max(0.0, -float(np.min(clearances)))
    near = np.flatnonzero(np.abs(clearances) <= clearance_limit)
    side_contact = bool(
        len(near) >= 2
        and float(np.ptp(axial_parameters[near])) >= cfg.side_contact_separation_fraction
    )
    return ContactAssessment(
        support_valid=bool(float(np.min(np.abs(clearances))) <= clearance_limit),
        side_contact=side_contact,
        contact_limited=bool(maximum_penetration > planned_limit),
        boundary_limited=any(item[4] for item in projected),
        hard_failure=bool(maximum_penetration > hard_limit),
        maximum_penetration_m=maximum_penetration,
        support_point_world=support_point.copy(),
        support_normal_world=support_normal,
        active_triangle=support_triangle,
        barrel_clearances_m=clearances.copy(),
        barrel_axial_parameters=axial_parameters.copy(),
    )


def assess_swept_target(
    *,
    mesh: SurfaceNavigationMesh,
    capsule: Spherocylinder,
    current: CapsulePose,
    proposed: CapsulePose,
    active_triangle: int,
    cfg: IdealSurfaceConfig,
) -> SweptTargetAssessment:
    """Clip a planned penetration to the configured allowance; preserve hard faults."""
    del current  # Reserved for denser swept sampling in the live integration layer.
    original = assess_pose(mesh, capsule, proposed, active_triangle, cfg)
    if original.hard_failure or not original.contact_limited:
        return SweptTargetAssessment(proposed, original)
    planned = cfg.planned_penetration_radius_fraction * capsule.radius_m
    correction = max(0.0, original.maximum_penetration_m - planned)
    clipped_pose = proposed.translated(correction * original.support_normal_world)
    clipped = assess_pose(mesh, capsule, clipped_pose, original.active_triangle, cfg)
    clipped = replace(
        clipped,
        contact_limited=True,
        maximum_penetration_m=original.maximum_penetration_m,
    )
    return SweptTargetAssessment(clipped_pose, clipped)


class ContactClassifier:
    """Temporal side-contact classifier with the contract's 0.1 s stability gate."""

    def __init__(self, cfg: IdealSurfaceConfig, capsule: Spherocylinder) -> None:
        self.cfg = cfg
        self.capsule = capsule
        self._valid_time_s = 0.0
        self._invalid_time_s = 0.0
        self._side_contact = False

    def reset(self) -> None:
        self._valid_time_s = 0.0
        self._invalid_time_s = 0.0
        self._side_contact = False

    def observe(
        self,
        pose: CapsulePose,
        barrel_clearances: np.ndarray,
        barrel_axial_parameters: np.ndarray,
        dt: float,
    ) -> ContactClassifierResult:
        del pose
        clearances = np.asarray(barrel_clearances, dtype=np.float64).reshape(-1)
        parameters = np.asarray(barrel_axial_parameters, dtype=np.float64).reshape(-1)
        if len(clearances) != len(parameters) or len(clearances) < 2:
            raise ValueError("at least two aligned barrel samples are required")
        threshold = self.cfg.contact_clearance_radius_fraction * self.capsule.radius_m
        valid = np.flatnonzero(np.abs(clearances) <= threshold)
        separated = False
        if len(valid) >= 2:
            span = float(np.max(parameters[valid]) - np.min(parameters[valid]))
            separated = span >= self.cfg.side_contact_separation_fraction
        if separated:
            self._valid_time_s += float(dt)
            self._invalid_time_s = 0.0
            if self._valid_time_s + 1.0e-12 >= self.cfg.logical_stability_s:
                self._side_contact = True
        else:
            self._valid_time_s = 0.0
            self._invalid_time_s += float(dt)
            if self._invalid_time_s + 1.0e-12 >= self.cfg.logical_stability_s:
                self._side_contact = False
        return ContactClassifierResult(
            side_contact=bool(self._side_contact),
            stable_time_s=float(self._valid_time_s if separated else self._invalid_time_s),
        )
