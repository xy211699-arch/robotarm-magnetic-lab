"""One-second quintic targets for the frozen fifteen-action contract."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .capsule_geometry import Spherocylinder
from .config import IdealSurfaceConfig
from .contact import CapsulePose
from .geometry import LocalFrame, normalized, orientation_from_axis_and_image_up, quintic, rotation_matrix
from .surface_mesh import SurfaceNavigationMesh
from .types import ControllerSnapshot, IdealSurfaceAction, START_TILT_ACTIONS


@dataclass(frozen=True)
class ActionTarget:
    theta_rad: float
    phi_rad: float
    tangent_delta_world: np.ndarray
    axial_roll_rad: float
    uses_tilt_anchor: bool


@dataclass(frozen=True)
class TrajectoryEvaluation:
    pose: CapsulePose
    quaternion_for_sim: np.ndarray
    surface_point_world: np.ndarray
    surface_normal_world: np.ndarray
    surface_triangle_id: int
    theta_rad: float
    phi_rad: float
    boundary_limited: bool


def target_for_action(
    action: IdealSurfaceAction, snapshot: ControllerSnapshot, cfg: IdealSurfaceConfig
) -> ActionTarget:
    action = IdealSurfaceAction(action)
    theta, phi = snapshot.theta_rad, snapshot.phi_rad
    delta = np.zeros(3, dtype=np.float64)
    roll = 0.0
    uses_tilt_anchor = False
    if action in START_TILT_ACTIONS:
        theta = cfg.tilt_step_rad
        phi = math.radians(45.0 * (int(action) - 1))
        uses_tilt_anchor = True
    elif action is IdealSurfaceAction.TILT_MORE:
        theta = min(snapshot.theta_rad + cfg.tilt_step_rad, math.pi / 2.0)
        uses_tilt_anchor = True
    elif action is IdealSurfaceAction.RISE:
        theta = max(snapshot.theta_rad - cfg.tilt_step_rad, 0.0)
        if theta <= cfg.upright_enter_rad:
            theta = 0.0
        uses_tilt_anchor = True
    elif action in (IdealSurfaceAction.PRECESS_POS, IdealSurfaceAction.PRECESS_NEG):
        sign = 1.0 if action is IdealSurfaceAction.PRECESS_POS else -1.0
        phi = snapshot.phi_rad + sign * cfg.precession_step_rad
    elif action in (IdealSurfaceAction.ROLL_POS, IdealSurfaceAction.ROLL_NEG):
        sign = 1.0 if action is IdealSurfaceAction.ROLL_POS else -1.0
        tangent = snapshot.axis_tangent_world
        if float(np.linalg.norm(tangent)) > 1.0e-12:
            delta = -sign * cfg.roll_arc_length_m * np.cross(snapshot.surface_normal_world, tangent)
            roll = sign * cfg.roll_arc_length_m / cfg.capsule_radius_m
    return ActionTarget(float(theta), float(phi), delta, float(roll), uses_tilt_anchor)


def _rotation_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the deterministic minimum rotation from ``first`` to ``second``."""
    first = normalized(first, name="rotation source")
    second = normalized(second, name="rotation target")
    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(first @ second, -1.0, 1.0))
    if sine <= 1.0e-12:
        if cosine > 0.0:
            return np.eye(3)
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(float(first @ reference)) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        return rotation_matrix(normalized(np.cross(first, reference)), math.pi)
    return rotation_matrix(cross / sine, math.atan2(sine, cosine))


def evaluate_trajectory(
    *,
    start: ControllerSnapshot,
    target: ActionTarget,
    progress: float,
    mesh: SurfaceNavigationMesh,
    capsule: Spherocylinder,
    recovery_radius_m: float,
    tilt_anchor_world: np.ndarray | None = None,
) -> TrajectoryEvaluation:
    blend = quintic(progress)
    theta = start.theta_rad + blend * (target.theta_rad - start.theta_rad)
    phi = start.phi_rad + blend * (target.phi_rad - start.phi_rad)
    # Reconstruct the persistent zero-azimuth ray from the stored axis and phi.
    # Reusing the current camera image-up as a fresh zero on every action causes
    # the compass frame to rotate and corrupts TILT_MORE/RISE directions.
    axis_tangent = start.axis_tangent_world
    if float(np.linalg.norm(axis_tangent)) > 1.0e-12 and start.theta_rad > 1.0e-8:
        projected_reference = rotation_matrix(
            start.surface_normal_world, -start.phi_rad
        ) @ axis_tangent
    else:
        frame_reference = start.image_up_world.copy()
        projected_reference = frame_reference - float(
            frame_reference @ start.surface_normal_world
        ) * start.surface_normal_world
    if float(np.linalg.norm(projected_reference)) <= 1.0e-12:
        projected_reference = start.axis_tangent_world
    frame = LocalFrame(start.surface_point_world, start.surface_normal_world, projected_reference)
    axis = normalized(
        math.cos(theta) * frame.normal_world + math.sin(theta) * frame.direction(phi),
        name="trajectory capsule axis",
    )
    hit = mesh.advance(
        start.surface_triangle_id,
        start.surface_point_world,
        blend * target.tangent_delta_world,
        recovery_radius_m,
    )
    # Parallel-transport the selected tilt ray onto the new triangle normal.
    if float(np.linalg.norm(target.tangent_delta_world)) > 0.0:
        tangent = axis - float(axis @ hit.normal_world) * hit.normal_world
        if float(np.linalg.norm(tangent)) > 1.0e-12:
            axis = math.cos(theta) * hit.normal_world + math.sin(theta) * normalized(tangent)
            axis = normalized(axis)
    rigid_rotation = _rotation_between(start.axis_world, axis)
    image_up = rigid_rotation @ start.image_up_world
    if abs(target.axial_roll_rad) > 0.0:
        image_up = rotation_matrix(axis, blend * target.axial_roll_rad) @ image_up
    if target.uses_tilt_anchor:
        if tilt_anchor_world is None:
            raise ValueError("tilt action requires a fixed active anchor")
        anchor = np.asarray(tilt_anchor_world, dtype=np.float64).reshape(3)
        anchor_hit = mesh.advance(
            start.surface_triangle_id, anchor, np.zeros(3), recovery_radius_m
        )
        hit = anchor_hit
        # Keep the selected surface anchor fixed while the spherical end-cap
        # contact migrates: the end-cap centre follows ``h * axis`` and its
        # radius remains along the local wall normal.  Rotating the old
        # centre-to-contact vector as if the spherical point were material
        # fixed would drive the neighbouring sphere surface into the wall.
        center = (
            anchor
            + capsule.cylinder_half_length_m * axis
            + capsule.radius_m * hit.normal_world
        )
    else:
        support = capsule.support_distance(axis, hit.normal_world)
        desired_center = hit.point_world + support * hit.normal_world
        start_support = capsule.support_distance(
            start.axis_world, start.surface_normal_world
        )
        start_desired_center = (
            start.surface_point_world
            + start_support * start.surface_normal_world
        )
        # Preserve the safe correction inherited from a previous anchored tilt.
        # This also makes progress=0 exactly reproduce the submitted pose.
        center = start.position_world + (desired_center - start_desired_center)
    quaternion = orientation_from_axis_and_image_up(axis, image_up)
    return TrajectoryEvaluation(
        pose=CapsulePose(center, axis, image_up),
        quaternion_for_sim=quaternion,
        surface_point_world=hit.point_world.copy(),
        surface_normal_world=hit.normal_world.copy(),
        surface_triangle_id=hit.triangle_id,
        theta_rad=float(theta),
        phi_rad=float(phi),
        boundary_limited=bool(hit.boundary_limited),
    )
