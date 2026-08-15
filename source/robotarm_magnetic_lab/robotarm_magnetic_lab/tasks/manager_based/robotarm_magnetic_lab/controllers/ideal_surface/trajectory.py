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
    if action in START_TILT_ACTIONS:
        theta = cfg.tilt_step_rad
        phi = math.radians(45.0 * (int(action) - 1))
    elif action is IdealSurfaceAction.TILT_MORE:
        theta = min(snapshot.theta_rad + cfg.tilt_step_rad, math.pi / 2.0)
    elif action is IdealSurfaceAction.RISE:
        theta = max(snapshot.theta_rad - cfg.tilt_step_rad, 0.0)
        if theta <= cfg.upright_enter_rad:
            theta = 0.0
    elif action in (IdealSurfaceAction.PRECESS_POS, IdealSurfaceAction.PRECESS_NEG):
        sign = 1.0 if action is IdealSurfaceAction.PRECESS_POS else -1.0
        phi = snapshot.phi_rad + sign * cfg.precession_step_rad
    elif action in (IdealSurfaceAction.ROLL_POS, IdealSurfaceAction.ROLL_NEG):
        sign = 1.0 if action is IdealSurfaceAction.ROLL_POS else -1.0
        tangent = snapshot.axis_tangent_world
        if float(np.linalg.norm(tangent)) > 1.0e-12:
            delta = -sign * cfg.roll_arc_length_m * np.cross(snapshot.surface_normal_world, tangent)
            roll = sign * cfg.roll_arc_length_m / cfg.capsule_radius_m
    return ActionTarget(float(theta), float(phi), delta, float(roll))


def evaluate_trajectory(
    *,
    start: ControllerSnapshot,
    target: ActionTarget,
    progress: float,
    mesh: SurfaceNavigationMesh,
    capsule: Spherocylinder,
    recovery_radius_m: float,
) -> TrajectoryEvaluation:
    blend = quintic(progress)
    theta = start.theta_rad + blend * (target.theta_rad - start.theta_rad)
    phi = start.phi_rad + blend * (target.phi_rad - start.phi_rad)
    frame_reference = start.image_up_world.copy()
    projected_reference = frame_reference - float(frame_reference @ start.surface_normal_world) * start.surface_normal_world
    if float(np.linalg.norm(projected_reference)) <= 1.0e-12:
        # At exactly 90 degrees the image-up vector may align with the wall normal.
        # The current tilt ray is the deterministic parallel-transport continuation.
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
    image_up = start.image_up_world.copy()
    if abs(target.axial_roll_rad) > 0.0:
        image_up = rotation_matrix(axis, blend * target.axial_roll_rad) @ image_up
    support = capsule.support_distance(axis, hit.normal_world)
    center = hit.point_world + support * hit.normal_world
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
