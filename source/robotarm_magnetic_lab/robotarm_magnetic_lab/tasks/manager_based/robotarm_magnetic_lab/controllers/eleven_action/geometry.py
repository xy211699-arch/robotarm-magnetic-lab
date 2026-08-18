"""Dependency-light geometry for the TASK-005 dynamic action controller."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import CapsuleState, ElevenActionId


_EPS = 1.0e-12


def normalized(value, *, name: str = "vector") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or length <= _EPS:
        raise ValueError(f"{name} must be finite and non-zero")
    return result / length


def quaternion_wxyz_to_matrix(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    length = float(np.linalg.norm(q))
    if not np.isfinite(q).all() or length <= _EPS:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = q / length
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def capsule_axis_world(state: CapsuleState) -> np.ndarray:
    """Directed non-camera-to-camera axis: the camera optical local -Z axis."""
    return quaternion_wxyz_to_matrix(state.quaternion_wxyz) @ np.asarray([0.0, 0.0, -1.0])


def camera_frame(state: CapsuleState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frozen optical, image-up and image-right world directions."""
    rotation = quaternion_wxyz_to_matrix(state.quaternion_wxyz)
    optical = normalized(rotation @ np.asarray([0.0, 0.0, -1.0]), name="optical axis")
    up = rotation @ np.asarray([0.0, 1.0, 0.0])
    up = normalized(up - float(up @ optical) * optical, name="image up")
    right = rotation @ np.asarray([1.0, 0.0, 0.0])
    right = normalized(right - float(right @ optical) * optical - float(right @ up) * up, name="image right")
    return optical, up, right


_GRID_COMPONENTS = {
    ElevenActionId.VIEW_UP: (1.0, 0.0),
    ElevenActionId.VIEW_UP_RIGHT: (1.0, 1.0),
    ElevenActionId.VIEW_RIGHT: (0.0, 1.0),
    ElevenActionId.VIEW_DOWN_RIGHT: (-1.0, 1.0),
    ElevenActionId.VIEW_DOWN: (-1.0, 0.0),
    ElevenActionId.VIEW_DOWN_LEFT: (-1.0, -1.0),
    ElevenActionId.VIEW_LEFT: (0.0, -1.0),
    ElevenActionId.VIEW_UP_LEFT: (1.0, -1.0),
}


def grid_direction_world(action_id: ElevenActionId, image_up_world, image_right_world) -> np.ndarray:
    action = ElevenActionId(int(action_id))
    if action not in _GRID_COMPONENTS:
        raise ValueError(f"action {action.name} has no grid direction")
    up_scale, right_scale = _GRID_COMPONENTS[action]
    return normalized(
        up_scale * normalized(image_up_world, name="image up")
        + right_scale * normalized(image_right_world, name="image right"),
        name="grid direction",
    )


def view_target_axis(start_axis_world, grid_direction, *, angle_rad: float) -> np.ndarray:
    start = normalized(start_axis_world, name="start optical axis")
    direction = np.asarray(grid_direction, dtype=np.float64).reshape(3)
    direction -= float(direction @ start) * start
    direction = normalized(direction, name="projected grid direction")
    return normalized(math.cos(float(angle_rad)) * start + math.sin(float(angle_rad)) * direction)


@dataclass(frozen=True)
class FrozenSupportPoint:
    local_offset_m: np.ndarray
    anchor_world_m: np.ndarray

    def __post_init__(self) -> None:
        for name in ("local_offset_m", "anchor_world_m"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(3).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def freeze_support_material_point(
    state: CapsuleState,
    normal_world,
    *,
    radius_m: float,
    half_length_m: float,
) -> FrozenSupportPoint:
    """Freeze the spherocylinder material point farthest toward the wall (-normal)."""
    if radius_m <= 0.0 or half_length_m <= 0.0:
        raise ValueError("spherocylinder dimensions must be positive")
    rotation = quaternion_wxyz_to_matrix(state.quaternion_wxyz)
    axis = capsule_axis_world(state)
    normal = normalized(normal_world, name="surface inward normal")
    axial_sign = float(np.sign(float(axis @ normal)))
    world_offset = -float(radius_m) * normal - float(half_length_m) * axial_sign * axis
    local_offset = rotation.T @ world_offset
    anchor = state.position_world_m + world_offset
    return FrozenSupportPoint(local_offset, anchor)


def reconstruct_material_point(state: CapsuleState, local_offset_m) -> tuple[np.ndarray, np.ndarray]:
    rotation = quaternion_wxyz_to_matrix(state.quaternion_wxyz)
    offset_world = rotation @ np.asarray(local_offset_m, dtype=np.float64).reshape(3)
    point = state.position_world_m + offset_world
    velocity = state.linear_velocity_world_m_s + np.cross(state.angular_velocity_world_rad_s, offset_world)
    return point, velocity


def tangent_projection(value, normal_world) -> np.ndarray:
    normal = normalized(normal_world, name="surface normal")
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    return vector - float(vector @ normal) * normal


def support_tangent_error(*, anchor_world_m, support_world_m, normal_world) -> np.ndarray:
    return tangent_projection(
        np.asarray(anchor_world_m, dtype=np.float64).reshape(3)
        - np.asarray(support_world_m, dtype=np.float64).reshape(3),
        normal_world,
    )


def classify_contact_region(axial_coordinate_m: float, cylinder_half_length_m: float):
    # Local import avoids a circular dependency while keeping one public classifier.
    from .contact_history import ContactRegion

    sigma = float(axial_coordinate_m)
    half_length = float(cylinder_half_length_m)
    if not math.isfinite(sigma) or half_length <= 0.0:
        raise ValueError("contact coordinate and half length must be valid")
    if sigma > half_length:
        return ContactRegion.CAMERA_HEMISPHERE
    if sigma < -half_length:
        return ContactRegion.NONCAMERA_HEMISPHERE
    return ContactRegion.SIDEWALL

