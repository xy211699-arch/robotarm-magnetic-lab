"""Frozen-frame geometry for VIEW and MOVE actions."""

from __future__ import annotations

import math
import numpy as np

from .types import ActionId


def normalize(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return value / norm


def camera_image_axes_from_ros_rotation(rotation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return optical, image-up, and image-right axes from a ROS camera pose.

    Isaac Lab's ROS camera frame uses +Z forward, +Y image-down, and +X
    image-right. Keeping this conversion in one helper prevents the controller
    from silently substituting capsule-body axes for image axes.
    """
    value = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(value).all():
        raise ValueError("camera rotation must be finite")
    optical = normalize(value[:, 2])
    up = normalize(-value[:, 1])
    right = normalize(value[:, 0])
    if abs(float(np.dot(optical, up))) > 1.0e-6 or abs(float(np.dot(optical, right))) > 1.0e-6:
        raise ValueError("camera rotation does not define orthogonal image axes")
    return optical, up, right


_VIEW_IMAGE_DIRECTIONS = {
    ActionId.VIEW_UP: (1.0, 0.0),
    ActionId.VIEW_UP_RIGHT: (1.0, 1.0),
    ActionId.VIEW_RIGHT: (0.0, 1.0),
    ActionId.VIEW_DOWN_RIGHT: (-1.0, 1.0),
    ActionId.VIEW_DOWN: (-1.0, 0.0),
    ActionId.VIEW_DOWN_LEFT: (-1.0, -1.0),
    ActionId.VIEW_LEFT: (0.0, -1.0),
    ActionId.VIEW_UP_LEFT: (1.0, -1.0),
}


def view_target_axis(optical_axis, camera_up, camera_right, action_id: ActionId, cone_deg: float) -> np.ndarray:
    optical = normalize(optical_axis)
    up = normalize(np.asarray(camera_up, dtype=np.float64) - optical * np.dot(camera_up, optical))
    right = normalize(np.asarray(camera_right, dtype=np.float64) - optical * np.dot(camera_right, optical))
    if action_id == ActionId.HOLD_VIEW:
        return optical
    if action_id not in _VIEW_IMAGE_DIRECTIONS:
        raise ValueError(f"not a VIEW action: {action_id}")
    up_sign, right_sign = _VIEW_IMAGE_DIRECTIONS[action_id]
    image_direction = normalize(up_sign * up + right_sign * right)
    angle = math.radians(float(cone_deg))
    return normalize(math.cos(angle) * optical + math.sin(angle) * image_direction)


def unsigned_axis_tilt(axis, normal) -> float:
    cosine = float(np.clip(abs(np.dot(normalize(axis), normalize(normal))), 0.0, 1.0))
    return math.acos(cosine)


def move_direction(axis, normal, sign: int) -> np.ndarray:
    normal_value = normalize(normal)
    axis_value = normalize(axis)
    tangent_axis = axis_value - np.dot(axis_value, normal_value) * normal_value
    direction = normalize(np.cross(normal_value, normalize(tangent_axis)))
    if sign not in (-1, 1):
        raise ValueError("MOVE sign must be -1 or +1")
    return float(sign) * direction


def quintic_progress(value: float) -> float:
    t = float(np.clip(value, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
