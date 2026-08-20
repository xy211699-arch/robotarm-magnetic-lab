"""TASK-007 virtual-magnet eleven-action closed-loop controller."""

from .config import ClosedLoopProfile, load_profile, profile_sha256
from .controller import VirtualMagnetElevenActionController
from .geometry import (
    camera_image_axes_from_ros_rotation,
    move_direction,
    normalize,
    quintic_progress,
    unsigned_axis_tilt,
    view_target_axis,
)
from .pose_inverse import (
    PoseInverseResult,
    PoseInverseState,
    integrate_pose_increment,
    numerical_pose_jacobian,
    solve_pose_increment,
)
from .types import (
    ActionId,
    ActionResult,
    ControllerCommand,
    ControllerState,
    ControllerTelemetry,
    FrozenActionTarget,
    Lifecycle,
)

__all__ = [
    "ActionId",
    "ActionResult",
    "camera_image_axes_from_ros_rotation",
    "ClosedLoopProfile",
    "ControllerCommand",
    "ControllerState",
    "ControllerTelemetry",
    "FrozenActionTarget",
    "Lifecycle",
    "VirtualMagnetElevenActionController",
    "PoseInverseResult",
    "PoseInverseState",
    "integrate_pose_increment",
    "load_profile",
    "move_direction",
    "normalize",
    "numerical_pose_jacobian",
    "profile_sha256",
    "quintic_progress",
    "unsigned_axis_tilt",
    "view_target_axis",
    "solve_pose_increment",
]
