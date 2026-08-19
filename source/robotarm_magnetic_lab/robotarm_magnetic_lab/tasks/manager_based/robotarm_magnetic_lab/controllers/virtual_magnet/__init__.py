"""TASK-007 virtual-magnet eleven-action closed-loop controller."""

from .config import ClosedLoopProfile, load_profile, profile_sha256
from .geometry import move_direction, normalize, quintic_progress, unsigned_axis_tilt, view_target_axis
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
    "ClosedLoopProfile",
    "ControllerCommand",
    "ControllerState",
    "ControllerTelemetry",
    "FrozenActionTarget",
    "Lifecycle",
    "load_profile",
    "move_direction",
    "normalize",
    "profile_sha256",
    "quintic_progress",
    "unsigned_axis_tilt",
    "view_target_axis",
]
