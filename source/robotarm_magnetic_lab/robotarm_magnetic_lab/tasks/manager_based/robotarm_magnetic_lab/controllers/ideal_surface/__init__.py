"""Public API for the privileged ideal-surface motion layer."""

from .action_mask import compute_action_mask
from .config import IdealSurfaceConfig
from .types import (
    ControllerSnapshot,
    ControllerState,
    IdealActionResult,
    IdealActionStatus,
    IdealSurfaceAction,
    START_TILT_ACTIONS,
    SurfaceFlags,
)

__all__ = [
    "ControllerSnapshot",
    "ControllerState",
    "IdealActionResult",
    "IdealActionStatus",
    "IdealSurfaceAction",
    "IdealSurfaceConfig",
    "START_TILT_ACTIONS",
    "SurfaceFlags",
    "compute_action_mask",
]
