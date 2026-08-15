"""Public API for the privileged ideal-surface motion layer."""

from .action_mask import compute_action_mask
from .capsule_geometry import Spherocylinder
from .config import IdealSurfaceConfig
from .geometry import (
    LocalFrame,
    normalized,
    orientation_from_axis_and_image_up,
    quaternion_wxyz_from_matrix,
    quaternion_wxyz_to_matrix,
    quintic,
    rotation_matrix,
)
from .surface_mesh import SurfaceHit, SurfaceLostError, SurfaceNavigationMesh
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
    "LocalFrame",
    "Spherocylinder",
    "START_TILT_ACTIONS",
    "SurfaceHit",
    "SurfaceFlags",
    "SurfaceLostError",
    "SurfaceNavigationMesh",
    "compute_action_mask",
    "normalized",
    "orientation_from_axis_and_image_up",
    "quaternion_wxyz_from_matrix",
    "quaternion_wxyz_to_matrix",
    "quintic",
    "rotation_matrix",
]
