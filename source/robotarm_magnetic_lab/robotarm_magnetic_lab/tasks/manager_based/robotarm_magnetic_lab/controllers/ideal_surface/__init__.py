"""Public API for the privileged ideal-surface motion layer."""

from .action_mask import compute_action_mask
from .capsule_geometry import Spherocylinder
from .config import IdealSurfaceConfig
from .contact import (
    ActiveAnchor,
    CapsulePose,
    ContactAssessment,
    ContactClassifier,
    ContactClassifierResult,
    SweptTargetAssessment,
    assess_pose,
    assess_swept_target,
    separate_initial_capsule_from_surface,
    select_active_anchor,
)
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
from .controller import ControllerOutput, IdealSurfaceController
from .trajectory import ActionTarget, TrajectoryEvaluation, evaluate_trajectory, target_for_action
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
    "ControllerOutput",
    "ControllerState",
    "ActiveAnchor",
    "CapsulePose",
    "ContactAssessment",
    "ContactClassifier",
    "ContactClassifierResult",
    "IdealActionResult",
    "IdealActionStatus",
    "IdealSurfaceAction",
    "IdealSurfaceConfig",
    "IdealSurfaceController",
    "LocalFrame",
    "Spherocylinder",
    "START_TILT_ACTIONS",
    "SurfaceHit",
    "SurfaceFlags",
    "SurfaceLostError",
    "SurfaceNavigationMesh",
    "ActionTarget",
    "TrajectoryEvaluation",
    "SweptTargetAssessment",
    "assess_pose",
    "assess_swept_target",
    "separate_initial_capsule_from_surface",
    "compute_action_mask",
    "evaluate_trajectory",
    "normalized",
    "orientation_from_axis_and_image_up",
    "quaternion_wxyz_from_matrix",
    "quaternion_wxyz_to_matrix",
    "quintic",
    "rotation_matrix",
    "select_active_anchor",
    "target_for_action",
]
