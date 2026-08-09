"""Optional interactive UI helpers for robotarm_magnetic_lab."""

from .capsule_camera_view import (
    CapsuleCameraViewHandle,
    CapsulePoseViewHandle,
    attach_capsule_camera_policy_view,
    attach_capsule_pose_view,
    configure_capsule_camera_view,
    configure_capsule_pose_view,
)
from .coverage_view import (
    KitCoveragePointCloudView,
    ProjectionConfig,
    coverage_colors,
    export_coverage_projection,
)

__all__ = [
    "CapsuleCameraViewHandle",
    "CapsulePoseViewHandle",
    "attach_capsule_camera_policy_view",
    "attach_capsule_pose_view",
    "configure_capsule_camera_view",
    "configure_capsule_pose_view",
    "KitCoveragePointCloudView",
    "ProjectionConfig",
    "coverage_colors",
    "export_coverage_projection",
]
