"""Optional interactive UI helpers for robotarm_magnetic_lab."""

from .capsule_camera_view import (
    CapsuleCameraViewHandle,
    CapsulePoseViewHandle,
    attach_capsule_camera_policy_view,
    attach_capsule_pose_view,
    configure_capsule_camera_view,
    configure_capsule_pose_view,
)

__all__ = [
    "CapsuleCameraViewHandle",
    "CapsulePoseViewHandle",
    "attach_capsule_camera_policy_view",
    "attach_capsule_pose_view",
    "configure_capsule_camera_view",
    "configure_capsule_pose_view",
]
