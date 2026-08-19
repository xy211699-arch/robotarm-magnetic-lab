"""Isaac/PhysX runtime adapter for TASK-006 six-DOF boundary latching.

The dependency-light controller requests lock transitions.  This adapter is
the only layer allowed to clear capsule velocity and change rigid-body
authority.  It never writes a root pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..controllers.eleven_action.latch import LatchBackendName, LatchReason


def _vector(value, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(size).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LatchReadback:
    backend: LatchBackendName
    latched: bool
    position_world_m: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity_world_m_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray
    locked_position_axis_mask: int
    locked_rotation_axis_mask: int
    kinematic_enabled: bool
    reason: LatchReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", LatchBackendName(self.backend))
        if self.reason is not None:
            object.__setattr__(self, "reason", LatchReason(self.reason))
        object.__setattr__(self, "position_world_m", _vector(self.position_world_m, 3))
        object.__setattr__(self, "quaternion_wxyz", _vector(self.quaternion_wxyz, 4))
        object.__setattr__(self, "linear_velocity_world_m_s", _vector(self.linear_velocity_world_m_s, 3))
        object.__setattr__(self, "angular_velocity_world_rad_s", _vector(self.angular_velocity_world_rad_s, 3))


def _get_or_create(api: Any, stem: str):
    attribute = getattr(api, f"Get{stem}Attr")()
    if not attribute:
        attribute = getattr(api, f"Create{stem}Attr")()
    return attribute


class CapsuleLatchRuntime:
    """Apply one explicitly selected latch backend to one Isaac Lab rigid object."""

    def __init__(
        self,
        capsule: Any,
        physx_api: Any,
        rigid_api: Any | None,
        backend: LatchBackendName,
    ) -> None:
        self.capsule = capsule
        self.physx_api = physx_api
        self.rigid_api = rigid_api
        self.backend = LatchBackendName(backend)
        self._last_reason: LatchReason | None = None

    @classmethod
    def dynamic_lock_flags(cls, capsule: Any, physx_api: Any, rigid_api: Any | None = None):
        return cls(capsule, physx_api, rigid_api, LatchBackendName.DYNAMIC_LOCK_FLAGS)

    @classmethod
    def kinematic(cls, capsule: Any, physx_api: Any, rigid_api: Any):
        return cls(capsule, physx_api, rigid_api, LatchBackendName.KINEMATIC)

    @classmethod
    def auto_fallback(cls, *_args, **_kwargs):
        raise RuntimeError("fallback requires tracked profile selection")

    def _clear_wrench(self) -> None:
        composer = getattr(self.capsule, "permanent_wrench_composer", None)
        if composer is not None:
            composer.reset()

    def _zero_velocity(self) -> None:
        try:
            import torch

            device = self.capsule.data.root_com_vel_w.torch.device
            zeros = torch.zeros((1, 6), dtype=torch.float32, device=device)
        except (ImportError, AttributeError):
            zeros = np.zeros((1, 6), dtype=np.float32)
        writer = getattr(self.capsule, "write_root_velocity_to_sim_index", None)
        if writer is None:
            writer = getattr(self.capsule, "write_root_velocity_to_sim", None)
        if writer is None:
            raise RuntimeError("capsule root-velocity writer is unavailable")
        try:
            writer(root_velocity=zeros)
        except TypeError:
            writer(zeros)

    def _set_dynamic_masks(self, value: int) -> None:
        _get_or_create(self.physx_api, "LockedPosAxis").Set(int(value))
        _get_or_create(self.physx_api, "LockedRotAxis").Set(int(value))

    def _set_kinematic(self, enabled: bool) -> None:
        if self.rigid_api is None:
            raise RuntimeError("kinematic backend requires UsdPhysics.RigidBodyAPI")
        _get_or_create(self.rigid_api, "KinematicEnabled").Set(bool(enabled))

    def lock_current(self, _state: Any, reason: LatchReason) -> LatchReadback:
        self._clear_wrench()
        self._zero_velocity()
        if self.backend is LatchBackendName.DYNAMIC_LOCK_FLAGS:
            self._set_dynamic_masks(0b111)
        else:
            self._set_kinematic(True)
        self._last_reason = LatchReason(reason)
        result = self.readback()
        if not result.latched:
            raise RuntimeError("latch backend readback mismatch")
        return result

    def unlock_zeroed(self, _state: Any) -> LatchReadback:
        self._clear_wrench()
        self._zero_velocity()
        if self.backend is LatchBackendName.DYNAMIC_LOCK_FLAGS:
            self._set_dynamic_masks(0)
        else:
            self._set_kinematic(False)
        result = self.readback()
        if result.latched:
            raise RuntimeError("unlock backend readback mismatch")
        return result

    def readback(self) -> LatchReadback:
        position, quaternion, linear, angular = self._read_capsule_state()
        pos_attr = _get_or_create(self.physx_api, "LockedPosAxis")
        rot_attr = _get_or_create(self.physx_api, "LockedRotAxis")
        pos_mask = int(pos_attr.Get() or 0)
        rot_mask = int(rot_attr.Get() or 0)
        kinematic = False
        if self.rigid_api is not None:
            kinematic = bool(_get_or_create(self.rigid_api, "KinematicEnabled").Get() or False)
        latched = (
            pos_mask == 0b111 and rot_mask == 0b111
            if self.backend is LatchBackendName.DYNAMIC_LOCK_FLAGS
            else kinematic
        )
        return LatchReadback(
            backend=self.backend,
            latched=latched,
            position_world_m=position,
            quaternion_wxyz=quaternion,
            linear_velocity_world_m_s=linear,
            angular_velocity_world_rad_s=angular,
            locked_position_axis_mask=pos_mask,
            locked_rotation_axis_mask=rot_mask,
            kinematic_enabled=kinematic,
            reason=self._last_reason,
        )

    def _read_capsule_state(self):
        data = getattr(self.capsule, "data", None)
        if data is not None and hasattr(data, "root_link_pose_w"):
            pose = data.root_link_pose_w.torch[0].detach().cpu().numpy()
            velocity = data.root_com_vel_w.torch[0].detach().cpu().numpy()
            quaternion = pose[3:7][[3, 0, 1, 2]]
            return pose[:3], quaternion, velocity[:3], velocity[3:6]
        state = self.capsule.state()
        return (
            state.position_world_m,
            state.quaternion_wxyz,
            state.linear_velocity_world_m_s,
            state.angular_velocity_world_rad_s,
        )
