"""Dependency-light contracts for TASK-006 six-DOF boundary latching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = "task006_hybrid_latched_v1"
PROFILE_RELATIVE_PATH = Path("configs/eleven_action/hybrid_latched_profile.json")
PROFILE_KEYS = {
    "schema_version",
    "physics_hz",
    "policy_rgb_hz",
    "view_error_limit_deg",
    "support_drift_limit_m",
    "release_window_s",
    "release_position_delta_limit_m",
    "release_axis_delta_limit_deg",
    "preferred_backend",
    "fallback_backend",
    "selected_backend",
}


class LatchIntent(str, Enum):
    NONE = "none"
    LOCK = "lock"
    UNLOCK = "unlock"


class LatchReason(str, Enum):
    INITIAL = "initial"
    HOLD = "hold"
    VIEW_TARGET = "view_target"
    CAMERA_CONTACT = "camera_contact"
    ACTION_BOUNDARY = "action_boundary"
    REJECTED_MOVE = "rejected_move"


class LatchBackendName(str, Enum):
    DYNAMIC_LOCK_FLAGS = "dynamic_lock_flags"
    TENSOR_DISABLE_SIMULATION = "tensor_disable_simulation"
    KINEMATIC = "kinematic"


@dataclass(frozen=True)
class LatchedContactSnapshot:
    """Immutable contact classification captured at a completed latch."""

    any_contact: bool = False
    camera_contact: bool = False
    sidewall_contact: bool = False
    source_physics_substep: int = 0

    def __post_init__(self) -> None:
        if self.source_physics_substep < 0:
            raise ValueError("source_physics_substep must be nonnegative")


@dataclass(frozen=True)
class LatchProfile:
    schema_version: str
    physics_hz: int
    policy_rgb_hz: int
    view_error_limit_deg: float
    support_drift_limit_m: float
    release_window_s: float
    release_position_delta_limit_m: float
    release_axis_delta_limit_deg: float
    preferred_backend: LatchBackendName
    fallback_backend: LatchBackendName
    selected_backend: LatchBackendName

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported latch profile schema: {self.schema_version!r}")
        if self.physics_hz != 240 or self.policy_rgb_hz != 1:
            raise ValueError("TASK-006 requires 240 Hz physics and 1 Hz policy RGB")
        numeric = (
            self.view_error_limit_deg,
            self.support_drift_limit_m,
            self.release_window_s,
            self.release_position_delta_limit_m,
            self.release_axis_delta_limit_deg,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("latch thresholds must be finite and positive")
        frozen = {
            "view_error_limit_deg": (self.view_error_limit_deg, 3.0),
            "support_drift_limit_m": (self.support_drift_limit_m, 0.002),
            "release_window_s": (self.release_window_s, 0.05),
            "release_position_delta_limit_m": (self.release_position_delta_limit_m, 0.0005),
            "release_axis_delta_limit_deg": (self.release_axis_delta_limit_deg, 1.0),
        }
        for name, (value, expected) in frozen.items():
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"{name} must remain {expected}")
        if self.preferred_backend is not LatchBackendName.DYNAMIC_LOCK_FLAGS:
            raise ValueError("preferred backend must remain dynamic_lock_flags")
        if self.fallback_backend is not LatchBackendName.KINEMATIC:
            raise ValueError("fallback backend must remain kinematic")


def _default_profile_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / PROFILE_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"tracked latch profile not found: {PROFILE_RELATIVE_PATH}")


def _profile_path(path: Path | None = None) -> Path:
    return _default_profile_path() if path is None else Path(path).expanduser().resolve()


def _load_raw(path: Path | None = None) -> dict[str, Any]:
    raw = json.loads(_profile_path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("latch profile root must be an object")
    missing = PROFILE_KEYS - set(raw)
    extra = set(raw) - PROFILE_KEYS
    if missing or extra:
        raise ValueError(f"latch profile keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return raw


def load_latch_profile(path: Path | None = None) -> LatchProfile:
    raw = _load_raw(path)
    return LatchProfile(
        schema_version=str(raw["schema_version"]),
        physics_hz=int(raw["physics_hz"]),
        policy_rgb_hz=int(raw["policy_rgb_hz"]),
        view_error_limit_deg=float(raw["view_error_limit_deg"]),
        support_drift_limit_m=float(raw["support_drift_limit_m"]),
        release_window_s=float(raw["release_window_s"]),
        release_position_delta_limit_m=float(raw["release_position_delta_limit_m"]),
        release_axis_delta_limit_deg=float(raw["release_axis_delta_limit_deg"]),
        preferred_backend=LatchBackendName(raw["preferred_backend"]),
        fallback_backend=LatchBackendName(raw["fallback_backend"]),
        selected_backend=LatchBackendName(raw["selected_backend"]),
    )


def latch_profile_sha256(path: Path | None = None) -> str:
    canonical = json.dumps(_load_raw(path), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
