"""Strict tracked profile for the TASK-005 simulation-only controller."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = "task005_eleven_action_dynamic_v1"
PROFILE_RELATIVE_PATH = Path("configs/eleven_action/dynamic_profile.json")
PROFILE_KEYS = {
    "schema_version", "capsule_mass_kg", "capsule_radius_m",
    "capsule_cylinder_half_length_m", "physics_hz", "action_duration_s",
    "view_motion_duration_s", "view_hold_duration_s", "view_cone_half_angle_deg",
    "move_min_tilt_deg", "contact_history_s", "axis_kp_nm_per_rad",
    "axis_kd_nms_per_rad", "support_kp_n_per_m", "support_kd_ns_per_m",
    "support_normal_preload_n", "total_force_limit_n", "total_torque_limit_nm",
    "force_slew_limit_n_per_s", "torque_slew_limit_nm_per_s",
    "support_drift_limit_m", "move_force_k", "move_force_k_max", "move_force_k_step",
}
VIEW_KP_GRID = (0.005, 0.01, 0.02)
VIEW_KD_GRID = (0.0008, 0.0016, 0.0032)
SUPPORT_KP_GRID = (5.0, 10.0, 20.0)
SUPPORT_KD_GRID = (0.2, 0.4, 0.8)


def _exact_grid(value: float, candidates: tuple[float, ...]) -> bool:
    return any(math.isclose(value, item, rel_tol=0.0, abs_tol=1.0e-12) for item in candidates)


@dataclass(frozen=True)
class DynamicProfile:
    schema_version: str
    capsule_mass_kg: float
    capsule_radius_m: float
    capsule_cylinder_half_length_m: float
    physics_hz: int
    action_duration_s: float
    view_motion_duration_s: float
    view_hold_duration_s: float
    view_cone_half_angle_deg: float
    move_min_tilt_deg: float
    contact_history_s: float
    axis_kp_nm_per_rad: float
    axis_kd_nms_per_rad: float
    support_kp_n_per_m: float
    support_kd_ns_per_m: float
    support_normal_preload_n: float
    total_force_limit_n: float
    total_torque_limit_nm: float
    force_slew_limit_n_per_s: float
    torque_slew_limit_nm_per_s: float
    support_drift_limit_m: float
    move_force_k: float
    move_force_k_max: float
    move_force_k_step: float

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported profile schema: {self.schema_version!r}")
        numeric = [
            value for name, value in self.__dict__.items()
            if name not in ("schema_version", "physics_hz")
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("profile numeric values must be finite")
        if self.physics_hz != 240:
            raise ValueError("physics_hz must be exactly 240")
        if self.action_substeps != 240:
            raise ValueError("action duration must produce exactly 240 substeps")
        if self.view_motion_substeps != 192 or self.view_hold_substeps != 48:
            raise ValueError("VIEW timing must contain 192 motion and 48 hold substeps")
        if self.view_motion_substeps + self.view_hold_substeps != self.action_substeps:
            raise ValueError("VIEW timing must sum to the one-second action")
        if self.contact_history_substeps != 12:
            raise ValueError("contact history must contain exactly 12 substeps")
        positive = (
            self.capsule_mass_kg, self.capsule_radius_m, self.capsule_cylinder_half_length_m,
            self.axis_kp_nm_per_rad, self.axis_kd_nms_per_rad, self.support_kp_n_per_m,
            self.support_kd_ns_per_m, self.support_normal_preload_n,
            self.total_force_limit_n, self.total_torque_limit_nm,
            self.force_slew_limit_n_per_s, self.torque_slew_limit_nm_per_s,
            self.support_drift_limit_m,
        )
        if not all(value > 0.0 for value in positive):
            raise ValueError("profile authority and geometry must be positive")
        if not _exact_grid(self.axis_kp_nm_per_rad, VIEW_KP_GRID):
            raise ValueError("axis_kp_nm_per_rad is outside the authorized grid")
        if not _exact_grid(self.axis_kd_nms_per_rad, VIEW_KD_GRID):
            raise ValueError("axis_kd_nms_per_rad is outside the authorized grid")
        if not _exact_grid(self.support_kp_n_per_m, SUPPORT_KP_GRID):
            raise ValueError("support_kp_n_per_m is outside the authorized grid")
        if not _exact_grid(self.support_kd_ns_per_m, SUPPORT_KD_GRID):
            raise ValueError("support_kd_ns_per_m is outside the authorized grid")
        locked = {
            "capsule_mass_kg": (self.capsule_mass_kg, 0.0057349997),
            "capsule_radius_m": (self.capsule_radius_m, 0.0065),
            "capsule_cylinder_half_length_m": (self.capsule_cylinder_half_length_m, 0.006),
            "view_cone_half_angle_deg": (self.view_cone_half_angle_deg, 15.0),
            "move_min_tilt_deg": (self.move_min_tilt_deg, 60.0),
            "support_normal_preload_n": (self.support_normal_preload_n, 0.1),
            "total_force_limit_n": (self.total_force_limit_n, 1.25),
            "total_torque_limit_nm": (self.total_torque_limit_nm, 0.02),
            "force_slew_limit_n_per_s": (self.force_slew_limit_n_per_s, 50.0),
            "torque_slew_limit_nm_per_s": (self.torque_slew_limit_nm_per_s, 0.2),
            "support_drift_limit_m": (self.support_drift_limit_m, 0.002),
            "move_force_k_max": (self.move_force_k_max, 3.0),
            "move_force_k_step": (self.move_force_k_step, 0.1),
        }
        for name, (value, expected) in locked.items():
            if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"{name} must remain {expected}")
        if not 0.9 - 1.0e-12 <= self.move_force_k <= self.move_force_k_max + 1.0e-12:
            raise ValueError("move_force_k must be inside [0.9, 3.0]")
        grid_index = (self.move_force_k - 0.9) / self.move_force_k_step
        if not math.isclose(grid_index, round(grid_index), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("move_force_k must lie on the 0.1 grid")

    @property
    def action_substeps(self) -> int:
        return _exact_substeps(self.action_duration_s, self.physics_hz)

    @property
    def view_motion_substeps(self) -> int:
        return _exact_substeps(self.view_motion_duration_s, self.physics_hz)

    @property
    def view_hold_substeps(self) -> int:
        return _exact_substeps(self.view_hold_duration_s, self.physics_hz)

    @property
    def contact_history_substeps(self) -> int:
        return _exact_substeps(self.contact_history_s, self.physics_hz)

    @property
    def gravity_m_s2(self) -> float:
        return 9.81

    @property
    def move_force_n(self) -> float:
        return self.move_force_k * self.capsule_mass_kg * self.gravity_m_s2


def _exact_substeps(duration_s: float, physics_hz: int) -> int:
    raw = float(duration_s) * int(physics_hz)
    rounded = round(raw)
    if not math.isclose(raw, rounded, rel_tol=0.0, abs_tol=1.0e-10):
        return -1
    return int(rounded)


def _default_profile_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / PROFILE_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"tracked dynamic profile not found: {PROFILE_RELATIVE_PATH}")


def _profile_path(path: Path | None = None) -> Path:
    return _default_profile_path() if path is None else Path(path).expanduser().resolve()


def _load_raw(path: Path | None = None) -> dict[str, Any]:
    raw = json.loads(_profile_path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dynamic profile root must be an object")
    missing = PROFILE_KEYS - set(raw)
    extra = set(raw) - PROFILE_KEYS
    if missing or extra:
        raise ValueError(f"profile keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return raw


def load_dynamic_profile(path: Path | None = None) -> DynamicProfile:
    raw = _load_raw(path)
    values: dict[str, Any] = {"schema_version": raw["schema_version"]}
    for key in PROFILE_KEYS - {"schema_version", "physics_hz"}:
        if isinstance(raw[key], bool):
            raise ValueError(f"{key} must be numeric, not boolean")
        values[key] = float(raw[key])
    if isinstance(raw["physics_hz"], bool):
        raise ValueError("physics_hz must be an integer")
    values["physics_hz"] = int(raw["physics_hz"])
    if values["physics_hz"] != raw["physics_hz"]:
        raise ValueError("physics_hz must be an integer")
    return DynamicProfile(**values)


def dynamic_profile_sha256(path: Path | None = None) -> str:
    canonical = json.dumps(_load_raw(path), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
