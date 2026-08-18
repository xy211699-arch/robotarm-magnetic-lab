"""Tracked simulation-authority profile and shared controller configuration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = "task004_simulation_authority_v1"
PROFILE_RELATIVE_PATH = Path("configs/local_primitives/simulation_profile.json")
PROFILE_KEYS = {
    "schema_version", "axis_kp_nm_per_rad", "axis_kd_nms_per_rad",
    "roll_damping_nms_per_rad", "pose_torque_limit_nm", "anchor_kp_n_per_m",
    "anchor_kd_ns_per_m", "endpoint_pin_force_n", "total_force_limit_n",
    "total_torque_limit_nm", "force_slew_limit_n_per_s",
    "torque_slew_limit_nm_per_s", "motion_duration_s", "hard_timeout_s",
}


@dataclass(frozen=True)
class SimulationAuthorityProfile:
    """Simulation-only authority values loaded from the tracked JSON profile."""

    schema_version: str
    axis_kp_nm_per_rad: float
    axis_kd_nms_per_rad: float
    roll_damping_nms_per_rad: float
    pose_torque_limit_nm: float
    anchor_kp_n_per_m: float
    anchor_kd_ns_per_m: float
    endpoint_pin_force_n: float
    total_force_limit_n: float
    total_torque_limit_nm: float
    force_slew_limit_n_per_s: float
    torque_slew_limit_nm_per_s: float
    motion_duration_s: tuple[float, float, float, float]
    hard_timeout_s: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported profile schema: {self.schema_version!r}")
        positive = (
            self.axis_kp_nm_per_rad, self.axis_kd_nms_per_rad,
            self.roll_damping_nms_per_rad, self.pose_torque_limit_nm,
            self.anchor_kp_n_per_m, self.anchor_kd_ns_per_m,
            self.endpoint_pin_force_n, self.total_force_limit_n,
            self.total_torque_limit_nm, self.force_slew_limit_n_per_s,
            self.torque_slew_limit_nm_per_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("simulation authority values must be finite and positive")
        if self.pose_torque_limit_nm > self.total_torque_limit_nm:
            raise ValueError("pose torque limit cannot exceed total torque limit")
        if self.total_force_limit_n > 5.0:
            raise ValueError("total force exceeds the 5 N numerical envelope")
        if self.total_torque_limit_nm > 0.02:
            raise ValueError("total torque exceeds the 0.02 N m numerical envelope")
        if self.force_slew_limit_n_per_s > 50.0:
            raise ValueError("force slew exceeds the 50 N/s numerical envelope")
        if self.torque_slew_limit_nm_per_s > 0.2:
            raise ValueError("torque slew exceeds the 0.2 N m/s numerical envelope")
        if len(self.motion_duration_s) != 4 or len(self.hard_timeout_s) != 4:
            raise ValueError("duration tuples must contain exactly four values")
        if any(
            not (math.isfinite(motion) and math.isfinite(timeout) and 0.0 < motion < timeout < 10.0)
            for motion, timeout in zip(self.motion_duration_s, self.hard_timeout_s)
        ):
            raise ValueError("each duration must satisfy 0 < motion < timeout < 10 seconds")


def _default_profile_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / PROFILE_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"tracked simulation profile not found: {PROFILE_RELATIVE_PATH}")


def _profile_path(path: Path | None = None) -> Path:
    return _default_profile_path() if path is None else Path(path).expanduser().resolve()


def _load_raw_profile(path: Path | None = None) -> dict[str, Any]:
    raw = json.loads(_profile_path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("simulation profile root must be an object")
    missing = PROFILE_KEYS - set(raw)
    extra = set(raw) - PROFILE_KEYS
    if missing or extra:
        raise ValueError(f"profile keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return raw


def load_simulation_profile(path: Path | None = None) -> SimulationAuthorityProfile:
    """Load and strictly validate the tracked simulation-only profile."""

    raw = _load_raw_profile(path)
    values = dict(raw)
    for key in PROFILE_KEYS - {"schema_version", "motion_duration_s", "hard_timeout_s"}:
        if isinstance(raw[key], bool):
            raise ValueError(f"{key} must be numeric, not boolean")
        values[key] = float(raw[key])
    values["motion_duration_s"] = tuple(float(v) for v in raw["motion_duration_s"])
    values["hard_timeout_s"] = tuple(float(v) for v in raw["hard_timeout_s"])
    return SimulationAuthorityProfile(**values)


def simulation_profile_sha256(path: Path | None = None) -> str:
    """Hash canonical sorted compact JSON rather than file formatting."""

    canonical = json.dumps(_load_raw_profile(path), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LocalPrimitiveControllerCfg:
    """Geometry, acceptance thresholds, and one tracked authority profile."""

    capsule_mass_kg: float = 0.0057349997
    capsule_half_total_length_m: float = 0.0125
    capsule_half_cylinder_length_m: float = 0.006
    gravity_m_s2: float = 9.81
    axis_kp_nm_per_rad: float = 0.001
    axis_kd_nms_per_rad: float = 0.00008
    roll_damping_nms_per_rad: float = 0.00002
    pose_torque_limit_nm: float = 0.001
    anchor_kp_n_per_m: float = 10.0
    anchor_kd_ns_per_m: float = 0.4
    endpoint_pin_force_n: float = 0.1
    total_force_limit_n: float = 1.0
    total_torque_limit_nm: float = 0.005
    force_slew_limit_n_per_s: float = 20.0
    torque_slew_limit_nm_per_s: float = 0.05
    motion_duration_s: tuple[float, float, float, float] = (5.5, 4.5, 3.5, 8.0)
    hard_timeout_s: tuple[float, float, float, float] = (8.0, 7.0, 6.0, 9.5)
    profile_sha256: str = ""
    stable_duration_s: float = 0.4
    max_stable_linear_speed_m_s: float = 0.02
    max_stable_angular_speed_rad_s: float = 0.15
    transition_tolerance_rad: float = math.radians(3.0)
    tilt_tolerance_rad: float = math.radians(2.0)
    cone_tilt_rmse_limit_rad: float = math.radians(5.0)
    cone_coverage_tolerance_rad: float = math.radians(10.0)
    side_start_min_rad: float = math.radians(75.0)
    side_start_max_rad: float = math.radians(105.0)
    upright_start_max_rad: float = math.radians(5.0)
    cone_start_tolerance_rad: float = math.radians(3.0)
    target_tilt_rad: float = math.radians(30.0)

    def __post_init__(self) -> None:
        SimulationAuthorityProfile(
            schema_version=PROFILE_SCHEMA_VERSION,
            axis_kp_nm_per_rad=self.axis_kp_nm_per_rad,
            axis_kd_nms_per_rad=self.axis_kd_nms_per_rad,
            roll_damping_nms_per_rad=self.roll_damping_nms_per_rad,
            pose_torque_limit_nm=self.pose_torque_limit_nm,
            anchor_kp_n_per_m=self.anchor_kp_n_per_m,
            anchor_kd_ns_per_m=self.anchor_kd_ns_per_m,
            endpoint_pin_force_n=self.endpoint_pin_force_n,
            total_force_limit_n=self.total_force_limit_n,
            total_torque_limit_nm=self.total_torque_limit_nm,
            force_slew_limit_n_per_s=self.force_slew_limit_n_per_s,
            torque_slew_limit_nm_per_s=self.torque_slew_limit_nm_per_s,
            motion_duration_s=self.motion_duration_s,
            hard_timeout_s=self.hard_timeout_s,
        )
        if self.profile_sha256 and len(self.profile_sha256) != 64:
            raise ValueError("profile_sha256 must be empty or a 64-character digest")
        geometry = (
            self.capsule_mass_kg, self.capsule_half_total_length_m,
            self.capsule_half_cylinder_length_m, self.gravity_m_s2,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in geometry):
            raise ValueError("capsule geometry, mass, and gravity must be finite and positive")

    @property
    def weight_n(self) -> float:
        return self.capsule_mass_kg * self.gravity_m_s2

    @property
    def torque_limit_nm(self) -> float:
        return self.total_torque_limit_nm

    @property
    def xy_force_limit_n(self) -> float:
        return self.total_force_limit_n

    @property
    def downward_preload_n(self) -> float:
        return self.endpoint_pin_force_n


def make_local_primitive_controller_cfg(path: Path | None = None) -> LocalPrimitiveControllerCfg:
    """Build the shared controller config from the tracked profile."""

    profile = load_simulation_profile(path)
    values = {key: getattr(profile, key) for key in PROFILE_KEYS if key != "schema_version"}
    return LocalPrimitiveControllerCfg(**values, profile_sha256=simulation_profile_sha256(path))


LocalPrimitiveActionCfg = LocalPrimitiveControllerCfg


def make_local_primitive_action_cfg(path: Path | None = None) -> LocalPrimitiveControllerCfg:
    """Compatibility factory; Isaac Lab wraps the same tracked profile."""

    return make_local_primitive_controller_cfg(path)
