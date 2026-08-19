"""Strict loading for the one shared TASK-007 controller profile."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = "task007_virtual_magnet_closed_loop_v1"


@dataclass(frozen=True)
class ClosedLoopProfile:
    schema_version: str
    physics_hz: int
    feedback_hz: int
    action_hz: int
    action_duration_s: float
    motion_duration_s: float
    stabilization_duration_s: float
    view_cone_deg: float
    move_target_m: float
    move_acceptance_min_m: float
    move_acceptance_max_m: float
    move_tilt_min_deg: float
    contact_window_s: float
    stability_window_s: float
    boundary_linear_speed_m_s: float
    boundary_angular_speed_rad_s: float
    hold_axis_kp_nm: float
    hold_axis_kd_nm_s: float
    hold_anchor_kp_n_m: float
    hold_anchor_kd_n_s_m: float
    view_axis_kp_nm: float
    view_axis_kd_nm_s: float
    view_anchor_kp_n_m: float
    view_anchor_kd_n_s_m: float
    move_tangent_kp_n_m: float
    move_tangent_kd_n_s_m: float
    move_cross_kp_n_m: float
    move_cross_kd_n_s_m: float
    max_desired_force_n: float
    max_desired_torque_nm: float
    stabilization_max_desired_torque_nm: float
    force_weights: list[float]
    torque_weights: list[float]
    move_force_weights: list[float]
    move_torque_weights: list[float]
    translation_fd_step_m: float
    rotation_fd_step_rad: float
    inverse_damping: float
    relative_regularization: float
    translation_trust_m: float
    rotation_trust_rad: float
    stabilization_rotation_trust_rad: float
    minimum_separation_m: float
    maximum_separation_m: float
    maximum_relative_angle_rad: float
    condition_limit: float
    nominal_position_capsule_m: list[float]
    nominal_quaternion_capsule_xyzw: list[float]
    wrench_filter_time_constant_s: float
    coupling_ramp_s: float
    feedback_enabled: bool

    @property
    def action_substeps(self) -> int:
        return int(round(self.physics_hz * self.action_duration_s))

    @property
    def feedback_stride(self) -> int:
        return self.physics_hz // self.feedback_hz

    @property
    def motion_substeps(self) -> int:
        return int(round(self.physics_hz * self.motion_duration_s))

    @property
    def stabilization_substeps(self) -> int:
        return self.action_substeps - self.motion_substeps

    @property
    def contact_window_substeps(self) -> int:
        return int(round(self.physics_hz * self.contact_window_s))

    @property
    def stability_window_substeps(self) -> int:
        return int(round(self.physics_hz * self.stability_window_s))


def default_profile_path() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        candidate = parent / "configs/virtual_magnet/closed_loop_v1.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("configs/virtual_magnet/closed_loop_v1.json")


def _finite(value) -> bool:
    if isinstance(value, bool) or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return False


def load_profile(path: str | Path | None = None) -> ClosedLoopProfile:
    source = default_profile_path() if path is None else Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    expected = {item.name for item in fields(ClosedLoopProfile)}
    if set(data) != expected:
        raise ValueError(f"profile keys mismatch missing={sorted(expected-set(data))} unknown={sorted(set(data)-expected)}")
    if not all(_finite(value) for value in data.values()):
        raise ValueError("profile contains a non-finite or unsupported value")
    profile = ClosedLoopProfile(**data)
    if profile.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported profile schema: {profile.schema_version}")
    if profile.physics_hz != 240 or profile.feedback_hz != 60 or profile.action_hz != 1:
        raise ValueError("TASK-007 rates must be 240/60/1 Hz")
    if profile.action_substeps != 240 or profile.feedback_stride != 4:
        raise ValueError("TASK-007 action cadence must be 240 substeps with 60 Hz feedback")
    if profile.motion_substeps != 192 or profile.stabilization_substeps != 48:
        raise ValueError("TASK-007 timing must be 0.8 s motion plus 0.2 s stabilization")
    if profile.contact_window_substeps != 12 or profile.stability_window_substeps != 24:
        raise ValueError("contact/stability windows must be 12/24 substeps")
    if not (profile.move_acceptance_min_m <= profile.move_target_m <= profile.move_acceptance_max_m):
        raise ValueError("MOVE target must lie in its acceptance interval")
    weight_groups = (
        profile.force_weights,
        profile.torque_weights,
        profile.move_force_weights,
        profile.move_torque_weights,
    )
    if any(len(group) != 3 for group in weight_groups):
        raise ValueError("wrench weights must contain three force and three torque values per action family")
    if len(profile.nominal_position_capsule_m) != 3 or len(profile.nominal_quaternion_capsule_xyzw) != 4:
        raise ValueError("nominal relative pose has invalid shape")
    positive = [
        profile.translation_fd_step_m,
        profile.rotation_fd_step_rad,
        profile.translation_trust_m,
        profile.rotation_trust_rad,
        profile.stabilization_rotation_trust_rad,
        profile.minimum_separation_m,
        profile.maximum_separation_m,
        profile.max_desired_force_n,
        profile.max_desired_torque_nm,
        profile.stabilization_max_desired_torque_nm,
    ]
    if min(positive) <= 0.0 or profile.minimum_separation_m >= profile.maximum_separation_m:
        raise ValueError("profile limits must be positive and ordered")
    return profile


def profile_sha256(path: str | Path | None = None) -> str:
    source = default_profile_path() if path is None else Path(path)
    return hashlib.sha256(source.read_bytes()).hexdigest()
