"""Configuration loading and validation for the repository-local magnetic model."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from pathlib import Path


def _default_resource():
    return importlib.resources.files(__package__).joinpath("resources/default.json")


def config_sha256() -> str:
    """Return the digest of the authoritative packaged magnetic configuration."""
    return hashlib.sha256(_default_resource().read_bytes()).hexdigest()


def load_config(resource_root: str | Path | None = None):
    """Load the packaged config, or an explicit legacy root for regression only."""
    if resource_root is None:
        with _default_resource().open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    else:
        path = Path(resource_root) / "data" / "config" / "default.json"
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)

    cube = config["magnets"]["main_cube"]
    if len(cube["dimensions_m"]) != 3 or min(cube["dimensions_m"]) <= 0:
        raise ValueError("main_cube.dimensions_m must contain three positive values")
    if cube["remanence_t"] <= 0:
        raise ValueError("main_cube.remanence_t must be positive")
    robot = config["robot"]
    if len(robot["arm_joint_names"]) != len(robot["goal_joint_positions_rad"]):
        raise ValueError("arm joint names and goal positions must have equal lengths")
    if robot["max_joint_speed_rad_s"] <= 0:
        raise ValueError("robot.max_joint_speed_rad_s must be positive")
    if robot["settle_position_delta_rad"] <= 0:
        raise ValueError("robot.settle_position_delta_rad must be positive")
    if robot["settle_velocity_tolerance_rad_s"] <= 0:
        raise ValueError("robot.settle_velocity_tolerance_rad_s must be positive")
    if robot["settle_required_frames"] < 1 or robot["settle_timeout_s"] <= 0:
        raise ValueError("robot settle frame count and timeout must be positive")
    safe_roll = float(robot["collision_safe_roll_rad"])
    safe_range = robot["collision_safe_roll_range_rad"]
    if len(safe_range) != 2 or not safe_range[0] <= safe_roll <= safe_range[1]:
        raise ValueError("collision_safe_roll_rad must lie inside its safe range")
    cylinder = config["magnets"]["target_cylinder"]
    if cylinder["mass_kg"] <= 0:
        raise ValueError("target_cylinder.mass_kg must be positive")
    if cylinder["diameter_m"] <= 0 or cylinder["height_m"] <= 0:
        raise ValueError("target_cylinder dimensions must be positive")
    capsule = config["external_magnet"]["capsule"]
    if capsule["diameter_m"] <= cylinder["diameter_m"]:
        raise ValueError("capsule diameter must exceed the internal magnet diameter")
    if capsule["length_m"] <= cylinder["height_m"]:
        raise ValueError("capsule length must exceed the internal magnet length")
    if capsule["wall_thickness_m"] <= 0 or capsule["total_mass_kg"] <= 0:
        raise ValueError("capsule wall thickness and mass must be positive")
    planning = config["planning"]
    if len(planning["tool_axis_local"]) != 3:
        raise ValueError("planning.tool_axis_local must contain three values")
    if len(planning["task_tool_to_l6_rpy_deg"]) != 3:
        raise ValueError("planning.task_tool_to_l6_rpy_deg must contain three values")
    if planning["tool_position_tolerance_m"] <= 0:
        raise ValueError("planning.tool_position_tolerance_m must be positive")
    if not 0 < planning["execution_speed_scale"] <= 1:
        raise ValueError("planning.execution_speed_scale must be in (0, 1]")
    for key in ("asm_collision_inflation_m", "required_self_clearance_m", "max_tracking_error_rad"):
        if planning[key] < 0:
            raise ValueError(f"planning.{key} must be non-negative")
    for key in (
        "trajectory_validation_dt_s",
        "lookahead_horizon_s",
        "replan_cooldown_s",
        "graph_max_joint_speed_rad_s",
        "graph_max_joint_acceleration_rad_s2",
        "mount_translation_tolerance_m",
        "mount_rotation_tolerance_deg",
    ):
        if planning[key] <= 0:
            raise ValueError(f"planning.{key} must be positive")
    if planning["graph_fallback_roll_samples"] < 2:
        raise ValueError("planning.graph_fallback_roll_samples must be at least 2")
    return config
