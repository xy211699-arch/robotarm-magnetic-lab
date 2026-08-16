"""TASK-003 live physics inspection and strict real-dynamics decision gate.

The schema and validator intentionally have no Isaac Sim imports so focused
tests can exercise the contract before launching Kit.  Live inspection is
added below the pure gate and imports version-dependent APIs only at runtime.
"""

from __future__ import annotations

import math
from typing import Any


TASK_ID = "Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0"
REQUIRED_REPORT_KEYS = (
    "repository",
    "task",
    "physics",
    "capsule",
    "stomach",
    "contact_sensor",
    "runtime_contract",
    "gate",
)
FORBIDDEN_RUNTIME_CALLS = (
    "write_root_pose_to_sim",
    "write_root_velocity_to_sim",
    "set_transforms",
    "set_velocities",
)


def _close(value: Any, expected: float, tolerance: float = 1.0e-9) -> bool:
    try:
        return math.isfinite(float(value)) and abs(float(value) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def build_gate(report: dict) -> dict:
    """Evaluate the immutable TASK-003 preflight contract."""
    failures: list[str] = []
    missing = [key for key in REQUIRED_REPORT_KEYS[:-1] if key not in report]
    if missing:
        return {
            "status": "needs_decision",
            "failures": [f"missing report section: {key}" for key in missing],
        }

    capsule = report["capsule"]
    physics = report["physics"]
    stomach = report["stomach"]
    task = report["task"]
    runtime = report["runtime_contract"]

    if task.get("id") != TASK_ID:
        failures.append("unexpected task id")
    if int(task.get("num_envs", 0)) != 1:
        failures.append("task must use exactly one environment")
    if list(task.get("action_terms", [])) != ["dynamic_force"]:
        failures.append("dynamic_force must be the only action term")
    forbidden_terms = list(runtime.get("magnetic_or_ideal_terms", []))
    if forbidden_terms:
        failures.append(f"forbidden actuator terms are active: {forbidden_terms}")

    if not _close(physics.get("dt_s"), 1.0 / 240.0):
        failures.append("physics dt is not 1/240 s")
    if not _close(physics.get("environment_rate_hz"), 60.0):
        failures.append("environment rate is not 60 Hz")
    if int(physics.get("render_interval", 0)) != 4:
        failures.append("render interval is not four physics steps")
    if not bool(physics.get("scene_ccd_enabled")):
        failures.append("CCD is not active at scene and body levels")

    if bool(capsule.get("kinematic_enabled")):
        failures.append("capsule is kinematic")
    if not bool(capsule.get("gravity_enabled")):
        failures.append("capsule gravity is disabled")
    if not bool(capsule.get("ccd_enabled")):
        failures.append("CCD is not active at scene and body levels")
    if not bool(capsule.get("collision_enabled")):
        failures.append("capsule collision is disabled")
    if capsule.get("shape") != "Capsule" or str(capsule.get("axis", "")).upper() != "Z":
        failures.append("capsule collider is not a Z-axis spherocylinder")
    if not _close(capsule.get("radius_m"), 0.0065):
        failures.append("capsule radius is not 6.5 mm")
    if not _close(capsule.get("cylinder_height_m"), 0.012):
        failures.append("capsule cylinder height is not 12 mm")
    if not _close(capsule.get("total_length_m"), 0.025):
        failures.append("capsule total length is not 25 mm")
    if not _close(capsule.get("mass_kg"), float(capsule.get("mass_kg", 0.0)), 0.0):
        failures.append("capsule mass is not finite")
    try:
        if float(capsule.get("mass_kg", 0.0)) <= 0.0:
            failures.append("capsule mass is not positive")
    except (TypeError, ValueError):
        failures.append("capsule mass is not positive")
    inertia = capsule.get("inertia_kg_m2", [])
    try:
        if len(inertia) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in inertia):
            failures.append("capsule inertia is not finite and positive")
    except (TypeError, ValueError):
        failures.append("capsule inertia is not finite and positive")

    if not bool(stomach.get("collision_enabled")):
        failures.append("stomach collision is disabled")
    if not bool(stomach.get("static")):
        failures.append("stomach collider is not static")
    if not bool(report["contact_sensor"].get("present")):
        failures.append("contact sensor is unavailable")
    if list(runtime.get("forbidden_calls", [])):
        failures.append("forbidden runtime state writer")
    if not bool(runtime.get("force_at_center_of_mass")):
        failures.append("force application at center of mass is unverified")
    if not bool(runtime.get("commanded_torque_zero")):
        failures.append("commanded torque is not identically zero")

    return {"status": "pass" if not failures else "needs_decision", "failures": failures}


def validate_preflight_report(report: dict) -> None:
    """Raise when a report cannot prove the frozen real-dynamics contract."""
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"missing report sections: {missing}")
    evaluated = build_gate(report)
    recorded = report.get("gate")
    if recorded != evaluated:
        raise ValueError(f"stale or inconsistent gate: recorded={recorded}, evaluated={evaluated}")
    if evaluated["status"] != "pass":
        raise ValueError("; ".join(evaluated["failures"]))


if __name__ == "__main__":
    raise SystemExit(
        "Live preflight is not available until the TASK-003 task configuration is implemented."
    )
