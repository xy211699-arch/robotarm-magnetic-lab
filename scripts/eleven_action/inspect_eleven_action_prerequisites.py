"""TASK-005 source/live prerequisite inspection for flat and stomach tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))


def source_report() -> dict:
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
        dynamic_profile_sha256,
        load_dynamic_profile,
    )

    profile = load_dynamic_profile()
    runtime = ROOT / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/eleven_action.py"
    text = runtime.read_text(encoding="utf-8")
    forbidden_fragments = (
        "write_root_" + "pose_to_sim",
        "write_root_" + "velocity_to_sim",
        "set_world_" + "pose",
        "set_linear_" + "velocity",
        "set_angular_" + "velocity",
    )
    return {
        "physics_hz": profile.physics_hz,
        "mass_kg": profile.capsule_mass_kg,
        "geometry": {
            "radius_m": profile.capsule_radius_m,
            "cylinder_half_length_m": profile.capsule_cylinder_half_length_m,
            "directed_axis_local": [0.0, 0.0, -1.0],
        },
        "profile_sha256": dynamic_profile_sha256(),
        "com_wrench": "positions=None" in text and "is_global=True" in text,
        "contact_fields": ["position", "normal", "impulse"],
        "forbidden_runtime_calls": [item for item in forbidden_fragments if item in text],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    args, _ = parser.parse_known_args()
    report = source_report()
    report["requested_task"] = args.task
    report["requested_device"] = args.device
    report["gate"] = "pass" if not report["forbidden_runtime_calls"] and report["com_wrench"] else "needs_decision"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
