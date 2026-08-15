from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ideal_surface"
    / "inspect_ideal_surface_prerequisites.py"
)
SPEC = importlib.util.spec_from_file_location("ideal_surface_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_report():
    return {
        "repository": {"commit": "a" * 40, "branch": "feature/test"},
        "task": {
            "id": "Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0"
        },
        "capsule": {
            "shape_class": "spherocylinder",
            "radius_m": 0.005,
            "cylinder_half_length_m": 0.0075,
            "long_axis_local": [0, 0, 1],
        },
        "camera": {
            "optical_axis_local": [0, 0, 1],
            "image_up_axis_local": [0, -1, 0],
        },
        "surface": {
            "vertex_count": 24529,
            "triangle_count": 49047,
            "geometry_sha256": (
                "67b4e06a4f5cfc3b8d51e5411942226"
                "d4bcabd3a6a937a456057e408a990ad36"
            ),
            "inward_normal_confirmed": True,
        },
        "pose_write_api": {
            "pose_method": "write_root_pose_to_sim",
            "velocity_method": "write_root_velocity_to_sim",
            "quaternion_order": "wxyz",
        },
        "initial_contact": {"valid": True, "triangle_id": 1},
        "gate": {"status": "pass", "failures": []},
    }


def test_preflight_report_requires_all_design_gates():
    report = valid_report()
    MODULE.validate_preflight_report(report)
    assert set(report) == MODULE.REQUIRED_REPORT_KEYS
    assert report["task"]["id"] == (
        "Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0"
    )
    assert report["capsule"]["radius_m"] > 0.0
    assert report["capsule"]["cylinder_half_length_m"] > 0.0
    assert report["capsule"]["long_axis_local"] in ([0, 0, 1], [0, 0, -1])
    assert report["camera"]["optical_axis_local"] == [0, 0, 1]
    assert report["surface"]["triangle_count"] == 49047
    assert report["surface"]["vertex_count"] == 24529
    assert report["pose_write_api"]["pose_method"]
    assert report["pose_write_api"]["velocity_method"]
    assert report["gate"]["status"] in {"pass", "needs_decision"}
