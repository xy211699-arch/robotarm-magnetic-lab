"""Focused TASK-003 test fixtures and source-tree import setup."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(PACKAGE_ROOT))


@pytest.fixture
def valid_report():
    template = {
        "repository": {
            "path": str(REPOSITORY),
            "branch": "feature/TASK-003-dynamic-capsule-force-teleop",
            "commit": "0" * 40,
        },
        "task": {
            "id": "Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0",
            "num_envs": 1,
            "action_terms": ["dynamic_force"],
            "event_terms": ["reset_scene"],
        },
        "physics": {
            "dt_s": 1.0 / 240.0,
            "environment_rate_hz": 60.0,
            "render_interval": 4,
            "scene_ccd_enabled": True,
            "scene_ccd_attribute": "enable_ccd",
        },
        "capsule": {
            "prim_path": "/World/envs/env_0/Scene/MagneticDemo/target_magnet",
            "shape": "Capsule",
            "axis": "Z",
            "radius_m": 0.0065,
            "cylinder_height_m": 0.012,
            "total_length_m": 0.025,
            "world_scale": [1.0, 1.0, 1.0],
            "collision_enabled": True,
            "kinematic_enabled": False,
            "gravity_enabled": True,
            "ccd_enabled": True,
            "mass_kg": 0.005735,
            "inertia_kg_m2": [1.0e-7, 1.0e-7, 1.0e-7],
            "center_of_mass_m": [0.0, 0.0, 0.0],
        },
        "stomach": {
            "prim_path": "/World/envs/env_0/Stomach/Physics_Collision_Mesh/Stomach",
            "collision_enabled": True,
            "static": True,
            "geometry_sha256": "8" * 64,
            "vertex_count": 24529,
            "triangle_count": 49047,
            "boundary_edge_count": 21,
        },
        "contact_sensor": {"present": True, "name": "capsule_contact"},
        "runtime_contract": {
            "scanned_files": ["mdp/dynamic_force_action.py"],
            "forbidden_calls": [],
            "force_at_center_of_mass": True,
            "commanded_torque_zero": True,
            "magnetic_or_ideal_terms": [],
        },
        "gate": {"status": "pass", "failures": []},
    }

    def factory():
        return deepcopy(template)

    return factory
