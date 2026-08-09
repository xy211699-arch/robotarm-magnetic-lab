"""Contract tests for the P0 prerequisite inspector report."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "action_layer"
    / "inspect_p0_coverage_prerequisites.py"
)


def _load_inspector():
    spec = importlib.util.spec_from_file_location("p0_prerequisite_inspector", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prerequisite_report_requires_contract_fields():
    inspector = _load_inspector()
    report = {
        "schema_version": 1,
        "repository": {"commit": "abc", "branch": "feature"},
        "dependencies": {"python": "3.12", "torch": "2", "warp": "1"},
        "atomic_task": {
            "environment_id": "Template-Robotarm-Magnetic-Atomic-Table-Lab-v0",
            "registered": True,
            "action_ids": {"HOLD": 0},
        },
        "camera": {
            "prim_path": "/World/camera",
            "update_period_s": 1.0,
            "offset_position_m": [0.0, 0.0, -0.0127],
            "offset_quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
            "convention": "ros",
            "optical_transform_confirmed": True,
        },
        "stomach": {
            "root_prim": "/World/Stomach",
            "meshes": [
                {
                    "prim_path": "/World/Stomach/Inner",
                    "vertex_count": 3,
                    "face_count": 1,
                    "topology": {"face_vertex_counts": {"3": 1}},
                    "world_transform": [1.0] * 16,
                    "world_bounds_m": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                    "purpose": "default",
                    "visibility": "inherited",
                    "material_bindings": [],
                    "surface_role": "candidate",
                }
            ],
            "selection_unambiguous": True,
            "selected_inner_surface_prims": ["/World/Stomach/Inner"],
        },
        "gpu_ray_apis": [
            {
                "name": "isaaclab.utils.warp.ops.raycast_mesh",
                "available": True,
                "gpu_batched": True,
                "first_hit": True,
                "face_id": True,
            }
        ],
        "gate": {"status": "pass", "reasons": []},
    }

    inspector.validate_report(report)


def test_prerequisite_report_rejects_missing_mesh_topology():
    inspector = _load_inspector()
    report = inspector.empty_report()
    report["stomach"]["meshes"] = [
        {
            "prim_path": "/World/Stomach/Inner",
            "vertex_count": 3,
            "face_count": 1,
            "world_transform": [1.0] * 16,
            "world_bounds_m": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            "purpose": "default",
            "visibility": "inherited",
            "material_bindings": [],
            "surface_role": "candidate",
        }
    ]

    try:
        inspector.validate_report(report)
    except ValueError as error:
        assert "topology" in str(error)
    else:
        raise AssertionError("missing topology must fail report validation")
