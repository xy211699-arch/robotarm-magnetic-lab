from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/local_primitives/controller.py"


def test_controller_has_no_environment_geometry_or_state_write_dependencies():
    source = RUNTIME.read_text(encoding="utf-8").lower()
    forbidden = (
        "surface_mesh", "clearance", "nearest_triangle", "raycast", "swept",
        "local_normal", "tangent_frame", "project_to_surface", "write_root",
        "set_root", "set_world_pose", "set_linear_velocity", "set_angular_velocity",
        "magnetic_action", "ideal_surface", "legacy_bridge",
    )
    assert [token for token in forbidden if token in source] == []
