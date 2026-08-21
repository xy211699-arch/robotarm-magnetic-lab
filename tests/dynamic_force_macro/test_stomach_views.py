from pathlib import Path


def test_stomach_launcher_contains_required_three_views_and_normal_gate():
    source = (Path(__file__).resolve().parents[2] / "scripts/dynamic_force_macro/teleop_stomach.py").read_text()
    assert "configure_capsule_camera_view" in source
    assert "enable_view=not HEADLESS" in source
    assert "require_camera_facing_normal=True" in source
    assert "StatusPanel" in source
    assert "simulation_app.update()" in source


def test_stomach_launcher_uses_selected_profile_without_tuning():
    source = (Path(__file__).resolve().parents[2] / "scripts/dynamic_force_macro/teleop_stomach.py").read_text()
    assert "selected_profile.json" in source
    assert "move_force_ratio" in source
    assert "view_force_ratio" in source
    assert "up_force_ratio" in source
    assert "search_group" not in source
    assert "coarse_candidates" not in source
