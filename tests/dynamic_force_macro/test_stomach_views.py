from pathlib import Path


def test_stomach_launcher_contains_required_three_views_and_normal_gate():
    source = (Path(__file__).resolve().parents[2] / "scripts/dynamic_force_macro/teleop_stomach.py").read_text()
    assert "configure_capsule_camera_view" in source
    assert "enable_view=not HEADLESS" in source
    assert "require_camera_facing_normal=True" in source
    assert "StatusPanel" not in source
    assert "omni.ui.Window" not in source
    assert "TASK008_STOMACH_FORCE_CONFIG" not in source
    assert "simulation_app.update()" in source


def test_stomach_launcher_uses_confirmed_migrated_force_values_without_tuning():
    source = (Path(__file__).resolve().parents[2] / "scripts/dynamic_force_macro/teleop_stomach.py").read_text()
    assert 'default=0.40' in source
    assert 'default=0.25' in source
    assert 'default=0.85' in source
    assert 'f"力度：MOVE=' not in source
    assert "TASK008_COVERAGE_VIEW_READY" in source
    assert "coverage.update_view()" in source
    assert "search_group" not in source
    assert "coarse_candidates" not in source


def test_stomach_material_uses_broadened_low_glare_highlight():
    source = (
        Path(__file__).resolve().parents[2] / "assets/stomach/stomach_environment_lab.usda"
    ).read_text(encoding="utf-8")
    assert "float inputs:roughness = 0.78" in source
    assert "float inputs:ior = 1.34" in source
