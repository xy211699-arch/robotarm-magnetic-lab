"""Static contract checks for the interactive TASK-009B replacement gate."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stomach_coverage" / "calibrate_entry_anchor_region.py"


def test_calibrator_preserves_dynamic_body_and_uses_parameterized_force_positioning():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "write_root_pose_to_sim_index" in source
    assert "sim.physics_manager.pause()" in source
    assert "sim.physics_manager.get_simulation_time()" in source
    assert "kinematic capsule is forbidden" in source
    assert "CreateStatic" not in source
    assert "CreateKinematicEnabledAttr" not in source
    assert 'getattr(raw_input, "name", raw_input)' in source
    assert "ParameterizedForceKeyboard(alpha=0.5)" in source
    assert 'phase = "DYNAMIC_CONTROL"' in source
    assert "base.action_manager.process_action(action)" in source
    assert "base.action_manager.apply_action()" in source
    assert "range(PHYSICS_STEPS_PER_CONTROL)" in source


def test_calibrator_preserves_force_clock_and_stability_contract():
    source = SCRIPT.read_text(encoding="utf-8")
    for declaration in (
        "PHYSICS_DT_S = 1.0 / PHYSICS_HZ",
        "STABLE_STEPS = 60",
        "MAX_SETTLE_STEPS = 480",
        "MAX_LINEAR_SPEED_M_S = 0.002",
        "MAX_ANGULAR_SPEED_RAD_S = np.deg2rad(5.0)",
    ):
        assert declaration in source
    assert "term.reset()" in source
    assert "capsule.permanent_wrench_composer.reset()" in source
    assert 'mode, alpha = ParameterizedForceMode.HOLD, keyboard.force.alpha' in source
    assert "continuous_stable_steps + 1" in source
    assert "initial_velocity_zero=False" in source


def test_calibrator_uses_true_surface_and_connected_geodesic_artifacts():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "nearest_surface_point(reference, settled_pose[:3])" in source
    assert "shared_edge_adjacency(reference)" in source
    assert "geodesic_face_distances" in source
    assert '"entry_anchor_v1.json"' in source
    assert '"entry_region_v1.json"' in source
    assert "region.connected_components != 1" in source
    assert '"--resume_anchor"' in source
    assert '"--initial_radius_mm"' in source
    assert "choices=tuple(range(10, 81, 5))" in source
