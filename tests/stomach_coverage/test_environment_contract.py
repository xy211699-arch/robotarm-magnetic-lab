from pathlib import Path

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    CONTROL_HZ,
    PHYSICS_HZ,
    PHYSICS_STEPS_PER_CONTROL,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.parameterized_force_action import (
    ParameterizedForceActionTermCfg,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_parameterized_force_stomach_env_cfg import (
    RobotarmMagneticParameterizedForceStomachCoverageLabEnvCfg,
)


ROOT = Path(__file__).resolve().parents[2]


def test_stomach_environment_uses_audited_parameterized_force_term():
    cfg = RobotarmMagneticParameterizedForceStomachCoverageLabEnvCfg()
    assert isinstance(cfg.actions.parameterized_force, ParameterizedForceActionTermCfg)
    assert not hasattr(cfg.actions, "dynamic_force_macro")
    assert cfg.sim.dt == 1.0 / PHYSICS_HZ
    assert cfg.decimation == PHYSICS_STEPS_PER_CONTROL
    assert 1.0 / (cfg.sim.dt * cfg.decimation) == CONTROL_HZ
    assert cfg.scene.capsule_camera.update_period == 1.0 / CONTROL_HZ
    assert cfg.sim.render_interval == PHYSICS_HZ // 60


def test_stomach_environment_does_not_import_one_second_macro_controller():
    source = (
        ROOT
        / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/"
        "robotarm_magnetic_lab/robotarm_magnetic_parameterized_force_stomach_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "dynamic_force_macro" not in source
    assert "ParameterizedForceActionTermCfg" in source


def test_integration_validator_requires_six_mode_boundary_evidence():
    source = (
        ROOT / "scripts/stomach_coverage/validate_environment_integration.py"
    ).read_text(encoding="utf-8")
    for mode in ("HOLD", "MOVE_POS", "MOVE_NEG", "VIEW_POS", "VIEW_NEG", "UP"):
        assert f"ParameterizedForceMode.{mode}" in source
    assert 'keys != ["policy.rgb"]' in source
    assert "end_frame - start_frame != 1" in source


def test_coverage_validator_is_bound_to_frozen_pose_library_and_70mm_area_contract():
    source = (
        ROOT / "scripts/stomach_coverage/validate_coverage_calculation.py"
    ).read_text(encoding="utf-8")
    assert 'MAX_OBSERVATION_DISTANCE_M, 0.07' in source
    assert 'pose_manifest.get("live_reload_validation", {}).get("status") != "pass"' in source
    assert 'selected_ids = pose_manifest["fixed_live_reload_pose_ids"]' in source
    assert "cfg.episode_length_s = 100.0" in source
    assert "camera._update_buffers_impl(camera._ALL_ENV_MASK)" in source
    assert "target_vertex_area_weights(reference)" in source
    assert "visible_from_first_hits" in source
    assert "camera_facing_first_hits" in source
    assert '"post_reset_initial_C0"' in source


def test_gate5_uses_one_recorded_camera_and_exact_control_boundaries():
    source = (
        ROOT / "scripts/stomach_coverage/teleop_stomach_coverage.py"
    ).read_text(encoding="utf-8")
    assert "configure_capsule_recorded_camera_view(cfg)" in source
    assert "attach_capsule_recorded_camera_view(env)" in source
    assert "configure_capsule_camera_view" not in source
    assert "force_boundary_capture=True" in source
    assert "camera_facing_normal_sign=-1" in source
    assert "PHYSICS_STEPS_PER_CONTROL" in source
    assert "TASK009B_THREE_VIEW_READY" in source
