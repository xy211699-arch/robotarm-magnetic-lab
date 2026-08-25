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

