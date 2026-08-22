import gymnasium as gym
import pytest

import robotarm_magnetic_lab.tasks  # noqa: F401
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_dynamic_force_macro_table_env_cfg import RobotarmMagneticDynamicForceMacroTableLabEnvCfg
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_dynamic_force_macro_stomach_env_cfg import RobotarmMagneticDynamicForceMacroStomachLabEnvCfg
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_dynamic_force_stomach_env_cfg import RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg


@pytest.mark.parametrize("task", [
    "Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0",
    "Template-Robotarm-Magnetic-Dynamic-Force-Macro-Stomach-Lab-v0",
])
def test_task_registered(task):
    assert gym.spec(task) is not None


@pytest.mark.parametrize("cfg_type", [RobotarmMagneticDynamicForceMacroTableLabEnvCfg, RobotarmMagneticDynamicForceMacroStomachLabEnvCfg])
def test_task_contract(cfg_type):
    cfg = cfg_type()
    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(1 / 240)
    assert cfg.sim.render_interval == 4
    assert cfg.scene.capsule_camera.update_period == pytest.approx(1 / 30)
    assert cfg.sim.device == "cpu"
    assert list(name for name in vars(cfg.actions) if not name.startswith("_")) == ["dynamic_force_macro"]
    assert vars(cfg.rewards) == {}


def test_stomach_uses_confirmed_migrated_force_ratios():
    cfg = RobotarmMagneticDynamicForceMacroStomachLabEnvCfg()
    term = cfg.actions.dynamic_force_macro
    assert term.move_force_ratio == pytest.approx(0.40)
    assert term.view_force_ratio == pytest.approx(0.25)
    assert term.up_force_ratio == pytest.approx(0.85)


def test_task008_starts_at_opposite_longitudinal_quarter_without_changing_task003():
    cfg = RobotarmMagneticDynamicForceMacroStomachLabEnvCfg()
    assert cfg.scene.capsule.init_state.pos == pytest.approx(
        (1.078678615084, 0.177197008592, 0.011793035807)
    )
    y_min = -0.010734736771726316
    y_max = 0.24037395987787247
    expected_right_quarter_y = y_min + 0.75 * (y_max - y_min)
    assert abs(cfg.scene.capsule.init_state.pos[1] - expected_right_quarter_y) < 0.001
    assert cfg.scene.capsule.init_state.pos[2] > 0.01

    task003_cfg = RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg()
    assert task003_cfg.scene.capsule.init_state.pos == pytest.approx(
        (1.04643745, 0.05368121, 0.00190115)
    )


def test_stomach_close_range_lighting_is_low_glare():
    cfg = RobotarmMagneticDynamicForceMacroStomachLabEnvCfg()
    assert cfg.scene.dome_light.spawn.intensity == pytest.approx(5.0)
    for name in ("capsule_led_top", "capsule_led_bottom", "capsule_led_left", "capsule_led_right"):
        light = getattr(cfg.scene, name).spawn
        assert light.intensity == pytest.approx(4.0)
        assert light.radius == pytest.approx(0.0030)
