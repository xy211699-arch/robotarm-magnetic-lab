import gymnasium as gym

import robotarm_magnetic_lab.tasks  # noqa: F401
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_eleven_action_flat_env_cfg import (
    RobotarmMagneticElevenActionFlatLabEnvCfg,
)


TASK_ID = "Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0"


def test_flat_task_is_registered_and_has_only_scalar_eleven_action():
    assert TASK_ID in gym.registry
    cfg = RobotarmMagneticElevenActionFlatLabEnvCfg()
    assert list(vars(cfg.actions)) == ["eleven_action"]
    assert cfg.actions.eleven_action.surface_kind == "flat"
    assert cfg.scene.num_envs == 1
    assert cfg.decimation == 4
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.sim.render_interval == 2
    assert cfg.sim.physics.enable_ccd
    assert cfg.sim.device != "cpu"  # parse_env_cfg/CLI retains authority over device.


def test_policy_observation_remains_only_capsule_rgb():
    cfg = RobotarmMagneticElevenActionFlatLabEnvCfg()
    terms = [name for name, value in vars(cfg.observations.policy).items() if hasattr(value, "func")]
    assert terms == ["rgb"]
