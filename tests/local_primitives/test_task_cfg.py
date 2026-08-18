import gymnasium as gym

import robotarm_magnetic_lab.tasks  # noqa: F401
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_local_primitives_flat_env_cfg import (
    RobotarmMagneticLocalPrimitivesFlatLabEnvCfg,
)


FLAT_ID = "Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0"
STOMACH_ID = "Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0"


def term_names(cfg):
    return [name for name, value in vars(cfg).items() if value is not None and not name.startswith("_")]


def test_flat_task_is_registered_isolated_and_uses_frozen_rates():
    spec = gym.spec(FLAT_ID)
    cfg = RobotarmMagneticLocalPrimitivesFlatLabEnvCfg()
    assert "LocalPrimitivesFlat" in spec.kwargs["env_cfg_entry_point"]
    assert cfg.scene.num_envs == 1
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.decimation == 4
    assert cfg.sim.render_interval == 4
    assert cfg.scene.capsule_camera.update_period == 1.0 / 30.0
    assert cfg.sim.device == "cpu"
    assert term_names(cfg.actions) == ["local_primitive"]
    assert term_names(cfg.events) == ["reset_scene"]
    assert term_names(cfg.rewards) == []
    assert term_names(cfg.terminations) == ["time_out"]


def test_stomach_task_is_registered_with_only_local_primitive_action():
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_local_primitives_stomach_env_cfg import (
        RobotarmMagneticLocalPrimitivesStomachLabEnvCfg,
    )

    spec = gym.spec(STOMACH_ID)
    cfg = RobotarmMagneticLocalPrimitivesStomachLabEnvCfg()
    assert "LocalPrimitivesStomach" in spec.kwargs["env_cfg_entry_point"]
    assert term_names(cfg.actions) == ["local_primitive"]
    assert cfg.scene.num_envs == 1
    assert cfg.sim.device == "cpu"
