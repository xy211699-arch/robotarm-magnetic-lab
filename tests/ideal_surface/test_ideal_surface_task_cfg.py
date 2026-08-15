from __future__ import annotations

import gymnasium as gym
from isaaclab.managers import ObservationTermCfg

import robotarm_magnetic_lab.tasks  # noqa: F401


ENV_ID = "Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0"


def _cfg_type():
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_ideal_surface_stomach_env_cfg import (
        RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg,
    )

    return RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg


def _term_names(group):
    return [
        name
        for name, value in vars(group).items()
        if not name.startswith("_") and value is not None
    ]


def _observation_term_names(group):
    return [name for name, value in vars(group).items() if isinstance(value, ObservationTermCfg)]


def test_task_is_registered_at_one_hertz_with_one_scalar_action():
    spec = gym.spec(ENV_ID)
    assert "RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg" in spec.kwargs["env_cfg_entry_point"]
    cfg = _cfg_type()()
    assert cfg.scene.num_envs == 1
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.decimation == 240
    assert cfg.scene.capsule_camera.update_period == 1.0
    assert _term_names(cfg.actions) == ["ideal_surface"]
    assert cfg.actions.ideal_surface.class_type.action_dim.fget is not None


def test_task_does_not_enable_magnetic_capsule_forcing():
    cfg = _cfg_type()()
    assert "magnetic_physics" not in _term_names(cfg.actions)
    assert "magnetic_collision_bridge" not in _term_names(cfg.events)
    assert _term_names(cfg.events) == ["reset_scene"]


def test_policy_observations_contain_no_capsule_or_surface_truth():
    cfg = _cfg_type()()
    names = _observation_term_names(cfg.observations.policy)
    assert names == ["rgb"]
    assert not any(
        token in name.lower()
        for name in names
        for token in ("capsule", "surface", "contact", "coverage", "pose", "ray")
    )


def test_only_timeout_and_ideal_hard_failure_terminate_episode():
    cfg = _cfg_type()()
    assert set(_term_names(cfg.terminations)) == {"time_out", "ideal_surface_hard_failure"}
