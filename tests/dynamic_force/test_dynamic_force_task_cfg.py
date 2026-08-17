"""Registration, timing, and actuator-isolation tests for TASK-003."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from isaaclab.managers import ObservationTermCfg
from isaaclab_physx.physics import PhysxCfg

import robotarm_magnetic_lab.tasks  # noqa: F401


ENV_ID = "Template-Robotarm-Magnetic-Dynamic-Force-Stomach-Teleop-Lab-v0"


def _cfg_type():
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_dynamic_force_stomach_env_cfg import (
        RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg,
    )

    return RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg


def _term_names(group):
    return [
        name
        for name, value in vars(group).items()
        if not name.startswith("_") and value is not None
    ]


def test_dynamic_force_task_has_frozen_rates_and_action():
    spec = gym.spec(ENV_ID)
    assert "RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg" in spec.kwargs["env_cfg_entry_point"]
    cfg = _cfg_type()()
    assert cfg.scene.num_envs == 1
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.decimation == 4
    assert cfg.sim.render_interval == 4
    assert cfg.scene.capsule_camera.update_period == 1.0 / 30.0
    assert cfg.sim.device == "cpu"
    assert isinstance(cfg.sim.physics, PhysxCfg)
    assert cfg.sim.physics.enable_ccd is True
    assert _term_names(cfg.actions) == ["dynamic_force"]
    assert cfg.actions.dynamic_force.force_weight_ratio == 0.9
    assert cfg.actions.dynamic_force.vertical_force_weight_ratio == 1.1


def test_dynamic_force_task_flips_only_its_stomach_and_starts_capsule_sideways():
    cfg = _cfg_type()()
    assert cfg.scene.stomach.init_state.pos == (2.091607300678508, 0.0, 0.06585919085865222)
    assert cfg.scene.stomach.init_state.rot == (0.0, 1.0, 0.0, 0.0)
    assert cfg.scene.capsule.init_state.pos == (1.04643745, 0.05368121, 0.00190115)
    assert cfg.scene.capsule.init_state.rot == (
        0.73137642,
        0.07274179,
        0.67346616,
        0.07899674,
    )

    # XYZW quaternion rotates the capsule's local Z/long axis almost into the
    # world XY plane: the reset is side-lying rather than upright.
    x, y, z, w = cfg.scene.capsule.init_state.rot
    capsule_axis_world = (
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    )
    assert abs(capsule_axis_world[2]) < 0.1

    # The selected wall point is the left quarter along the stomach's world-Y
    # longitudinal extent; the center differs only by its surface-normal lift.
    expected_quarter_y = -0.010734736771726316 + 0.25 * (
        0.24037395987787247 - -0.010734736771726316
    )
    assert abs(cfg.scene.capsule.init_state.pos[1] - expected_quarter_y) < 0.002


def test_task_contains_no_forbidden_actuator_or_event():
    cfg = _cfg_type()()
    assert "magnetic_physics" not in _term_names(cfg.actions)
    assert "ideal_surface" not in _term_names(cfg.actions)
    assert "joint_position" not in _term_names(cfg.actions)
    assert _term_names(cfg.events) == ["reset_scene"]
    assert "magnetic_collision_bridge" not in _term_names(cfg.events)


def test_task_has_rgb_only_no_rewards_and_timeout_only():
    cfg = _cfg_type()()
    observation_names = [
        name
        for name, value in vars(cfg.observations.policy).items()
        if isinstance(value, ObservationTermCfg)
    ]
    assert observation_names == ["rgb"]
    assert _term_names(cfg.rewards) == []
    assert _term_names(cfg.terminations) == ["time_out"]


def test_new_action_runtime_contains_no_capsule_state_setter():
    source = (
        Path(__file__).resolve().parents[2]
        / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/"
        "robotarm_magnetic_lab/mdp/dynamic_force_action.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "write_root_pose",
        "write_root_velocity",
        "set_transforms",
        "set_velocities",
    ):
        assert forbidden not in source
