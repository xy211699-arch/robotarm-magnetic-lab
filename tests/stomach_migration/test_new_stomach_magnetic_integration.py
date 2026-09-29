"""Static contract checks for the isolated new-stomach magnetic task."""

from __future__ import annotations

import hashlib
from pathlib import Path

import gymnasium as gym

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.actuator_vector_action import (
    ActuatorVectorAction,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.new_stomach_actuator_action import (
    NewStomachActuatorVectorAction,
    audited_new_stomach_collision,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_env_cfg import (
    NEW_STOMACH_ASSET_USD_PATH,
    NEW_STOMACH_GEOMETRY,
    RobotarmMagneticNewStomachLabEnvCfg,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_vector_env_cfg import (
    NEW_STOMACH_COLLIDER,
    RobotarmMagneticNewStomachVectorEnvCfg,
)


def test_magnetic_task_reuses_existing_controller_and_one_second_boundary():
    cfg = RobotarmMagneticNewStomachVectorEnvCfg()
    assert issubclass(NewStomachActuatorVectorAction, ActuatorVectorAction)
    assert cfg.actions.magnet.class_type is NewStomachActuatorVectorAction
    assert cfg.actions.magnet.world_mesh_path == NEW_STOMACH_COLLIDER
    assert cfg.actions.magnetic_physics.asset_name == "capsule"
    assert cfg.decimation == 240
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.sim.render_interval == 240
    assert cfg.terminations.collision.func is audited_new_stomach_collision


def test_visual_zero_agent_task_retains_absolute_joint_action():
    cfg = RobotarmMagneticNewStomachLabEnvCfg()
    assert hasattr(cfg.actions, "joint_position")
    assert not hasattr(cfg.actions, "magnet")
    assert cfg.decimation == 12


def test_confirmed_asset_and_collider_identity():
    assert hashlib.sha256(Path(NEW_STOMACH_ASSET_USD_PATH).read_bytes()).hexdigest() == (
        NEW_STOMACH_GEOMETRY.asset_usd_sha256
    )
    assert NEW_STOMACH_GEOMETRY.confirmed
    assert NEW_STOMACH_COLLIDER.endswith("/geometry/CollisionMesh")


def test_vector_task_has_separate_gym_id():
    visual = gym.spec("Template-Robotarm-Magnetic-New-Stomach-Lab-v0")
    vector = gym.spec("Template-Robotarm-Magnetic-New-Stomach-Vector-Lab-v0")
    assert visual.id != vector.id
    assert "new_stomach_vector_env_cfg" in vector.kwargs["env_cfg_entry_point"]
