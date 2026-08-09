"""Configuration isolation for the dedicated atomic stomach teleop task."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))
for module_name in tuple(sys.modules):
    if module_name == "robotarm_magnetic_lab" or module_name.startswith("robotarm_magnetic_lab."):
        del sys.modules[module_name]

import robotarm_magnetic_lab.tasks  # noqa: F401
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_atomic_stomach_teleop_env_cfg import (
    RobotarmMagneticAtomicStomachTeleopLabEnvCfg,
)


ENV_ID = "Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0"


def test_task_registration_and_frozen_timing_contract():
    spec = gym.spec(ENV_ID)
    assert "RobotarmMagneticAtomicStomachTeleopLabEnvCfg" in spec.kwargs["env_cfg_entry_point"]
    cfg = RobotarmMagneticAtomicStomachTeleopLabEnvCfg()
    assert cfg.scene.num_envs == 1
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.decimation == 12
    assert cfg.scene.capsule_camera.update_period == 1.0
    assert set(vars(cfg.actions)) == {"atomic", "magnetic_physics"}
    assert cfg.actions.atomic.environment_collision_mesh_prim_path.endswith(
        "/Physics_Collision_Mesh/Stomach"
    )
    assert cfg.actions.atomic.environment_collision_clearance_m == 0.005


def test_deployable_observations_exclude_privileged_fields():
    cfg = RobotarmMagneticAtomicStomachTeleopLabEnvCfg()
    policy_names = {
        name
        for name, value in vars(cfg.observations.policy).items()
        if value is not None and value.__class__.__name__ == "ObservationTermCfg"
    }
    assert policy_names == {"joint_pos", "joint_vel", "external_magnet"}
    forbidden = ("capsule", "coverage", "contact", "depth", "stomach", "ray", "wrench")
    assert not any(token in name.lower() for name in policy_names for token in forbidden)
