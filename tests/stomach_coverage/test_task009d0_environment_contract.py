from __future__ import annotations

import inspect
from pathlib import Path

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_stomach_env_cfg import (
    STOMACH_ASSET_USD_PATH,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_task009d0_env_cfg import (
    RobotarmMagneticTask009D0EnvCfg,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.task009d0_vector_env import (
    FORMAL_STEPS,
    RESET_HOLD_CYCLES,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = (
    ROOT
    / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/"
    "robotarm_magnetic_lab/__init__.py"
)


def test_new_task_is_separate_and_old_task_registration_is_unchanged():
    source = REGISTRY.read_text(encoding="utf-8")
    assert "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0" in source
    assert "Template-Robotarm-Magnetic-Task009D0-Vector-Coverage-Lab-v0" in source
    assert "Task009BTrainingEnv" in source
    assert "Task009D0VectorEnv" in source


def test_task009d0_cfg_preserves_physics_camera_and_assets():
    cfg = RobotarmMagneticTask009D0EnvCfg()
    assert cfg.sim.dt == 1 / 240
    assert cfg.decimation == 24
    assert cfg.scene.capsule_camera.width == 1280
    assert cfg.scene.capsule_camera.height == 720
    assert cfg.scene.capsule_camera.update_period == 0.1
    assert cfg.scene.stomach.spawn.usd_path == STOMACH_ASSET_USD_PATH
    assert cfg.scene.env_spacing == 4.0


def test_training_mode_rejects_explicit_validation_pose_ids():
    cfg = RobotarmMagneticTask009D0EnvCfg()
    assert cfg.pose_split == "train"
    assert cfg.explicit_pose_ids is None


def test_actor_group_has_only_rgb_and_previous_action():
    cfg = RobotarmMagneticTask009D0EnvCfg()
    names = {
        key
        for key, value in vars(cfg.observations.policy).items()
        if value is not None and not key.startswith("_")
    }
    assert {"rgb", "previous_action"}.issubset(names)
    source = inspect.getsource(type(cfg.observations.policy)).lower()
    for forbidden in ("pose", "velocity", "contact", "coverage", "pose_id", "split"):
        assert forbidden not in source
    assert hasattr(cfg.observations, "privileged")


def test_synchronous_constants_are_exact():
    assert FORMAL_STEPS == 1200
    assert RESET_HOLD_CYCLES == 10
