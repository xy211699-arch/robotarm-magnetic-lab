from pathlib import Path
import subprocess

import gymnasium as gym
import numpy as np

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_env_cfg import (
    NEW_STOMACH_ASSET_USD_PATH,
    NEW_STOMACH_GEOMETRY,
    NewStomachGeometryConfig,
    RobotarmMagneticNewStomachLabEnvCfg,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_stomach_env_cfg import (
    RobotarmMagneticStomachLabEnvCfg,
)


ROOT = Path(__file__).resolve().parents[2]


def test_new_environment_is_isolated_and_preserves_controller_contract():
    new_cfg = RobotarmMagneticNewStomachLabEnvCfg()
    old_cfg = RobotarmMagneticStomachLabEnvCfg()

    assert isinstance(new_cfg, RobotarmMagneticStomachLabEnvCfg)
    assert Path(NEW_STOMACH_ASSET_USD_PATH).resolve().is_relative_to(ROOT)
    assert new_cfg.scene.stomach.spawn.usd_path == NEW_STOMACH_ASSET_USD_PATH
    assert new_cfg.scene.stomach.spawn.scale == (1.0, 1.0, 1.0)
    assert tuple(new_cfg.scene.stomach.init_state.pos) == NEW_STOMACH_GEOMETRY.position_world_m
    assert tuple(new_cfg.scene.stomach.init_state.rot) == NEW_STOMACH_GEOMETRY.rotation_wxyz
    assert new_cfg.scene.capsule_camera.update_period == old_cfg.scene.capsule_camera.update_period
    assert new_cfg.sim.dt == old_cfg.sim.dt
    assert new_cfg.decimation == old_cfg.decimation
    assert type(new_cfg.actions) is type(old_cfg.actions)


def test_orientation_config_validates_horizontal_axes_and_unity_scale():
    path = ROOT / "configs" / "new_stomach_v1" / "orientation_v1.json"
    geometry = NewStomachGeometryConfig.from_json(path)

    assert geometry.scale == (1.0, 1.0, 1.0)
    assert geometry.collision_mesh_suffix == "/geometry/mesh"
    assert len(geometry.opening_axes_world) == 2
    for axis in geometry.opening_axes_world:
        assert np.degrees(np.arcsin(abs(axis[2]))) <= 1.0
    np.testing.assert_allclose(np.linalg.norm(geometry.rotation_xyzw), 1.0, atol=1.0e-12)


def test_new_task_registration_does_not_replace_legacy_task():
    legacy = gym.spec("Template-Robotarm-Magnetic-Stomach-Lab-v0")
    migrated = gym.spec("Template-Robotarm-Magnetic-New-Stomach-Lab-v0")

    assert legacy.kwargs["env_cfg_entry_point"].endswith(
        "robotarm_magnetic_stomach_env_cfg:RobotarmMagneticStomachLabEnvCfg"
    )
    assert migrated.kwargs["env_cfg_entry_point"].endswith(
        "robotarm_magnetic_new_stomach_env_cfg:RobotarmMagneticNewStomachLabEnvCfg"
    )


def test_project_launcher_imports_the_linked_worktree_package():
    command = (
        "import pathlib,robotarm_magnetic_lab;"
        "print(pathlib.Path(robotarm_magnetic_lab.__file__).resolve())"
    )
    result = subprocess.run(
        [str(ROOT / "run_isaaclab.sh"), "-p", "-c", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected = ROOT / "source" / "robotarm_magnetic_lab" / "robotarm_magnetic_lab" / "__init__.py"
    assert str(expected) in result.stdout
