from pathlib import Path

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    simulation_profile_sha256,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_dynamic_force_stomach_env_cfg import (
    RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_local_primitives_flat_env_cfg import (
    RobotarmMagneticLocalPrimitivesFlatLabEnvCfg,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_local_primitives_stomach_env_cfg import (
    RobotarmMagneticLocalPrimitivesStomachLabEnvCfg,
)


def test_flat_and_stomach_share_exact_profile():
    flat = RobotarmMagneticLocalPrimitivesFlatLabEnvCfg().actions.local_primitive
    stomach = RobotarmMagneticLocalPrimitivesStomachLabEnvCfg().actions.local_primitive
    assert flat.controller_cfg_values == stomach.controller_cfg_values
    assert flat.profile_sha256 == stomach.profile_sha256 == simulation_profile_sha256()


def test_stomach_preserves_task003_reset_and_timing():
    task003 = RobotarmMagneticDynamicForceStomachTeleopLabEnvCfg()
    stomach = RobotarmMagneticLocalPrimitivesStomachLabEnvCfg()
    assert stomach.scene.stomach.init_state.pos == task003.scene.stomach.init_state.pos
    assert stomach.scene.stomach.init_state.rot == task003.scene.stomach.init_state.rot
    assert stomach.scene.capsule.init_state.pos == task003.scene.capsule.init_state.pos
    assert stomach.scene.capsule.init_state.rot == task003.scene.capsule.init_state.rot
    assert stomach.sim.dt == task003.sim.dt == 1.0 / 240.0
    assert stomach.decimation == task003.decimation == 4
    assert stomach.sim.render_interval == task003.sim.render_interval == 4
    assert stomach.scene.capsule_camera.update_period == task003.scene.capsule_camera.update_period
    assert stomach.sim.physics.enable_ccd is True


def test_stomach_wrapper_contains_no_scene_or_controller_adaptation():
    path = Path(
        "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/"
        "robotarm_magnetic_lab/robotarm_magnetic_local_primitives_stomach_env_cfg.py"
    )
    source = path.read_text(encoding="utf-8")
    forbidden = (
        "__post_init__", "surface", "normal", "clearance", "raycast", "gain",
        "init_state", "write_root", "set_transforms", "set_velocities",
    )
    assert all(word not in source for word in forbidden)
