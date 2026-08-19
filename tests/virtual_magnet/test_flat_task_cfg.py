"""Configuration-level invariants for the TASK-007 flat environment."""

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_virtual_magnet_flat_env_cfg import (
    RobotarmMagneticVirtualMagnetFlatLabEnvCfg,
)


def test_task_has_exact_rates_and_no_joint_action_term():
    cfg = RobotarmMagneticVirtualMagnetFlatLabEnvCfg()
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.decimation == 240
    assert set(vars(cfg.actions)) >= {"request", "magnetic_physics"}
    assert not hasattr(cfg.actions, "joint_position")
    assert cfg.actions.request.class_type.__name__ == "VirtualMagnetRequestAction"
    assert cfg.actions.magnetic_physics.class_type.__name__ == "VirtualMagnetPhysicsAction"


def test_capsule_is_registered_as_rigid_object_and_bridge_owns_it():
    cfg = RobotarmMagneticVirtualMagnetFlatLabEnvCfg()
    assert cfg.scene.capsule.prim_path.endswith("/target_magnet")
    params = cfg.events.virtual_magnet_bridge.params
    assert params["asset_name"] == "capsule"
    assert params["contact_sensor_name"] == "capsule_contact"
    assert params["debug_xform"] is True


def test_policy_observation_excludes_privileged_physics_truth():
    cfg = RobotarmMagneticVirtualMagnetFlatLabEnvCfg()
    assert set(vars(cfg.observations.policy)) >= {"action_status"}
    forbidden = {"capsule_pose", "contact", "wrench", "magnet_pose"}
    assert forbidden.isdisjoint(vars(cfg.observations.policy))

