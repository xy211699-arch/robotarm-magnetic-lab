import numpy as np

import action_layer


def _snapshot():
    return action_layer.DeviceSnapshot(
        sim_time_s=0.0,
        joint_position_rad=np.zeros(9),
        joint_velocity_rad_s=np.zeros(9),
        joint_acceleration_rad_s2=np.zeros(9),
        joint_position_limits_rad=np.tile((-1.0, 1.0), (9, 1)),
        joint_velocity_limits_rad_s=np.ones(9),
        joint_acceleration_limits_rad_s2=np.ones(9) * 2.0,
        magnet_position_world_m=np.array([1.0, 0.0, 0.2]),
        magnet_rotation_world=np.eye(3),
        asm_clearance_m=0.01,
    )


def test_nonfinite_state_is_hard_failure():
    snapshot = _snapshot()
    snapshot.joint_position_rad[0] = np.nan
    check = action_layer.HardSafetyMonitor(action_layer.ActionLayerConfig()).check_snapshot(snapshot)
    assert not check.ok
    assert check.code is action_layer.HardFailureCode.NONFINITE_STATE


def test_capsule_effect_deviation_is_not_representable_as_hard_safety_input():
    monitor = action_layer.HardSafetyMonitor(action_layer.ActionLayerConfig())
    assert monitor.check_snapshot(_snapshot()).ok
