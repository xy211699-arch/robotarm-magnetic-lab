import numpy as np
from types import SimpleNamespace

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


class _FakeWorldCollisionChecker:
    required_clearance_m = 0.005

    def __init__(self, live_clearance=0.02, path_clearance=0.02):
        self.live_clearance = live_clearance
        self.path_clearance = path_clearance

    def check_configuration(self, _configuration):
        return SimpleNamespace(
            clearance_m=self.live_clearance,
            frame="l4",
            sphere_index=3,
            face_index=17,
        )

    def validate_path(self, _targets):
        return {
            "ok": self.path_clearance >= self.required_clearance_m,
            "kind": "ENVIRONMENT_COLLISION",
            "minimum_world_clearance_m": self.path_clearance,
            "frame": "l6",
        }


def _plan(snapshot):
    command = action_layer.initial_command_state(snapshot, np.array([0.0, 0.0, 1.0]))
    return action_layer.TrajectoryPlan(
        request=action_layer.ActionRequest(1, action_layer.AtomicAction.HOLD, 0.0),
        joint_targets_rad=np.tile(snapshot.joint_position_rad, (3, 1)),
        magnet_targets_world_m=np.tile(snapshot.magnet_position_world_m, (3, 1)),
        duration_s=0.1,
        final_command_state=command,
    )


def test_live_stomach_clearance_is_a_hard_environment_failure():
    monitor = action_layer.HardSafetyMonitor(
        action_layer.ActionLayerConfig(),
        world_collision_checker=_FakeWorldCollisionChecker(live_clearance=0.004),
    )
    check = monitor.check_snapshot(_snapshot())
    assert not check.ok
    assert check.code is action_layer.HardFailureCode.ENVIRONMENT_COLLISION
    assert "frame=l4" in check.detail


def test_swept_stomach_clearance_rejects_plan_before_execution():
    snapshot = _snapshot()
    monitor = action_layer.HardSafetyMonitor(
        action_layer.ActionLayerConfig(),
        world_collision_checker=_FakeWorldCollisionChecker(path_clearance=0.004),
    )
    check = monitor.check_plan(_plan(snapshot), snapshot)
    assert not check.ok
    assert check.code is action_layer.HardFailureCode.ENVIRONMENT_COLLISION
    assert "minimum_world_clearance_m" in check.detail
