import math

import numpy as np

import action_layer


CFG = action_layer.ActionLayerConfig(
    workspace_min_world_m=(-2.0, -2.0, -2.0),
    workspace_max_world_m=(2.0, 2.0, 2.0),
)


def snapshot(q=None, t=0.0):
    if q is None:
        q = np.zeros(9)
    return action_layer.DeviceSnapshot(
        sim_time_s=t,
        joint_position_rad=q,
        joint_velocity_rad_s=np.zeros(9),
        joint_acceleration_rad_s2=np.zeros(9),
        joint_position_limits_rad=np.tile((-4.0, 4.0), (9, 1)),
        joint_velocity_limits_rad_s=np.full(9, 2.0),
        joint_acceleration_limits_rad_s2=np.full(9, 4.0),
        magnet_position_world_m=np.array([1.0, 0.0, 0.2]),
        magnet_rotation_world=np.eye(3),
        asm_clearance_m=0.02,
    )


def executor():
    snap = snapshot()
    command = action_layer.initial_command_state(snap, np.array([1.0, 0.0, 0.0]))

    def ball_solver(desired, current):
        return current + np.array([0.01, 0.0, 0.0]), {"test": True}

    def arm_solver(_snapshot, displacement):
        result = np.zeros(6)
        result[:3] = displacement
        return result

    planner = action_layer.AtomicCommandPlanner(
        CFG,
        solve_ball_field=ball_solver,
        solve_arm_displacement=arm_solver,
        field_for_ball=lambda _q: np.array([1.0, 0.0, 0.0]),
    )
    return action_layer.AtomicActionExecutor(
        CFG, planner, action_layer.HardSafetyMonitor(CFG), command
    )


def test_busy_request_cannot_preempt_current_action():
    engine = executor()
    snap = snapshot()
    assert engine.submit(action_layer.AtomicAction.HOLD, snap, 1)
    engine.step(snap)
    assert engine.busy
    assert not engine.submit(action_layer.AtomicAction.TILT_POS, snap, 2)
    assert engine.last_rejected_code is action_layer.HardFailureCode.BUSY
    assert engine.request.request_id == 1


def test_hold_reaches_done_without_capsule_state():
    engine = executor()
    snap = snapshot()
    assert engine.submit(action_layer.AtomicAction.HOLD, snap, 1)
    for index in range(40):
        snap = snapshot(engine.last_safe_target.copy(), (index + 1) * CFG.control_dt_s)
        step = engine.step(snap)
        if step.state is action_layer.ExecutionState.DONE:
            break
    assert engine.state is action_layer.ExecutionState.DONE
    assert engine.last_result.status is action_layer.ActionStatus.DONE
    assert engine.last_result.to_dict()["action_id"] == 0


def test_invalid_id_is_hard_failure():
    engine = executor()
    snap = snapshot()
    assert not engine.submit(99, snap, 7)
    assert engine.last_result.status is action_layer.ActionStatus.HARD_FAILURE
    assert engine.last_result.hard_failure_code is action_layer.HardFailureCode.INVALID_ACTION


def test_mask_is_command_based_and_roll_is_available_at_horizontal_field():
    engine = executor()
    mask = engine.action_mask(snapshot())
    assert mask.allows(action_layer.AtomicAction.ROLL_POS)
    assert mask.allows(action_layer.AtomicAction.AZIMUTH_POS)


def test_tilt_command_updates_by_fixed_increment():
    engine = executor()
    snap = snapshot()
    request = action_layer.ActionRequest(1, action_layer.AtomicAction.TILT_POS, 0.0)
    plan = engine.planner.plan(request, snap, engine.command_state)
    assert math.isclose(
        plan.final_command_state.theta_rad,
        engine.command_state.theta_rad + CFG.tilt_increment_rad,
    )
