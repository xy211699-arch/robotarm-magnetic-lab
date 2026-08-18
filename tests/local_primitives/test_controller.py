import math

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    CapsuleState, LocalPrimitiveController, PrimitiveId, PrimitiveStatus,
    axis_at_tilt, compose_endpoint_wrench, non_camera_endpoint_state,
)


def state_for_axis(axis, position=(0, 0, 0.02), linear=(0, 0, 0), angular=(0, 0, 0)):
    axis = np.asarray(axis, dtype=np.float64)
    local = np.array([0.0, 0.0, -1.0])
    dot = float(np.dot(local, axis))
    if dot < -0.999999:
        quat = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        cross = np.cross(local, axis)
        quat = np.r_[math.sqrt((1.0 + dot) / 2.0), cross / math.sqrt(2.0 * (1.0 + dot))]
    return CapsuleState(0.0, position, quat, linear, angular)


def test_start_gates_and_busy_rejection():
    controller = LocalPrimitiveController()
    side = state_for_axis([1, 0, 0])
    upright = state_for_axis([0, 0, 1])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, side)
    assert not controller.start(PrimitiveId.UPRIGHT_TO_SIDE, 0.0, upright)
    assert controller.last_request_result == "busy"
    controller.reset()
    assert not controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, upright)
    assert controller.status == PrimitiveStatus.INVALID_START
    assert controller.start(PrimitiveId.UPRIGHT_TO_30_DEG, 0.0, upright)


def test_non_camera_endpoint_uses_rigid_body_kinematics():
    state = state_for_axis(
        [1, 0, 0], position=[0.1, 0.2, 0.03], angular=[0, 2, 0],
    )
    endpoint = non_camera_endpoint_state(state, 0.0125)
    np.testing.assert_allclose(endpoint.offset_world_m, [-0.0125, 0, 0], atol=1e-12)
    np.testing.assert_allclose(endpoint.position_world_m, [0.0875, 0.2, 0.03], atol=1e-12)
    np.testing.assert_allclose(
        endpoint.velocity_world_m_s,
        state.linear_velocity_world_m_s
        + np.cross(state.angular_velocity_world_rad_s, endpoint.offset_world_m),
    )


def test_endpoint_force_converts_to_equivalent_com_wrench():
    offset = np.array([-0.0125, 0, 0])
    endpoint_force = np.array([0, 0, -0.1])
    force, torque = compose_endpoint_wrench(
        offset, endpoint_force, np.zeros(3), np.zeros(3),
    )
    np.testing.assert_allclose(force, endpoint_force)
    np.testing.assert_allclose(torque, np.cross(offset, endpoint_force))


def test_vertical_com_damping_has_no_height_target():
    controller = LocalPrimitiveController()
    stationary = state_for_axis([1, 0, 0], position=[0, 0, 10])
    moving = state_for_axis([1, 0, 0], position=[0, 0, 10], linear=[0, 0, 0.1])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, stationary)
    stationary_command, _ = controller.update(stationary, 0.1)
    controller.reset()
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, moving)
    moving_command, _ = controller.update(moving, 0.1)
    assert moving_command.force_world_n[2] < stationary_command.force_world_n[2]


def test_wrench_is_bounded_and_uses_non_camera_endpoint_force():
    controller = LocalPrimitiveController()
    state = state_for_axis([1, 0, 0], position=(1, 1, 1), linear=(10, 10, 0), angular=(10, 10, 10))
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, state)
    command, telemetry = controller.update(state, 1 / 240)
    assert np.linalg.norm(command.force_world_n) <= controller.cfg.total_force_limit_n + 1e-12
    assert telemetry.endpoint_force_world_n[2] == pytest.approx(-controller.cfg.endpoint_pin_force_n)
    assert np.linalg.norm(command.torque_world_nm) <= controller.cfg.total_torque_limit_nm + 1e-12
    np.testing.assert_allclose(telemetry.total_force_world_n, command.force_world_n)
    np.testing.assert_allclose(telemetry.total_torque_world_nm, command.torque_world_nm)
    assert telemetry.status == PrimitiveStatus.RUNNING


def test_controller_can_exceed_old_physical_torque_limit():
    controller = LocalPrimitiveController()
    state = state_for_axis([1, 0, 0])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, state)
    _, telemetry = controller.update(state, 1 / 240)
    assert np.linalg.norm(telemetry.total_torque_world_nm) > 3.0e-5
    assert np.linalg.norm(telemetry.total_torque_world_nm) <= 0.005


def test_total_wrench_obeys_slew_limits():
    controller = LocalPrimitiveController()
    state = state_for_axis([1, 0, 0])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, state)
    first, _ = controller.update(state, 1 / 240)
    second, _ = controller.update(state, 1 / 240)
    assert np.linalg.norm(first.force_world_n) <= controller.cfg.force_slew_limit_n_per_s / 240 + 1e-12
    assert np.linalg.norm(first.torque_world_nm) <= controller.cfg.torque_slew_limit_nm_per_s / 240 + 1e-12
    assert np.linalg.norm(second.force_world_n - first.force_world_n) <= controller.cfg.force_slew_limit_n_per_s / 240 + 1e-12
    assert np.linalg.norm(second.torque_world_nm - first.torque_world_nm) <= controller.cfg.torque_slew_limit_nm_per_s / 240 + 1e-12


def test_transition_completes_after_stable_hold():
    controller = LocalPrimitiveController()
    side = state_for_axis([1, 0, 0])
    upright = state_for_axis([0, 0, 1])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, side)
    dt = 0.1
    for _ in range(80):
        _, telemetry = controller.update(upright, dt)
        if telemetry.status == PrimitiveStatus.SUCCEEDED_HOLDING:
            break
    assert telemetry.status == PrimitiveStatus.SUCCEEDED_HOLDING
    assert controller.start(PrimitiveId.UPRIGHT_TO_30_DEG, 0.0, upright)


def test_simulation_first_stability_is_a_continuous_posture_window():
    controller = LocalPrimitiveController()
    side = state_for_axis([1, 0, 0])
    moving_upright = state_for_axis(
        [0, 0, 1], linear=[0.5, 0, 0], angular=[0, 1.0, 0],
    )
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, side)
    telemetry = None
    for _ in range(80):
        _, telemetry = controller.update(moving_upright, 0.1)
        if telemetry.status == PrimitiveStatus.SUCCEEDED_HOLDING:
            break
    assert telemetry.status == PrimitiveStatus.SUCCEEDED_HOLDING


def test_timeout_clears_wrench_and_accepts_new_valid_request():
    controller = LocalPrimitiveController()
    side = state_for_axis([1, 0, 0])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, side)
    command = None
    for _ in range(82):
        command, telemetry = controller.update(side, 0.1)
    assert telemetry.status == PrimitiveStatus.TIMED_OUT
    np.testing.assert_allclose(command.force_world_n, 0.0)
    np.testing.assert_allclose(command.torque_world_nm, 0.0)
    np.testing.assert_allclose(telemetry.total_force_world_n, 0.0)
    np.testing.assert_allclose(telemetry.total_torque_world_nm, 0.0)
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, side)


def test_cone_tracks_unwrapped_positive_phase_and_completes():
    controller = LocalPrimitiveController()
    start_axis = axis_at_tilt(math.radians(30), [1, 0])
    assert controller.start(PrimitiveId.CONE_30_DEG_ONE_REVOLUTION, 0.0, state_for_axis(start_axis))
    dt = 0.05
    telemetry = None
    for step in range(1, 190):
        elapsed = min(step * dt, controller.cfg.motion_duration_s[3])
        progress = min(elapsed / controller.cfg.motion_duration_s[3], 1.0)
        smooth = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
        phi = 2 * math.pi * smooth
        axis = axis_at_tilt(math.radians(30), [math.cos(phi), math.sin(phi)])
        _, telemetry = controller.update(state_for_axis(axis), dt)
        if telemetry.status == PrimitiveStatus.SUCCEEDED_HOLDING:
            break
    assert telemetry.cone_phase_rad >= 2 * math.pi - controller.cfg.cone_coverage_tolerance_rad
    assert telemetry.cone_tilt_rmse_rad < 1e-8
    assert telemetry.status == PrimitiveStatus.SUCCEEDED_HOLDING


def test_nonfinite_state_fails_closed():
    controller = LocalPrimitiveController()
    side = state_for_axis([1, 0, 0])
    assert controller.start(PrimitiveId.SIDE_TO_UPRIGHT, 0.0, side)
    bad = CapsuleState(0.0, [np.nan, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
    command, telemetry = controller.update(bad, 0.01)
    assert telemetry.status == PrimitiveStatus.NONFINITE
    np.testing.assert_allclose(command.force_world_n, 0.0)
    np.testing.assert_allclose(command.torque_world_nm, 0.0)
