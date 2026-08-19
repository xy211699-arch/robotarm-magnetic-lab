from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from virtual_magnet import (
    ActionId,
    ActionResult,
    ControllerState,
    Lifecycle,
    VirtualMagnetElevenActionController,
    load_profile,
)


def _state(*, position=(0.0, 0.0, 0.0), optical=(0.0, 0.0, 1.0), side=True, last_contact=0):
    position = np.asarray(position, dtype=float)
    return ControllerState(
        capsule_position=position,
        capsule_rotation=np.eye(3),
        capsule_magnet_position=position + np.array([0.0, 0.0, -0.004]),
        capsule_magnet_rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        optical_axis=np.asarray(optical, dtype=float),
        camera_up=np.array([0.0, 1.0, 0.0]),
        camera_right=np.array([1.0, 0.0, 0.0]),
        long_axis=np.array([1.0, 0.0, 0.0]),
        inward_normal=np.array([0.0, 0.0, 1.0]),
        contact_point=position,
        sidewall_contact=side,
        last_sidewall_contact_substep=last_contact,
    )


def _model(position, rotation):
    coordinates = np.concatenate((position, Rotation.from_matrix(rotation).as_rotvec()))
    return np.diag([0.2, 0.25, 0.3, 0.05, 0.06, 0.07]) @ coordinates


def _controller(profile=None):
    return VirtualMagnetElevenActionController(
        profile or load_profile(),
        _model,
        initial_magnet_position=np.array([0.0, 0.0, 0.08]),
        initial_magnet_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )


def test_action_occupies_exactly_240_steps_with_60_feedback_events():
    controller = _controller()
    state = _state()
    assert controller.submit(ActionId.HOLD_VIEW, state)
    assert not controller.submit(ActionId.VIEW_UP, state)
    commands = [controller.step(state) for _ in range(240)]
    assert all(command is not None for command in commands)
    assert controller.lifecycle == Lifecycle.TERMINAL
    assert controller.telemetry.substep == 240
    assert controller.telemetry.feedback_updates == 60
    assert controller.telemetry.result == ActionResult.COMPLETED
    assert controller.step(state) is None


def test_view_target_is_frozen_at_submission_and_pose_is_smooth():
    controller = _controller()
    start = _state()
    assert controller.submit(ActionId.VIEW_UP_RIGHT, start)
    frozen = controller.frozen_target.target_optical_axis.copy()
    changed = _state(optical=(0.2, 0.0, 0.9797959))
    positions = []
    for _ in range(240):
        command = controller.step(changed)
        positions.append(command.virtual_magnet_position.copy())
    np.testing.assert_allclose(controller.frozen_target.target_optical_axis, frozen)
    increments = np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1)
    assert np.max(increments) <= controller.profile.translation_trust_m / controller.profile.feedback_stride + 1.0e-10


def test_invalid_move_executes_full_hold_and_returns_rejected():
    controller = _controller()
    invalid = _state(side=False, last_contact=-1000)
    assert controller.submit(ActionId.MOVE_SIDE_POS, invalid)
    for _ in range(240):
        assert controller.step(invalid) is not None
    assert controller.telemetry.result == ActionResult.REJECTED
    assert controller.telemetry.substep == 240


def test_true_nonfinite_state_is_fault_but_finite_solver_saturation_is_not():
    controller = _controller()
    state = _state()
    assert controller.submit(ActionId.HOLD_VIEW, state)
    bad = replace(state, capsule_position=np.array([np.nan, 0.0, 0.0]))
    assert controller.step(bad) is None
    assert controller.lifecycle == Lifecycle.FAULT
    assert controller.telemetry.result == ActionResult.FAULT


def test_truth_feedback_changes_magnet_command_but_disabled_baseline_does_not():
    profile = load_profile()
    enabled = _controller(profile)
    disabled = _controller(replace(profile, feedback_enabled=False))
    state = _state()
    shifted = _state(position=(0.003, -0.002, 0.0), optical=(0.0, 0.1, 0.994987))
    for controller in (enabled, disabled):
        assert controller.submit(ActionId.HOLD_VIEW, state)
        for _ in range(4):
            controller.step(state)
    enabled_before = enabled.command.virtual_magnet_position.copy()
    disabled_before = disabled.command.virtual_magnet_position.copy()
    for _ in range(4):
        enabled.step(shifted)
        disabled.step(shifted)
    assert np.linalg.norm(enabled.command.virtual_magnet_position - enabled_before) > 1.0e-7
    np.testing.assert_allclose(disabled.command.virtual_magnet_position, disabled_before)
