import math

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
    ActionResult,
    CapsuleState,
    ElevenActionId,
    LatchedContactSnapshot,
    Lifecycle,
    load_dynamic_profile,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.contact_history import ContactSample
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.controller import ElevenActionController
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.geometry import capsule_axis_world
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.surface_query import FlatSurfaceQuery


def _quat_from_axis(axis):
    """WXYZ orientation mapping local -Z to axis with no material twist requirement."""
    axis = np.asarray(axis, dtype=np.float64).copy()
    axis /= np.linalg.norm(axis)
    source = np.asarray([0.0, 0.0, -1.0])
    dot = float(np.clip(source @ axis, -1.0, 1.0))
    if dot < -1.0 + 1.0e-12:
        return np.asarray([0.0, 1.0, 0.0, 0.0])
    cross = np.cross(source, axis)
    q = np.asarray([1.0 + dot, *cross])
    q /= np.linalg.norm(q)
    return q


def _state(axis=(0.0, 0.0, 1.0), angular_velocity=(0.0, 0.0, 0.0)):
    return CapsuleState(
        np.asarray([0.0, 0.0, 0.02]),
        _quat_from_axis(axis),
        np.zeros(3),
        np.asarray(angular_velocity),
    )


def _controller():
    return ElevenActionController(load_dynamic_profile(), FlatSurfaceQuery.regular_plane())


def _run(controller, state, *, start_substep=0):
    records = []
    for index in range(240):
        records.append(controller.step(state, physics_substep=start_substep + index))
    return records


@pytest.mark.parametrize("action", list(ElevenActionId)[1:9])
def test_each_view_uses_192_swing_48_hold_and_exact_240_substeps(action):
    controller = _controller()
    state = _state()
    assert controller.submit(action, state, physics_substep=0)
    records = _run(controller, state)
    start = records[0].telemetry.start_axis_world
    target = records[192].telemetry.desired_axis_world
    np.testing.assert_allclose(records[0].telemetry.desired_axis_world, start, atol=1.0e-12)
    np.testing.assert_allclose(records[192].telemetry.desired_axis_world, target, atol=1.0e-12)
    np.testing.assert_allclose(records[239].telemetry.desired_axis_world, target, atol=1.0e-12)
    assert records[238].telemetry.result is None
    assert records[239].telemetry.result is ActionResult.COMPLETED
    assert records[239].telemetry.substep_index == 240
    assert controller.lifecycle is Lifecycle.LATCHED_READY


def test_hold_and_consecutive_view_are_relative_to_each_real_start():
    controller = _controller()
    first = _state()
    assert controller.submit(ElevenActionId.HOLD_VIEW, first, physics_substep=0)
    hold = _run(controller, first)
    assert hold[-1].telemetry.result is ActionResult.COMPLETED

    assert controller.submit(ElevenActionId.VIEW_RIGHT, first, physics_substep=240)
    first_view = _run(controller, first, start_substep=240)
    first_target = first_view[-1].telemetry.desired_axis_world
    second_start_state = _state(first_target)
    assert controller.submit(ElevenActionId.VIEW_RIGHT, second_start_state, physics_substep=480)
    second_view = _run(controller, second_start_state, start_substep=480)
    np.testing.assert_allclose(second_view[0].telemetry.start_axis_world, capsule_axis_world(second_start_state))
    assert math.degrees(math.acos(np.clip(float(first_target @ second_view[-1].telemetry.desired_axis_world), -1, 1))) == pytest.approx(15.0)


def test_camera_contact_freezes_next_update_but_completes_full_action():
    controller = _controller()
    state = _state()
    assert controller.submit(ElevenActionId.VIEW_DOWN, state, physics_substep=0)
    records = []
    for index in range(240):
        if index == 101:
            controller.observe_contact(
                ContactSample(index, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.007)
            )
        records.append(controller.step(state, physics_substep=index))
    np.testing.assert_allclose(records[101].telemetry.desired_axis_world, capsule_axis_world(state))
    assert records[-1].telemetry.result is ActionResult.COMPLETED
    assert records[-1].telemetry.constrained
    assert records[-1].telemetry.contact_cancel_delay_substeps == 1


def test_persistent_camera_constraint_holds_same_direction_reverse_leaves_and_expires():
    controller = _controller()
    state = _state()
    controller.observe_contact(ContactSample(9, [0, 0, 0], [0, -1, 0], 0.007))
    assert controller.submit(ElevenActionId.VIEW_DOWN, state, physics_substep=10)
    blocked = _run(controller, state, start_substep=10)
    assert blocked[-1].telemetry.constrained
    np.testing.assert_allclose(blocked[-1].telemetry.desired_axis_world, capsule_axis_world(state))

    controller.observe_contact(ContactSample(250, [0, 0, 0], [0, -1, 0], 0.007))
    assert controller.submit(ElevenActionId.VIEW_UP, state, physics_substep=251)
    reverse = _run(controller, state, start_substep=251)
    assert not reverse[0].telemetry.constrained
    assert not np.allclose(reverse[-1].telemetry.desired_axis_world, capsule_axis_world(state))

    # No contact in the last 12 physics substeps means the temporary constraint is gone.
    assert controller.submit(ElevenActionId.VIEW_DOWN, state, physics_substep=503)
    expired = _run(controller, state, start_substep=503)
    assert not expired[0].telemetry.constrained


@pytest.mark.parametrize("tilt_deg", [59.999, 60.0, 75.0, 90.0])
def test_move_precondition_is_latched_once_and_rejected_actions_still_take_240_steps(tilt_deg):
    controller = _controller()
    normal = np.asarray([0.0, 0.0, 1.0])
    axis = np.asarray([math.sin(math.radians(tilt_deg)), 0.0, math.cos(math.radians(tilt_deg))])
    state = _state(axis)
    controller.set_latched_contact_snapshot(
        LatchedContactSnapshot(True, False, True, 8)
    )
    assert controller.submit(ElevenActionId.MOVE_SIDE_POS, state, physics_substep=10)
    records = _run(controller, state, start_substep=10)
    expected = ActionResult.REJECTED if tilt_deg < 60.0 else ActionResult.COMPLETED
    assert records[-1].telemetry.result is expected
    assert len(records) == 240


def test_move_requires_recent_sidewall_contact_and_uses_exact_three_phases():
    state = _state([1.0, 0.0, 0.0])
    rejected = _controller()
    assert rejected.submit(ElevenActionId.MOVE_SIDE_POS, state, physics_substep=20)
    rejected_records = _run(rejected, state, start_substep=20)
    assert rejected_records[-1].telemetry.result is ActionResult.REJECTED

    controller = _controller()
    controller.set_latched_contact_snapshot(
        LatchedContactSnapshot(True, False, True, 19)
    )
    assert controller.submit(ElevenActionId.MOVE_SIDE_POS, state, physics_substep=20)
    records = _run(controller, state, start_substep=20)
    for index in list(range(60)) + list(range(180, 240)):
        np.testing.assert_allclose(records[index].wrench.force_world_n, 0.0)
        np.testing.assert_allclose(records[index].wrench.torque_world_nm, 0.0)
    for index in range(60, 180):
        assert np.linalg.norm(records[index].wrench.force_world_n) > 0.0
        np.testing.assert_allclose(records[index].wrench.torque_world_nm, 0.0)
    frozen = records[60].telemetry.move_direction_world
    np.testing.assert_allclose(records[179].telemetry.move_direction_world, frozen)


def test_degenerate_move_is_completed_low_effect_and_large_finite_rate_is_not_fault():
    controller = _controller()
    state = _state([0.0, 0.0, -1.0], angular_velocity=[1.0e6, -1.0e6, 2.0e6])
    controller.set_latched_contact_snapshot(
        LatchedContactSnapshot(True, False, True, 3)
    )
    assert controller.submit(ElevenActionId.MOVE_SIDE_NEG, state, physics_substep=4)
    records = _run(controller, state, start_substep=4)
    assert records[-1].telemetry.result is ActionResult.COMPLETED
    assert records[-1].telemetry.direction_degenerate
    assert controller.lifecycle is Lifecycle.LATCHED_READY


def test_only_nonfinite_state_faults_and_executing_submit_is_discarded():
    controller = _controller()
    state = _state()
    assert controller.submit(ElevenActionId.VIEW_UP, state, physics_substep=0)
    assert not controller.submit(ElevenActionId.VIEW_DOWN, state, physics_substep=0)
    bad = CapsuleState([np.nan, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
    result = controller.step(bad, physics_substep=0)
    assert result.telemetry.result is ActionResult.FAULT
    assert controller.lifecycle is Lifecycle.FAULTED


def test_fixed_view_hold_is_latched_and_zero_wrench():
    controller = _controller()
    state = _state([1.0, 0.0, 0.0])
    assert controller.submit(ElevenActionId.HOLD_VIEW, state, physics_substep=0)
    step = controller.step(state, physics_substep=0)
    np.testing.assert_allclose(step.wrench.torque_world_nm, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(step.wrench.force_world_n, 0.0, atol=1.0e-12)


def test_hold_ignores_live_contact_wrench_while_latched():
    controller = _controller()
    state = _state([1.0, 0.0, 0.0])
    point = state.position_world_m + np.asarray([0.01, 0.0, 0.0])
    force = np.asarray([0.0, 0.0, 0.01])
    controller.observe_contact(
        ContactSample(
            0,
            point,
            [0.0, 0.0, 1.0],
            0.0,
            force_world_n=force,
        )
    )
    assert controller.submit(ElevenActionId.HOLD_VIEW, state, physics_substep=0)
    step = controller.step(state, physics_substep=0)
    np.testing.assert_allclose(step.wrench.force_world_n, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(step.wrench.torque_world_nm, 0.0, atol=1.0e-12)
