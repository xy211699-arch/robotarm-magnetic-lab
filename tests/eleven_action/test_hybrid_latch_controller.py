import math

import numpy as np

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
    ActionResult,
    CapsuleState,
    ElevenActionController,
    ElevenActionId,
    FlatSurfaceQuery,
    LatchBackendName,
    LatchIntent,
    LatchReason,
    LatchedContactSnapshot,
    Lifecycle,
    load_dynamic_profile,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.contact_history import ContactSample
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.geometry import quaternion_wxyz_to_matrix


def _quat_from_axis(axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    source = np.asarray([0.0, 0.0, -1.0])
    dot = float(np.clip(source @ axis, -1.0, 1.0))
    if dot < -1.0 + 1.0e-12:
        return np.asarray([0.0, 1.0, 0.0, 0.0])
    q = np.asarray([1.0 + dot, *np.cross(source, axis)])
    return q / np.linalg.norm(q)


def _state(axis=(0.0, 0.0, 1.0), position=(0.0, 0.0, 0.02)):
    return CapsuleState(position, _quat_from_axis(axis), np.zeros(3), np.zeros(3))


def _controller():
    controller = ElevenActionController(load_dynamic_profile(), FlatSurfaceQuery.regular_plane())
    controller.confirm_latched(LatchBackendName.DYNAMIC_LOCK_FLAGS)
    return controller


def _finish(controller, state, start=0):
    return [controller.step(state, physics_substep=start + index) for index in range(240)]


def _state_at_frozen_support(controller, axis, tangent_offset=(0.0, 0.0, 0.0)):
    quaternion = _quat_from_axis(axis)
    rotated_offset = quaternion_wxyz_to_matrix(quaternion) @ controller._support.local_offset_m
    position = controller._support.anchor_world_m - rotated_offset + np.asarray(tangent_offset)
    return CapsuleState(position, quaternion, np.zeros(3), np.zeros(3))


def test_view_latches_on_first_target_and_drift_gate_substep():
    controller = _controller()
    start = _state()
    assert controller.submit(ElevenActionId.VIEW_RIGHT, start, physics_substep=0)
    first = controller.step(start, physics_substep=0)
    assert first.latch_intent is LatchIntent.UNLOCK
    controller.confirm_unlocked()
    inside = _state_at_frozen_support(controller, controller.target_axis_world)
    reached = controller.step(inside, physics_substep=1)
    assert reached.latch_intent is LatchIntent.LOCK
    assert reached.latch_reason is LatchReason.VIEW_TARGET
    np.testing.assert_allclose(reached.wrench.force_world_n, 0.0)
    np.testing.assert_allclose(reached.wrench.torque_world_nm, 0.0)


def test_view_does_not_latch_when_only_angle_passes():
    controller = _controller()
    start = _state()
    assert controller.submit(ElevenActionId.VIEW_RIGHT, start, physics_substep=0)
    controller.step(start, physics_substep=0)
    controller.confirm_unlocked()
    moved = _state_at_frozen_support(
        controller, controller.target_axis_world, tangent_offset=(0.0021, 0.0, 0.0)
    )
    result = controller.step(moved, physics_substep=1)
    assert result.latch_intent is LatchIntent.NONE


def test_camera_contact_has_priority_and_action_still_takes_240_substeps():
    controller = _controller()
    state = _state()
    assert controller.submit(ElevenActionId.VIEW_DOWN, state, physics_substep=0)
    records = []
    for index in range(240):
        if index == 20:
            controller.observe_contact(ContactSample(index, [0, 0, 0], [0, 0, 1], 0.007))
        records.append(controller.step(state, physics_substep=index))
        if records[-1].latch_intent is LatchIntent.LOCK:
            controller.confirm_latched(LatchBackendName.DYNAMIC_LOCK_FLAGS)
    assert records[20].latch_reason is LatchReason.CAMERA_CONTACT
    assert records[-1].telemetry.result is ActionResult.COMPLETED
    assert records[-1].telemetry.substep_index == 240


def test_timeout_latches_actual_state_without_fourth_result():
    controller = _controller()
    state = _state()
    assert controller.submit(ElevenActionId.VIEW_RIGHT, state, physics_substep=0)
    records = _finish(controller, state)
    assert records[-1].latch_reason is LatchReason.ACTION_BOUNDARY
    assert records[-1].telemetry.result is ActionResult.COMPLETED
    assert controller.lifecycle is Lifecycle.LATCHED_READY


def test_hold_and_rejected_move_remain_latched_for_240_substeps():
    state = _state(axis=(1.0, 0.0, 0.0))
    hold = _controller()
    assert hold.submit(ElevenActionId.HOLD_VIEW, state, physics_substep=0)
    hold_rows = _finish(hold, state)
    rejected = _controller()
    rejected.set_latched_contact_snapshot(LatchedContactSnapshot(False, False, False, 0))
    assert rejected.submit(ElevenActionId.MOVE_SIDE_POS, state, physics_substep=0)
    rejected_rows = _finish(rejected, state)
    assert hold_rows[-1].telemetry.result is ActionResult.COMPLETED
    assert rejected_rows[-1].telemetry.result is ActionResult.REJECTED
    for row in hold_rows + rejected_rows:
        np.testing.assert_allclose(row.wrench.force_world_n, 0.0)
        np.testing.assert_allclose(row.wrench.torque_world_nm, 0.0)


def test_move_eligibility_uses_immutable_latched_snapshot():
    controller = _controller()
    state = _state(axis=(1.0, 0.0, 0.0))
    controller.set_latched_contact_snapshot(LatchedContactSnapshot(True, False, True, 12))
    assert controller.submit(ElevenActionId.MOVE_SIDE_NEG, state, physics_substep=20)
    assert controller.accepted_move
    # Absence of subsequent live force cannot erase the acceptance decision.
    first = controller.step(state, physics_substep=20)
    assert first.latch_intent is LatchIntent.UNLOCK
