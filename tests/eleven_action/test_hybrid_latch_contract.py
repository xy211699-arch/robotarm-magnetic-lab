from dataclasses import FrozenInstanceError

import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
    ActionResult,
    LatchBackendName,
    LatchIntent,
    LatchReason,
    LatchedContactSnapshot,
    load_latch_profile,
)


def test_public_results_remain_exactly_three():
    assert {item.value for item in ActionResult} == {"completed", "rejected", "fault"}


def test_latch_profile_is_frozen():
    profile = load_latch_profile()
    assert profile.schema_version == "task006_hybrid_latched_v1"
    assert profile.physics_hz == 240
    assert profile.policy_rgb_hz == 1
    assert profile.view_error_limit_deg == 3.0
    assert profile.support_drift_limit_m == 0.002
    assert profile.release_window_s == 0.05
    assert profile.release_position_delta_limit_m == 0.0005
    assert profile.release_axis_delta_limit_deg == 1.0
    assert profile.preferred_backend is LatchBackendName.DYNAMIC_LOCK_FLAGS
    assert profile.fallback_backend is LatchBackendName.KINEMATIC
    assert profile.selected_backend is LatchBackendName.DYNAMIC_LOCK_FLAGS


def test_contact_snapshot_is_immutable_and_independent_of_live_force():
    snapshot = LatchedContactSnapshot(True, False, True, 120)
    assert snapshot.sidewall_contact
    with pytest.raises(FrozenInstanceError):
        snapshot.sidewall_contact = False


def test_latch_intent_and_reason_are_internal_not_action_results():
    assert LatchIntent.LOCK.value == "lock"
    assert LatchIntent.UNLOCK.value == "unlock"
    assert LatchReason.VIEW_TARGET.value == "view_target"
    assert not hasattr(ActionResult, "TARGET_REACHED")
