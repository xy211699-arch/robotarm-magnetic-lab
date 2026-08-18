import json
import math

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
    ActionResult,
    ActionTelemetry,
    ElevenActionId,
    Lifecycle,
    load_dynamic_profile,
    dynamic_profile_sha256,
)


EXPECTED_IDS = {
    "HOLD_VIEW": 0,
    "VIEW_UP": 1,
    "VIEW_UP_RIGHT": 2,
    "VIEW_RIGHT": 3,
    "VIEW_DOWN_RIGHT": 4,
    "VIEW_DOWN": 5,
    "VIEW_DOWN_LEFT": 6,
    "VIEW_LEFT": 7,
    "VIEW_UP_LEFT": 8,
    "MOVE_SIDE_POS": 9,
    "MOVE_SIDE_NEG": 10,
}


def _write_profile(tmp_path, **changes):
    profile = load_dynamic_profile()
    values = dict(profile.__dict__)
    values.update(changes)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_public_ids_and_results_are_frozen():
    assert {item.name: int(item) for item in ElevenActionId} == EXPECTED_IDS
    assert -1 not in [int(item) for item in ElevenActionId]
    assert {item.value for item in ActionResult} == {"completed", "rejected", "fault"}
    assert {item.value for item in Lifecycle} == {"ready_hold", "executing", "faulted"}


def test_tracked_profile_has_exact_timing_and_canonical_digest():
    profile = load_dynamic_profile()
    assert profile.schema_version == "task005_eleven_action_dynamic_v1"
    assert profile.physics_hz == 240
    assert profile.action_substeps == 240
    assert profile.view_motion_substeps == 192
    assert profile.view_hold_substeps == 48
    assert profile.contact_history_substeps == 12
    assert profile.view_cone_half_angle_deg == pytest.approx(15.0)
    assert profile.move_min_tilt_deg == pytest.approx(60.0)
    assert profile.move_force_k == pytest.approx(0.9)
    assert len(dynamic_profile_sha256()) == 64


def test_profile_rejects_missing_extra_nonfinite_and_invalid_timing(tmp_path):
    path = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("capsule_mass_kg")
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        load_dynamic_profile(path)

    path = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["unexpected"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        load_dynamic_profile(path)

    with pytest.raises(ValueError, match="finite"):
        load_dynamic_profile(_write_profile(tmp_path, axis_kp_nm_per_rad=math.nan))
    with pytest.raises(ValueError, match="240"):
        load_dynamic_profile(_write_profile(tmp_path, action_duration_s=0.999))
    with pytest.raises(ValueError, match="12"):
        load_dynamic_profile(_write_profile(tmp_path, contact_history_s=0.049))


def test_profile_digest_ignores_json_formatting(tmp_path):
    path = _write_profile(tmp_path)
    digest = dynamic_profile_sha256(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(raw, indent=4, sort_keys=True), encoding="utf-8")
    assert dynamic_profile_sha256(path) == digest


def test_telemetry_exposes_required_contract_fields():
    telemetry = ActionTelemetry.empty("a" * 64)
    assert telemetry.lifecycle is Lifecycle.READY_HOLD
    assert telemetry.result is None
    assert telemetry.action_id is None
    assert telemetry.request_id == 0
    assert telemetry.substep_index == 0
    assert telemetry.profile_sha256 == "a" * 64
    assert telemetry.start_axis_world.shape == (3,)
    assert telemetry.end_axis_world.shape == (3,)
    assert telemetry.surface_normal_world.shape == (3,)
    assert telemetry.support_anchor_world_m.shape == (3,)
    assert telemetry.move_direction_world.shape == (3,)
    assert telemetry.force_world_n.shape == (3,)
    assert telemetry.torque_world_nm.shape == (3,)
    assert np.isfinite(telemetry.force_world_n).all()
