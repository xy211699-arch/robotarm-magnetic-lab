import json

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.local_primitives import (
    CapsuleState, PrimitiveId, PrimitiveRequest, load_simulation_profile,
    make_local_primitive_controller_cfg, simulation_profile_sha256,
)


def _write_profile(tmp_path, **changes):
    source = load_simulation_profile()
    values = {
        key: value for key, value in source.__dict__.items()
    }
    values.update(changes)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_simulation_profile_is_tracked_and_not_weight_limited():
    profile = load_simulation_profile()
    assert profile.pose_torque_limit_nm >= 1.0e-4
    assert profile.total_force_limit_n <= 5.0
    assert profile.total_torque_limit_nm <= 0.02
    assert profile.force_slew_limit_n_per_s <= 50.0
    assert profile.torque_slew_limit_nm_per_s <= 0.2
    assert len(simulation_profile_sha256()) == 64


def test_profile_accepts_unrealistic_but_numerically_allowed_values(tmp_path):
    path = _write_profile(tmp_path, total_force_limit_n=2.0, total_torque_limit_nm=0.01)
    profile = load_simulation_profile(path)
    assert profile.total_force_limit_n == 2.0
    assert profile.total_torque_limit_nm == 0.01


def test_profile_rejects_unknown_keys_and_numerical_envelope(tmp_path):
    path = _write_profile(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["unknown"] = 1.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        load_simulation_profile(path)
    path = _write_profile(tmp_path, total_force_limit_n=5.01)
    with pytest.raises(ValueError, match="5 N"):
        load_simulation_profile(path)


def test_canonical_digest_ignores_json_formatting(tmp_path):
    path = _write_profile(tmp_path)
    digest = simulation_profile_sha256(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(raw, sort_keys=True, indent=4), encoding="utf-8")
    assert simulation_profile_sha256(path) == digest


def test_shared_config_sources_authority_from_tracked_profile():
    cfg = make_local_primitive_controller_cfg()
    profile = load_simulation_profile()
    assert cfg.capsule_mass_kg == pytest.approx(0.0057349997)
    assert cfg.motion_duration_s == (5.5, 4.5, 3.5, 8.0)
    assert cfg.hard_timeout_s == (8.0, 7.0, 6.0, 9.5)
    assert cfg.total_force_limit_n == profile.total_force_limit_n
    assert cfg.endpoint_pin_force_n == profile.endpoint_pin_force_n
    assert cfg.profile_sha256 == simulation_profile_sha256()


def test_request_validates_primitive_and_azimuth():
    request = PrimitiveRequest(PrimitiveId.UPRIGHT_TO_30_DEG, np.pi / 2)
    assert request.primitive_id is PrimitiveId.UPRIGHT_TO_30_DEG
    assert request.azimuth_rad == pytest.approx(np.pi / 2)
    with pytest.raises(ValueError):
        PrimitiveRequest(0, np.nan)


def test_capsule_state_detects_nonfinite_values():
    state = CapsuleState(0.0, np.zeros(3), [1, 0, 0, 0], np.zeros(3), np.zeros(3))
    assert state.is_finite
    bad = CapsuleState(0.0, [np.nan, 0, 0], [1, 0, 0, 0], np.zeros(3), np.zeros(3))
    assert not bad.is_finite
