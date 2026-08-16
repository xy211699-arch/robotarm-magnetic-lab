"""Schema and decision-gate tests for TASK-003 real dynamics."""

from __future__ import annotations

import pytest

from scripts.dynamic_force.inspect_dynamic_force_prerequisites import (
    REQUIRED_REPORT_KEYS,
    build_gate,
    validate_preflight_report,
)


def test_gate_requires_true_dynamic_capsule_and_ccd(valid_report):
    report = valid_report()
    validate_preflight_report(report)
    assert set(report) == set(REQUIRED_REPORT_KEYS)
    assert report["capsule"]["kinematic_enabled"] is False
    assert report["capsule"]["gravity_enabled"] is True
    assert report["capsule"]["ccd_enabled"] is True
    assert report["physics"]["scene_ccd_enabled"] is True


def test_gate_rejects_runtime_pose_writer(valid_report):
    report = valid_report()
    report["runtime_contract"]["forbidden_calls"] = ["write_root_pose_to_sim"]
    report["gate"] = build_gate(report)
    with pytest.raises(ValueError, match="forbidden runtime state writer"):
        validate_preflight_report(report)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("capsule", "kinematic_enabled"), True, "capsule is kinematic"),
        (("capsule", "gravity_enabled"), False, "capsule gravity is disabled"),
        (("capsule", "ccd_enabled"), False, "CCD is not active"),
        (("physics", "scene_ccd_enabled"), False, "CCD is not active"),
        (("contact_sensor", "present"), False, "contact sensor is unavailable"),
        (("stomach", "static"), False, "stomach collider is not static"),
    ],
)
def test_gate_rejects_missing_real_dynamics_property(valid_report, path, value, message):
    report = valid_report()
    report[path[0]][path[1]] = value
    report["gate"] = build_gate(report)
    with pytest.raises(ValueError, match=message):
        validate_preflight_report(report)


def test_gate_rejects_wrong_frozen_rates(valid_report):
    report = valid_report()
    report["physics"]["dt_s"] = 1.0 / 120.0
    report["gate"] = build_gate(report)
    with pytest.raises(ValueError, match="physics dt is not 1/240 s"):
        validate_preflight_report(report)


def test_gate_rejects_wrong_capsule_geometry(valid_report):
    report = valid_report()
    report["capsule"]["radius_m"] = 0.007
    report["gate"] = build_gate(report)
    with pytest.raises(ValueError, match="capsule radius is not 6.5 mm"):
        validate_preflight_report(report)
