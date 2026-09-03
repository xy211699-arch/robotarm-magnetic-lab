from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stomach_coverage"))

from validate_task010_visual_dependence_gate import validate_gate_evidence  # noqa: E402


def _complete_fake_evidence():
    return {
        "v0": {
            "critic_isolation": True,
            "blind_visual_projection": True,
            "identical_trainable_parameters": True,
            "actor_observation_schema": True,
            "resnet_forward_count": True,
        },
        "v1": {"blind_forward_backward_save_restore": True},
        "v2": {
            "donor_mapping": True,
            "first_frame_repeat": True,
            "target_previous_action": True,
            "unique_variables": True,
            "curve_length": True,
        },
        "v3": {"status": "awaiting_manual_start"},
    }


def test_gate_accepts_complete_awaiting_manual_start_evidence():
    report = validate_gate_evidence(_complete_fake_evidence())
    assert report["status"] == "passed"
    assert report["v3"]["status"] == "awaiting_manual_start"


def test_gate_rejects_any_missing_evidence():
    evidence = _complete_fake_evidence()
    del evidence["v0"]["critic_isolation"]
    with pytest.raises(RuntimeError, match="critic_isolation"):
        validate_gate_evidence(evidence)


def test_gate_cannot_mark_formal_v3_complete():
    evidence = _complete_fake_evidence()
    evidence["v3"]["status"] = "completed"
    with pytest.raises(RuntimeError, match="awaiting_manual_start"):
        validate_gate_evidence(evidence)
