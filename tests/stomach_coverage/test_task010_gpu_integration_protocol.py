from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stomach_coverage/validate_task010_gpu_integration.py"
SPEC = importlib.util.spec_from_file_location("task010_gate2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(MODULE)


def _valid():
    return {"schema": MODULE.SCHEMA, "status": "pass", "num_envs": 12, "sequences": 3, "rollout_steps": 64,
            "formal_steps": 1200, "physics_steps_per_action": 24, "all_finite": True,
            "rgb_coverage_same_frame": True, "resnet_unchanged": True, "true_terminal": True,
            "timeout_bootstrap": False, "controlled_zero_reachable_accepted": True,
            "controlled_zero_pose_id": "train-0419",
            "devices": {name: "cuda:0" for name in ("environment", "physics", "camera", "coverage")},
            "shapes": {"actor": [12, 519], "critic": [12, 65], "action": [12, 2], "reward": [12], "reset_mask": [12]}}


def test_gate2_protocol_accepts_exact_contract():
    assert MODULE.validate_summary(_valid())["status"] == "pass"


@pytest.mark.parametrize("field", ["num_envs", "formal_steps", "physics_steps_per_action", "true_terminal", "resnet_unchanged"])
def test_gate2_protocol_rejects_drift(field):
    value = _valid(); value[field] = None
    with pytest.raises(ValueError, match=field): MODULE.validate_summary(value)
