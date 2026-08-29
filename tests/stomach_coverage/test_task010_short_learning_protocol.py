from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stomach_coverage/validate_task010_short_learning.py"
SPEC = importlib.util.spec_from_file_location("task010_gate3", SCRIPT); MODULE = importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(MODULE)


def _valid():
    return {"schema": MODULE.SCHEMA, "seed": 991010, "num_envs": 12, "updates_completed": 8,
            "actor_parameter_delta_l2": 0.1, "critic_parameter_delta_l2": 0.2,
            "all_finite": True, "resume_verified": True}


def test_short_learning_requires_real_updates(): assert MODULE.validate_summary(_valid())["updates_completed"] == 8


@pytest.mark.parametrize("field", ["seed", "num_envs", "updates_completed", "all_finite", "resume_verified"])
def test_short_learning_rejects_contract_drift(field):
    value = _valid(); value[field] = None
    with pytest.raises(ValueError, match=field): MODULE.validate_summary(value)


@pytest.mark.parametrize("field", ["actor_parameter_delta_l2", "critic_parameter_delta_l2"])
def test_short_learning_requires_parameter_change(field):
    value = _valid(); value[field] = 0.0
    with pytest.raises(ValueError, match="parameters did not change"): MODULE.validate_summary(value)
