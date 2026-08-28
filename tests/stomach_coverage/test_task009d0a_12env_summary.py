from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "stomach_coverage"
sys.path.insert(0, str(SCRIPTS))

from summarize_task009d0a_12env import RARE_C0_ABORT, decide, is_accepted_rare_c0_abort  # noqa: E402


def candidate() -> dict:
    return {
        "independent_process_repeats": 3,
        "minimum_gpu_free_memory_fraction": 0.20,
        "baseline_median_environment_transitions_per_second": 30.310967,
        "minimum_relative_improvement": 0.10,
    }


def test_12_env_requires_strictly_more_than_ten_percent():
    threshold = 30.310967 * 1.10
    assert decide(candidate(), [threshold, threshold, threshold], 0.30, 0) == 8
    assert decide(candidate(), [threshold + 0.01] * 3, 0.30, 0) == 12


def test_12_env_rejected_for_memory_or_faults():
    assert decide(candidate(), [40.0] * 3, 0.19, 0) == 8
    assert decide(candidate(), [40.0] * 2, 0.30, 1) == 8


def test_only_exact_pre_measurement_c0_abort_is_waivable():
    records = [
        {"status": "pass", "faults": [], "measurements": [1]},
        {"status": "fail", "faults": [RARE_C0_ABORT], "measurements": []},
        {"status": "pass", "faults": [], "measurements": [1]},
    ]
    assert is_accepted_rare_c0_abort(records)
    records[1]["faults"] = ["RuntimeError: another failure"]
    assert not is_accepted_rare_c0_abort(records)
