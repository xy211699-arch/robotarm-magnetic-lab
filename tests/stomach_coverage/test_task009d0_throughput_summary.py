from __future__ import annotations

from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "stomach_coverage"
sys.path.insert(0, str(SCRIPTS))

from summarize_task009d0_throughput import select_num_envs  # noqa: E402


def row(num_envs: int, throughput: float, free: float, faults: int = 0) -> dict:
    return {
        "num_envs": num_envs,
        "median_environment_transitions_per_second": throughput,
        "minimum_gpu_free_fraction": free,
        "fault_count": faults,
    }


def test_near_tie_selects_smaller_candidate():
    rows = [row(4, 39.0, 0.30), row(8, 42.0, 0.25)]
    assert select_num_envs(rows, near_tie_fraction=0.10, minimum_free=0.20) == 4


def test_fast_candidate_with_insufficient_memory_is_rejected():
    rows = [row(4, 30.0, 0.25), row(8, 50.0, 0.19)]
    assert select_num_envs(rows, near_tie_fraction=0.10, minimum_free=0.20) == 4


def test_faulted_candidate_is_rejected():
    rows = [row(2, 20.0, 0.40), row(4, 40.0, 0.40, faults=1)]
    assert select_num_envs(rows, near_tie_fraction=0.10, minimum_free=0.20) == 2


def test_no_eligible_candidate_is_an_error():
    with pytest.raises(ValueError, match="no throughput candidate"):
        select_num_envs([row(8, 99.0, 0.10)], near_tie_fraction=0.10, minimum_free=0.20)
