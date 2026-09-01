from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "stomach_coverage"
sys.path.insert(0, str(SCRIPTS))

from summarize_random_baselines import (  # noqa: E402
    EpisodeProtocolError,
    aggregate_formal,
    write_formal_outputs,
)


POLICIES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")


def _formal_curves() -> dict[str, list[dict]]:
    times = np.arange(3001, dtype=np.float64) / 10.0
    curves: dict[str, list[dict]] = {}
    for policy_index, policy in enumerate(POLICIES, start=1):
        for pose_index in range(5):
            start = 0.01 * pose_index
            finish = start + 0.001 * policy_index
            curves[f"{policy}-{pose_index}"] = [
                {
                    "policy_id": policy,
                    "sim_time_s": float(time_s),
                    "reachable_coverage_fraction": float(
                        start + (finish - start) * time_s / 300.0
                    ),
                    "raw_coverage_fraction": float(
                        start + (finish - start) * time_s / 300.0
                    ),
                }
                for time_s in times
            ]
    for pose_index in range(2):
        start = 0.02 * pose_index
        curves[f"HOLD-{pose_index}"] = [
            {
                "policy_id": "HOLD",
                "sim_time_s": float(time_s),
                "reachable_coverage_fraction": start,
                "raw_coverage_fraction": start,
            }
            for time_s in times
        ]
    return curves


def test_formal_aggregation_requires_exact_counts_and_3001_aligned_points():
    grouped = aggregate_formal(_formal_curves(), {})
    assert set(grouped) == {*POLICIES, "HOLD"}
    assert len(grouped["R1"]["time_s"]) == 3001
    assert grouped["R1"]["reachable_mean"][-1] == pytest.approx(0.021)
    assert grouped["R7"]["delta_mean"][-1] == pytest.approx(0.007)
    assert grouped["HOLD"]["delta_mean"][-1] == pytest.approx(0.0)


def test_formal_aggregation_rejects_missing_episode_without_repair():
    curves = _formal_curves()
    curves.pop("R4-3")
    with pytest.raises(EpisodeProtocolError, match="exactly 5"):
        aggregate_formal(curves, {})


def test_formal_aggregation_rejects_misaligned_time_without_interpolation():
    curves = _formal_curves()
    curves["R2-1"][99]["sim_time_s"] = 9.95
    with pytest.raises(EpisodeProtocolError, match="not exactly aligned"):
        aggregate_formal(curves, {})


def test_candidate_tables_use_exact_frozen_boundaries(tmp_path):
    grouped = aggregate_formal(_formal_curves(), {})
    artifacts = write_formal_outputs(
        tmp_path,
        grouped,
        {"candidate_times_s": [30, 60, 120, 180, 240, 300]},
        {"episode": {"status": "pass"}},
    )
    candidate = Path(artifacts["candidate_times_csv"]["path"])
    lines = candidate.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 8 * 6
    assert ",30," in lines[1]
    assert ",300," in lines[-1]


def test_candidate_tables_reject_non_boundary_time(tmp_path):
    grouped = aggregate_formal(_formal_curves(), {})
    config = {"candidate_times_s": [30.05]}
    with pytest.raises(EpisodeProtocolError, match="not an exact boundary"):
        write_formal_outputs(tmp_path, grouped, config, {})


def test_twenty_pose_comparison_uses_configured_1501_points_without_hold():
    times = np.arange(1501, dtype=np.float64) / 10.0
    curves = {}
    episodes = []
    pose_ids = [f"validation-{index:04d}" for index in range(20)]
    for policy in POLICIES:
        for pose_id in pose_ids:
            episode_id = f"{policy}-{pose_id}"
            episodes.append({"policy_id": policy, "action_cycles": 1500})
            curves[episode_id] = [
                {"policy_id": policy, "sim_time_s": float(time_s),
                 "reachable_coverage_fraction": float(time_s / 150.0),
                 "raw_coverage_fraction": float(time_s / 150.0)}
                for time_s in times
            ]
    config = {
        "schema": "robotarm_magnetic_lab.task009c_random_baseline_20pose_comparison",
        "validation_pose_ids": pose_ids,
        "formal_episodes": episodes,
    }
    grouped = aggregate_formal(curves, config)
    assert tuple(grouped) == POLICIES
    assert all(len(row["time_s"]) == 1501 for row in grouped.values())
