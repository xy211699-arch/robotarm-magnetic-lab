from __future__ import annotations

from copy import deepcopy

import pytest

from robotarm_magnetic_lab.runtime.task009c_episode_runner import (
    EPISODE_RECORD_SCHEMA,
    EpisodeProtocolError,
    EpisodeSpec,
    summarize_episode,
    validate_episode_records,
)


def _records(cycles: int):
    rows = []
    for index in range(cycles + 1):
        coverage = 0.1 + 0.4 * index / max(cycles, 1)
        mode = None if index == 0 else (index - 1) % 6
        mode_names = ("HOLD", "MOVE_POS", "MOVE_NEG", "VIEW_POS", "VIEW_NEG", "UP")
        rows.append(
            {
                "schema": EPISODE_RECORD_SCHEMA,
                "task_version": 1,
                "config_sha256": "config",
                "run_id": "run",
                "kind": "smoke" if cycles == 30 else "formal",
                "episode_id": "episode",
                "policy_id": "R1",
                "pose_id": "validation-0006",
                "environment_seed": 950006,
                "policy_seed": 961006,
                "boundary_index": index,
                "sim_time_s": 0.1 * index,
                "mode_id": mode,
                "mode_name": "C0" if mode is None else mode_names[mode],
                "alpha": None if mode is None else (0.0 if mode == 0 else 0.5),
                "force_ratio_mg": 0.0 if mode in (None, 0) else 1.0,
                "physics_substeps": 0 if index == 0 else 24,
                "actor_rgb_frame": 100 + index,
                "coverage_rgb_frame": 100 + index,
                "rgb_content_sha256": f"rgb-{index}",
                "reachable_current_visible_area_m2": 0.001,
                "reachable_cumulative_coverage_area_m2": coverage * 0.04,
                "reachable_coverage_fraction": coverage,
                "raw_cumulative_coverage_area_m2": coverage * 0.06,
                "raw_coverage_fraction": coverage,
                "capsule_position_world_m": [0.001 * index, 0.0, 0.0],
                "capsule_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "capsule_linear_velocity_world_m_s": [0.01, 0.0, 0.0],
                "capsule_angular_velocity_world_rad_s": [0.0, 0.1, 0.0],
                "finite": True,
                "terminated": False,
                "truncated": False,
            }
        )
    return rows


def test_episode_spec_requires_exact_ten_hz_duration():
    spec = EpisodeSpec.from_record(
        {
            "episode_id": "smoke",
            "kind": "smoke",
            "policy_id": "R1",
            "pose_id": "validation-0006",
            "environment_seed": 950006,
            "policy_seed": 961006,
            "duration_s": 3.0,
            "action_cycles": 30,
        }
    )
    assert spec.action_cycles == 30
    with pytest.raises(ValueError, match="inconsistent"):
        EpisodeSpec.from_record({**spec.__dict__, "duration_s": 2.9})


def test_three_second_episode_has_31_points_and_720_substeps():
    rows = validate_episode_records(_records(30), expected_cycles=30)
    summary = summarize_episode(rows)
    assert len(rows) == 31
    assert summary["action_cycles"] == 30
    assert summary["physics_substeps"] == 720
    assert summary["duration_s"] == pytest.approx(3.0)


def test_three_hundred_second_episode_has_3001_points_and_72000_substeps():
    rows = validate_episode_records(_records(3000), expected_cycles=3000)
    summary = summarize_episode(rows)
    assert len(rows) == 3001
    assert summary["physics_substeps"] == 72_000
    assert summary["duration_s"] == pytest.approx(300.0)
    assert summary["C0_reachable"] == pytest.approx(0.1)
    assert summary["C_final_reachable"] == pytest.approx(0.5)
    assert summary["delta_reachable"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda rows: rows.pop(10), "aligned points"),
        (lambda rows: rows[10].update(boundary_index=9), "indices"),
        (lambda rows: rows[10].update(sim_time_s=1.05), "timestamps"),
        (lambda rows: rows[10].update(actor_rgb_frame=108), "frames"),
        (lambda rows: rows[10].update(coverage_rgb_frame=999), "frames differ"),
        (lambda rows: rows[10].update(reachable_coverage_fraction=0.0), "decreased"),
        (lambda rows: rows[10].update(finite=False), "non-finite"),
        (lambda rows: rows[10].update(terminated=True), "terminated"),
        (lambda rows: rows[10].update(physics_substeps=23), "substep"),
    ),
)
def test_incomplete_or_misaligned_data_is_rejected_not_repaired(mutate, message):
    rows = deepcopy(_records(30))
    mutate(rows)
    with pytest.raises(EpisodeProtocolError, match=message):
        validate_episode_records(rows, expected_cycles=30)


def test_summary_reports_mode_mix_alpha_auc_and_motion():
    summary = summarize_episode(_records(30))
    assert sum(summary["mode_counts"].values()) == 30
    assert sum(summary["mode_fractions"].values()) == pytest.approx(1.0)
    assert summary["non_hold_alpha_mean"] == pytest.approx(0.5)
    assert summary["normalized_reachable_auc"] == pytest.approx(0.3)
    assert summary["total_com_displacement_m"] == pytest.approx(0.03)
    assert summary["maximum_linear_speed_m_s"] == pytest.approx(0.01)
    assert summary["maximum_angular_speed_rad_s"] == pytest.approx(0.1)
    assert summary["rgb_frame_unique"] is True
    assert summary["coverage_monotonic"] is True
