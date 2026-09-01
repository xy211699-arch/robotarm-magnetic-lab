from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotarm_magnetic_lab.baselines.random_baseline_comparison import (
    preserve_best_snapshot_images,
    select_best_entries,
)
from robotarm_magnetic_lab.baselines.random_policies import (
    COMPARISON_SNAPSHOT_TIMES_S,
    FROZEN_VALIDATION_POSE_IDS,
    POLICY_IDS,
    load_random_baseline_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/task009c/random_baseline_20pose_comparison_v1.json"


def _entries():
    return [
        {
            "episode_id": f"comparison-{pose_id}-{policy_id.lower()}",
            "policy_id": policy_id,
            "pose_id": pose_id,
            "C_final_reachable": 0.5 + pose_index / 1000.0 + policy_index / 10000.0,
        }
        for policy_index, policy_id in enumerate(POLICY_IDS, start=1)
        for pose_index, pose_id in enumerate(FROZEN_VALIDATION_POSE_IDS)
    ]


def test_config_expands_exact_shared_twenty_pose_matrix():
    config = load_random_baseline_config(CONFIG)
    assert tuple(config["validation_pose_ids"]) == FROZEN_VALIDATION_POSE_IDS
    assert tuple(config["candidate_times_s"]) == COMPARISON_SNAPSHOT_TIMES_S
    assert len(config["formal_episodes"]) == 140
    for policy_id in POLICY_IDS:
        rows = [row for row in config["formal_episodes"] if row["policy_id"] == policy_id]
        assert tuple(row["pose_id"] for row in rows) == FROZEN_VALIDATION_POSE_IDS
        assert all(row["action_cycles"] == 3000 for row in rows)


def test_config_uses_original_deterministic_seed_mapping():
    config = load_random_baseline_config(CONFIG)
    row = next(
        row for row in config["formal_episodes"]
        if row["pose_id"] == "validation-0097" and row["policy_id"] == "R6"
    )
    assert row["environment_seed"] == 950097
    assert row["policy_seed"] == 966097


def test_best_selection_is_per_policy_and_overall():
    per_policy, overall = select_best_entries(_entries())
    assert set(per_policy) == set(POLICY_IDS)
    assert all(row["pose_id"] == "validation-0097" for row in per_policy.values())
    assert overall["policy_id"] == "R7"
    assert overall["pose_id"] == "validation-0097"


def test_best_selection_rejects_incomplete_pose_matrix():
    with pytest.raises(ValueError, match="exactly 20"):
        select_best_entries(_entries()[:-1])


def test_preserves_eleven_images_for_each_policy_and_overall(tmp_path: Path):
    entries = _entries()
    for row in entries:
        directory = tmp_path / "coverage" / row["episode_id"]
        directory.mkdir(parents=True)
        for index, second in enumerate(COMPARISON_SNAPSHOT_TIMES_S, start=1):
            (directory / f"snapshot_{index:04d}_candidate_{second:03d}s.png").write_bytes(
                f"{row['episode_id']}:{second}".encode()
            )
            (directory / f"snapshot_{index:04d}_candidate_{second:03d}s.json").write_text("{}")
    manifest_path = preserve_best_snapshot_images(
        tmp_path, entries, COMPARISON_SNAPSHOT_TIMES_S
    )
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["artifacts"]) == 8 * 11
    assert len(list((tmp_path / "best_pose_snapshots").glob("*/*.png"))) == 8 * 11
    assert not list((tmp_path / "coverage").glob("*/snapshot_*_candidate_*s.png"))
