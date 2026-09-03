from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stomach_coverage"))

from robotarm_magnetic_lab.runtime.task010_visual_dependence_config import (
    load_visual_dependence_config,
)
from summarize_task010_visual_dependence import (
    episode_metrics,
    hierarchical_paired_bootstrap,
    summarize_visual_dependence,
)


CONFIG_PATH = (
    ROOT
    / "configs/task010/visual_dependence_v1.json"
)


def test_episode_metrics_use_1200_post_action_points():
    curve = np.linspace(0.0, 1.0, 1201)
    metrics = episode_metrics(curve)
    expected_nauc = np.trapezoid(curve, dx=0.1) / 120.0
    assert metrics["nAUC_120"] == pytest.approx(expected_nauc)
    assert metrics["C30"] == pytest.approx(curve[300])
    assert metrics["C60"] == pytest.approx(curve[600])
    assert metrics["C120"] == pytest.approx(curve[1200])


def test_unreached_threshold_is_retained():
    metrics = episode_metrics(np.linspace(0.0, 0.79, 1201))
    assert metrics["time_to_80"] is None
    assert metrics["reached_80"] is False


def _rows(comparison: str):
    rows = []
    for seed in (991001, 991002, 991003):
        for pose_index in range(20):
            pose_id = f"validation-{pose_index:04d}"
            base = np.linspace(0.1, 0.95, 1201)
            if comparison == "blind":
                treatment = np.linspace(0.1, 0.70, 1201)
            else:
                treatment = np.linspace(0.1, 0.75, 1201)
            rows.append(
                {
                    "condition": "normal",
                    "seed": seed,
                    "pose_id": pose_id,
                    "update": 750,
                    "coverage_fraction": base.tolist(),
                }
            )
            rows.append(
                {
                    "condition": comparison,
                    "seed": seed,
                    "pose_id": pose_id,
                    "update": 750,
                    "coverage_fraction": treatment.tolist(),
                }
            )
    return rows


def test_hierarchical_bootstrap_pairs_pose_within_seed():
    effect = hierarchical_paired_bootstrap(
        _rows("blind"), seed=20260903, replicates=100
    )
    assert effect["independent_seed_count"] == 3
    assert effect["paired_pose_count_per_seed"] == 20
    assert effect["ci95_low"] > 0.0


def test_summarize_visual_dependence_writes_audited_outputs(tmp_path):
    config = load_visual_dependence_config(CONFIG_PATH)
    normal = [row for row in _rows("blind") if row["condition"] == "normal"]
    blind = [row for row in _rows("blind") if row["condition"] == "blind"]
    donor = [row for row in _rows("donor") if row["condition"] == "donor"]
    first_frame = [row for row in _rows("first_frame") if row["condition"] == "first_frame"]
    primary = normal + blind + donor + first_frame
    sensitivity = []
    for seed in (991001, 991002, 991003):
        for pose_index in range(20):
            pose_id = f"validation-{pose_index:04d}"
            sensitivity.append(
                {
                    "condition": "normal",
                    "seed": seed,
                    "pose_id": pose_id,
                    "update": 1000,
                    "coverage_fraction": np.linspace(0.1, 0.9, 1201).tolist(),
                }
            )
            sensitivity.append(
                {
                    "condition": "blind",
                    "seed": seed,
                    "pose_id": pose_id,
                    "update": 1000,
                    "coverage_fraction": np.linspace(0.1, 0.6, 1201).tolist(),
                }
            )
    all_rows = primary + sensitivity
    assert len(all_rows) == 360
    (tmp_path / "summary_input.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in all_rows), encoding="utf-8"
    )
    result = summarize_visual_dependence(tmp_path, config)
    assert result["status"] == "summarized"
    assert (tmp_path / "condition_metrics.csv").is_file()
    assert (tmp_path / "artifact_audit.json").is_file()
