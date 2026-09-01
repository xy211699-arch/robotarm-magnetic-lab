from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stomach_coverage/plot_task010_four_seeds_vs_random.py"
SPEC = importlib.util.spec_from_file_location("task010_four_seed_plot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_model_loader_requires_ordered_twenty_pose_1201_curves(tmp_path: Path):
    path = tmp_path / "model.jsonl"
    rows = [
        {"pose_id": pose_id, "coverage_fraction": np.linspace(0.01, 0.9, 1201).tolist()}
        for pose_id in MODULE.POSE_IDS
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert MODULE.load_model_mean(path).shape == (1201,)
    rows.pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="20 poses"):
        MODULE.load_model_mean(path)


def test_random_loader_refuses_incomplete_twenty_pose_matrix(tmp_path: Path):
    (tmp_path / "episodes").mkdir()
    with pytest.raises(ValueError, match="missing 20"):
        MODULE.load_random_mean(tmp_path, "R1")


def test_random_loader_uses_exact_first_1201_of_3001_boundaries(tmp_path: Path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    for pose_id in MODULE.POSE_IDS:
        path = episodes / f"comparison-{pose_id}-r1.jsonl"
        rows = [
            {"policy_id": "R1", "pose_id": pose_id, "boundary_index": index,
             "sim_time_s": index / 10.0, "reachable_coverage_fraction": index / 3000.0}
            for index in range(3001)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    curve, paths = MODULE.load_random_mean(tmp_path, "R1")
    assert len(paths) == 20 and curve.shape == (1201,)
    assert curve[-1] == pytest.approx(0.4)
