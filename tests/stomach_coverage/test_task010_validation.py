from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stomach_coverage/summarize_task010_validation.py"
CHECKPOINT_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stomach_coverage/validate_task010_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("task010_validation_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _rows():
    return [
        {"pose_id": pose_id, "formal_steps": 1200, "checkpoint_sha256": "cp", "config_sha256": "cfg",
         "final_coverage": 0.1, "total_reward": 1.0, "mean_alpha": 0.5}
        for pose_id in MODULE.VALIDATION_POSE_IDS
    ]


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_validation_uses_exact_twenty_pose_ids():
    assert MODULE.VALIDATION_POSE_IDS == (
        "validation-0006", "validation-0011", "validation-0015", "validation-0017",
        "validation-0019", "validation-0035", "validation-0040", "validation-0042",
        "validation-0045", "validation-0046", "validation-0051", "validation-0058",
        "validation-0060", "validation-0063", "validation-0067", "validation-0068",
        "validation-0069", "validation-0092", "validation-0095", "validation-0097",
    )
    assert tuple(map(len, MODULE.validation_batches())) == (12, 8)


def test_validation_keeps_mutable_environment_outside_inference_mode():
    source = CHECKPOINT_SCRIPT.read_text(encoding="utf-8")
    assert "with torch.inference_mode()" not in source
    assert "with torch.no_grad():\n                        action = actor(" in source
    assert "observations, reward, terminated, truncated, step_extras = env.step(action)" in source


def test_validation_trajectory_uses_authoritative_accumulator_snapshot():
    source = CHECKPOINT_SCRIPT.read_text(encoding="utf-8")
    assert "runtime._snapshot(" in source
    assert "runtime.reachable_accumulator" in source
    assert "if terminal_boundary" in source
    assert "terminated[: len(batch)] | truncated[: len(batch)]" in source
    assert "latest = env._task009d0_coverage_runtime.latest_update" not in source
    assert "env._task010_recovery_tracker.previous_coverage" not in source


def test_summary_accepts_only_complete_unique_frozen_set(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    _write(path, _rows())
    summary = MODULE.summarize(path, checkpoint_sha256="cp", config_sha256="cfg")
    assert summary["pose_count"] == 20 and summary["batch_sizes"] == [12, 8]
    duplicate = _rows()
    duplicate[-1] = duplicate[0]
    _write(path, duplicate)
    with pytest.raises(ValueError, match="twenty unique"):
        MODULE.summarize(path, checkpoint_sha256="cp", config_sha256="cfg")


def test_summary_rejects_wrong_horizon_or_hash(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    rows = _rows(); rows[0]["formal_steps"] = 1199
    _write(path, rows)
    with pytest.raises(ValueError, match="1200"):
        MODULE.summarize(path, checkpoint_sha256="cp", config_sha256="cfg")
    rows = _rows(); rows[0]["checkpoint_sha256"] = "wrong"
    _write(path, rows)
    with pytest.raises(ValueError, match="checkpoint"):
        MODULE.summarize(path, checkpoint_sha256="cp", config_sha256="cfg")


def test_validation_curve_loader_requires_twenty_monotonic_1201_point_rows(tmp_path: Path):
    plot_script = Path(__file__).resolve().parents[2] / "scripts/stomach_coverage/plot_task010_validation_coverage.py"
    spec = importlib.util.spec_from_file_location("task010_validation_plot", plot_script)
    plot_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(plot_module)
    path = tmp_path / "coverage.jsonl"
    rows = [
        {"pose_id": f"validation-{index:04d}", "coverage_fraction": np.linspace(0.1, 0.9, 1201).tolist()}
        for index in range(20)
    ]
    _write(path, rows)
    assert plot_module.load_mean(path).shape == (1201,)
    rows[0]["coverage_fraction"][2] = 0.0
    _write(path, rows)
    with pytest.raises(ValueError, match="not monotonic"):
        plot_module.load_mean(path)


def test_checkpoint_validator_exposes_four_visual_conditions():
    source = CHECKPOINT_SCRIPT.read_text(encoding="utf-8")
    assert "--visual-condition" in source
    assert "normal" in source
    assert "blind" in source
    assert "donor" in source
    assert "first_frame" in source


def test_donor_validation_uses_cyclic_other_pose_and_target_actions():
    source = CHECKPOINT_SCRIPT.read_text(encoding="utf-8")
    assert "experiment.donor_pose_by_target[pose_id]" in source
    assert '"previous_action_source": "target_environment"' in source
    assert '"donor_pose_id"' in source


def test_first_frame_intervention_is_created_per_validation_batch():
    source = CHECKPOINT_SCRIPT.read_text(encoding="utf-8")
    assert "Task010VisualIntervention(" in source
    assert '"first_frame", num_envs=len(batch)' in source
