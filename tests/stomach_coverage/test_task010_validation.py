from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/stomach_coverage/summarize_task010_validation.py"
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
