from __future__ import annotations

import json
import math
from pathlib import Path
import time

import pytest
import torch

from robotarm_magnetic_lab.runtime import task010_training_health as health


EXPECTED = {
    "seed": 991001,
    "target_update": 1000,
    "git_commit": "a" * 40,
    "base_config_sha256": "b" * 64,
    "visual_dependence_config_sha256": "c" * 64,
    "visual_condition": "blind",
}


def _write_metric(path: Path, update: int, *, tps: float = 38.0, all_finite: bool = True, time_ns: int | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "robotarm_magnetic_lab.task010_update_metric",
        "update": update,
        "total_transitions": update * 768,
        "elapsed_s": 768.0 / tps,
        "time_ns": int(time_ns if time_ns is not None else time.time_ns()),
        "transitions_per_second": tps,
        "all_finite": all_finite,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=True) + "\n")
    return row


def _write_training_dir(root: Path, *, update: int = 1000, checkpoint_update: int | None = 1000):
    root.mkdir(parents=True, exist_ok=True)
    (root / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "runner_initialized",
                "seed": EXPECTED["seed"],
                "experiment_metadata": {
                    "visual_condition": "blind",
                    "base_config_sha256": EXPECTED["base_config_sha256"],
                    "visual_dependence_config_sha256": EXPECTED[
                        "visual_dependence_config_sha256"
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_metric(root / "metrics.jsonl", update)
    if checkpoint_update is not None:
        checkpoint = root / "checkpoints" / f"update_{checkpoint_update:04d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "robotarm_magnetic_lab.task010_checkpoint",
                "current_update": checkpoint_update,
                "git_commit": EXPECTED["git_commit"],
                "config_hash": EXPECTED["base_config_sha256"],
                "base_config_sha256": EXPECTED["base_config_sha256"],
                "visual_dependence_config_sha256": EXPECTED[
                    "visual_dependence_config_sha256"
                ],
                "experiment_metadata": {
                    "visual_condition": "blind",
                    "base_config_sha256": EXPECTED["base_config_sha256"],
                    "visual_dependence_config_sha256": EXPECTED[
                        "visual_dependence_config_sha256"
                    ],
                },
            },
            checkpoint,
        )
    return root


def test_checkpoint_update_parser_rejects_noncanonical_names():
    assert health.parse_checkpoint_update(Path("update_0400.pt")) == 400
    with pytest.raises(ValueError, match="checkpoint filename"):
        health.parse_checkpoint_update(Path("checkpoint-400.pt"))


def test_jsonl_reader_ignores_only_an_incomplete_trailing_fragment(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"update": 1}\n{"update": 2')
    assert health.read_complete_jsonl(path) == [{"update": 1}]
    path.write_bytes(b'{"update": 1}\nnot-json\n')
    with pytest.raises(ValueError):
        health.read_complete_jsonl(path)


def test_progress_classifies_starting_healthy_degraded_and_stalled(tmp_path):
    training = tmp_path / "training"
    assert health.progress_snapshot(training, now_epoch_s=1000.0)["health"] == "starting"

    metric_path = training / "metrics.jsonl"
    for update in range(1, 11):
        _write_metric(metric_path, update, tps=38.0, time_ns=1_000_000_000_000)
    healthy = health.progress_snapshot(training, now_epoch_s=1001.0)
    assert healthy["health"] == "healthy"
    assert healthy["median_tps_last_10"] == pytest.approx(38.0)

    metric_path.unlink()
    for update in range(1, 11):
        _write_metric(metric_path, update, tps=1.2, time_ns=1_000_000_000_000)
    degraded = health.progress_snapshot(training, now_epoch_s=1001.0)
    assert degraded["health"] == "degraded_performance"
    assert degraded["median_tps_last_10"] == pytest.approx(1.2)
    assert health.progress_snapshot(training, now_epoch_s=1301.0)["health"] == "suspected_stall"
    assert health.progress_snapshot(training, now_epoch_s=1901.0)["health"] == "critical_stall"


def test_remaining_updates_validates_formal_range():
    assert health.remaining_updates(400) == 600
    assert health.remaining_updates(1000) == 0
    with pytest.raises(ValueError, match="outside"):
        health.remaining_updates(1001)


def test_completion_audit_accepts_only_matching_update_1000(tmp_path):
    training = _write_training_dir(tmp_path / "seed")
    audit = health.audit_completion(training, EXPECTED, child_exit_code=0)
    assert audit["complete"] is True
    assert audit["error_type"] is None
    assert audit["observed_update"] == 1000
    assert audit["checkpoint_update"] == 1000
    assert len(audit["checkpoint_sha256"]) == 64


def test_completion_audit_classifies_zero_exit_before_target(tmp_path):
    training = _write_training_dir(tmp_path / "seed", update=437, checkpoint_update=400)
    audit = health.audit_completion(training, EXPECTED, child_exit_code=0)
    assert audit["complete"] is False
    assert audit["error_type"] == "incomplete_zero_exit"
    assert audit["expected_update"] == 1000
    assert audit["observed_update"] == 437
    assert audit["checkpoint_update"] == 400


def test_completion_audit_rejects_nonfinite_metric(tmp_path):
    training = _write_training_dir(tmp_path / "seed")
    with (training / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "update": 1000,
                    "all_finite": True,
                    "transitions_per_second": math.nan,
                    "time_ns": time.time_ns(),
                },
                allow_nan=True,
            )
            + "\n"
        )
    audit = health.audit_completion(training, EXPECTED, child_exit_code=0)
    assert audit["error_type"] == "non_finite_metric"


def test_completion_audit_rejects_checkpoint_identity_mismatch(tmp_path):
    training = _write_training_dir(tmp_path / "seed")
    checkpoint = training / "checkpoints" / "update_1000.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["git_commit"] = "d" * 40
    torch.save(payload, checkpoint)
    audit = health.audit_completion(training, EXPECTED, child_exit_code=0)
    assert audit["error_type"] == "identity_mismatch"


@pytest.mark.parametrize("payload", [None, b"", b"not-a-checkpoint"])
def test_completion_audit_rejects_missing_empty_or_corrupt_checkpoint(tmp_path, payload):
    training = _write_training_dir(tmp_path / "seed", checkpoint_update=None)
    checkpoint = training / "checkpoints" / "update_1000.pt"
    if payload is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(payload)
    audit = health.audit_completion(training, EXPECTED, child_exit_code=0)
    assert audit["complete"] is False
    assert audit["error_type"] in {"checkpoint_missing", "checkpoint_corrupt"}


def test_completion_audit_prioritizes_nonzero_child_exit(tmp_path):
    training = _write_training_dir(tmp_path / "seed")
    audit = health.audit_completion(training, EXPECTED, child_exit_code=23)
    assert audit["error_type"] == "child_nonzero_exit"
    assert audit["child_exit_code"] == 23
