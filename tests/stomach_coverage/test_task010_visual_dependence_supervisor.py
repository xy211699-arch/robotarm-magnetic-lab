from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stomach_coverage"))

import task010_visual_dependence_supervisor as supervisor  # noqa: E402


FAKE_STAGE = ROOT / "tests/fixtures/task010_visual_dependence_fake_stage.py"


def _b0_run_dir(tmp_path):
    root = tmp_path / "b0"
    seeds = {}
    for seed in (991001, 991002, 991003):
        for update in (750, 1000):
            path = root / "seeds" / f"seed_{seed}" / "training" / "checkpoints" / f"update_{update:04d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
        final = root / "seeds" / f"seed_{seed}" / "training/checkpoints/update_1000.pt"
        seeds[str(seed)] = {
            "state": "validated",
            "training_complete": True,
            "validation_complete": True,
            "latest_checkpoint": str(final),
            "latest_checkpoint_sha256": supervisor._sha256(final),
        }
    (root / "status.json").write_text(
        json.dumps({"state": "completed", "seeds": seeds}), encoding="utf-8"
    )
    return root


def _b1_run_dir(tmp_path):
    root = tmp_path / "b1"
    stages = {}
    for seed in (991001, 991002, 991003):
        checkpoints = root / "training" / "blind" / f"seed_{seed}" / "checkpoints"
        checkpoints.mkdir(parents=True)
        for update in (750, 1000):
            (checkpoints / f"update_{update:04d}.pt").write_bytes(
                f"blind-{seed}-{update}".encode()
            )
        final = checkpoints / "update_1000.pt"
        stages[f"train_blind_seed_{seed}"] = {
            "state": "completed",
            "completion_audit": {
                "complete": True,
                "observed_update": 1000,
                "latest_checkpoint": str(final),
                "checkpoint_sha256": supervisor._sha256(final),
            },
        }
    (root / "status.json").write_text(
        json.dumps({"state": "completed_training", "stages": stages}),
        encoding="utf-8",
    )
    return root


def _start(tmp_path, monkeypatch, *, fail_stage=None, delay="0.02", reuse_b1=False):
    monkeypatch.setenv("TASK010_VISUAL_DEPENDENCE_TEST_MODE", "1")
    monkeypatch.setenv("TASK010_VISUAL_DEPENDENCE_FAKE_DELAY", delay)
    if fail_stage is not None:
        monkeypatch.setenv("TASK010_VISUAL_DEPENDENCE_FAKE_FAIL_STAGE", fail_stage)
    command = [
            "start",
            "--config",
            str(ROOT / "configs/task010/visual_dependence_v1.json"),
            "--b0-run-dir",
            str(_b0_run_dir(tmp_path)),
            "--artifact-root",
            str(tmp_path / "artifact_root"),
            "--test-driver",
            str(FAKE_STAGE),
    ]
    if reuse_b1:
        command += ["--b1-run-dir", str(_b1_run_dir(tmp_path))]
    args = supervisor._parser().parse_args(command)
    return supervisor._start(args)


def test_fake_pipeline_has_exact_frozen_stage_order():
    names = supervisor.stage_names()
    expected = []
    for seed in (991001, 991002, 991003):
        expected.append(f"train_blind_seed_{seed}")
    for seed in (991001, 991002, 991003):
        for condition in ("normal", "blind", "donor", "first_frame"):
            expected.append(f"validate_update750_{condition}_seed_{seed}")
    for seed in (991001, 991002, 991003):
        for condition in ("normal", "blind"):
            expected.append(f"validate_update1000_{condition}_seed_{seed}")
    expected.extend(("summarize", "audit_artifacts"))
    assert names == tuple(expected)


def test_start_accepts_launcher_kit_args():
    args = supervisor._parser().parse_args(
        [
            "start",
            "--config",
            str(ROOT / "configs/task010/visual_dependence_v1.json"),
            "--b0-run-dir",
            "/tmp/unused-b0",
            "--artifact-root",
            "/tmp/unused-artifacts",
            "--kit_args=--/UJITSO/enabled=false",
        ]
    )
    assert args.command == "start"


def test_start_parser_accepts_completed_b1_run():
    args = supervisor._parser().parse_args(
        [
            "start",
            "--b0-run-dir",
            "/tmp/b0",
            "--b1-run-dir",
            "/tmp/b1",
        ]
    )
    assert args.b1_run_dir == Path("/tmp/b1")


def test_completed_b1_audit_freezes_both_checkpoint_hashes(tmp_path):
    root = _b1_run_dir(tmp_path)
    audit = supervisor.audit_completed_blind_run(root)
    assert set(audit) == {"991001", "991002", "991003"}
    for seed in audit:
        assert set(audit[seed]) == {"750", "1000"}
        for record in audit[seed].values():
            path = Path(record["path"])
            assert path.is_file()
            assert record["sha256"] == supervisor._sha256(path)


def test_completed_b1_audit_rejects_incomplete_training(tmp_path):
    root = _b1_run_dir(tmp_path)
    status = json.loads((root / "status.json").read_text())
    status["state"] = "paused_on_error"
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ValueError, match="completed_training"):
        supervisor.audit_completed_blind_run(root)


def test_completed_b0_audit_freezes_both_checkpoint_hashes(tmp_path):
    root = _b0_run_dir(tmp_path)
    audit = supervisor.audit_completed_b0_run(root)
    assert set(audit) == {"991001", "991002", "991003"}
    assert all(set(records) == {"750", "1000"} for records in audit.values())
    assert all(
        record["sha256"] == supervisor._sha256(Path(record["path"]))
        for records in audit.values()
        for record in records.values()
    )


def test_completed_b0_audit_rejects_incomplete_run(tmp_path):
    root = _b0_run_dir(tmp_path)
    status = json.loads((root / "status.json").read_text())
    status["seeds"]["991002"]["validation_complete"] = False
    (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ValueError, match="991002"):
        supervisor.audit_completed_b0_run(root)


def test_reused_b1_identity_change_is_persisted_as_paused_error(tmp_path):
    b1 = _b1_run_dir(tmp_path)
    checkpoints = supervisor.audit_completed_blind_run(b1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "b1_run_dir": str(b1),
        "b1_status_sha256": supervisor._sha256(b1 / "status.json"),
        "b1_checkpoints": checkpoints,
        "test_driver": str(FAKE_STAGE),
    }
    supervisor._atomic_json(run_dir / "manifest.json", manifest)
    supervisor._atomic_json(
        run_dir / "status.json",
        supervisor._initial_state(run_dir, "run", reused_blind=checkpoints),
    )
    Path(checkpoints["991001"]["750"]["path"]).write_bytes(b"changed")
    assert supervisor._worker(run_dir, continuation=False) == 1
    state = supervisor._read_json(run_dir / "status.json")
    assert state["state"] == "paused_on_error"
    assert "changed" in state["error_summary"]


def test_formal_start_rejects_tracked_worktree_modifications(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: " M docs/PROJECT_RUN_LOG.md\n?? scripts/unrelated.py\n",
    )
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        supervisor._require_clean_tracked_worktree()


def test_validation_stage_uses_b0_for_normal_and_b1_for_blind(tmp_path):
    run_dir = tmp_path / "run"
    b0 = tmp_path / "b0"
    manifest = {
        "config": str(ROOT / "configs/task010/visual_dependence_v1.json"),
        "base_config": str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
        "b0_run_dir": str(b0),
    }
    normal = supervisor._stage_command(manifest, "validate_update750_normal_seed_991001", run_dir, 1)
    blind = supervisor._stage_command(manifest, "validate_update750_blind_seed_991001", run_dir, 1)
    assert "--checkpoint" in normal and "--checkpoint" in blind
    assert normal[normal.index("--checkpoint") + 1].startswith(str(b0))
    assert blind[blind.index("--checkpoint") + 1].startswith(str(run_dir / "training" / "blind"))
    assert "--save-full-telemetry" in normal
    assert "--save-full-telemetry" in blind


def test_validation_stage_uses_reused_b1_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    b1 = _b1_run_dir(tmp_path)
    manifest = {
        "config": str(ROOT / "configs/task010/visual_dependence_v1.json"),
        "base_config": str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
        "b0_run_dir": str(tmp_path / "b0"),
        "b1_run_dir": str(b1),
        "b1_checkpoints": supervisor.audit_completed_blind_run(b1),
    }
    command = supervisor._stage_command(
        manifest, "validate_update750_blind_seed_991001", run_dir, 1
    )
    assert command[command.index("--checkpoint") + 1] == str(
        b1 / "training/blind/seed_991001/checkpoints/update_0750.pt"
    )


def test_sensitivity_validation_uses_b0_for_normal_and_b1_for_blind(tmp_path):
    run_dir = tmp_path / "run"
    b0 = tmp_path / "b0"
    manifest = {
        "config": str(ROOT / "configs/task010/visual_dependence_v1.json"),
        "base_config": str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
        "b0_run_dir": str(b0),
    }
    normal = supervisor._stage_command(manifest, "validate_update1000_normal_seed_991002", run_dir, 1)
    blind = supervisor._stage_command(manifest, "validate_update1000_blind_seed_991002", run_dir, 1)
    assert normal[normal.index("--checkpoint") + 1].startswith(str(b0))
    assert blind[blind.index("--checkpoint") + 1].startswith(str(run_dir / "training" / "blind"))


def test_training_retry_uses_latest_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = (
        run_dir / "training" / "blind" / "seed_991001" / "checkpoints" / "update_0950.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        "config": str(ROOT / "configs/task010/visual_dependence_v1.json"),
        "base_config": str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
        "b0_run_dir": "/tmp/unused-b0",
    }
    command = supervisor._stage_command(
        manifest,
        "train_blind_seed_991001",
        run_dir,
        2,
        resume_checkpoint=checkpoint,
    )
    assert "--resume-checkpoint" in command
    assert command[command.index("--resume-checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--max-updates") + 1] == "50"


def test_training_completion_requires_update_1000(tmp_path):
    run_dir = tmp_path / "run"
    metric = run_dir / "training" / "blind" / "seed_991001" / "metrics.jsonl"
    metric.parent.mkdir(parents=True)
    metric.write_text(
        '{"update": 950}\n',
        encoding="utf-8",
    )
    assert not supervisor._training_stage_is_complete(run_dir, "991001")


def test_repair_reopens_completed_training_without_update_1000(tmp_path):
    run_dir = tmp_path / "run"
    state = {
        "stages": {
            "train_blind_seed_991001": {
                "state": "completed",
                "attempts": 1,
            }
        }
    }
    assert supervisor._repair_training_stage_states(run_dir, state)
    assert state["stages"]["train_blind_seed_991001"]["state"] == "paused_on_error"


def test_training_progress_reads_latest_metric_and_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    training_dir = run_dir / "training" / "blind" / "seed_991001"
    training_dir.mkdir(parents=True)
    (training_dir / "metrics.jsonl").write_text(
        '{"update": 958, "total_transitions": 735360, "transitions_per_second": 36.2}\n',
        encoding="utf-8",
    )
    checkpoint = training_dir / "checkpoints" / "update_0950.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    progress = supervisor._training_progress(run_dir, "train_blind_seed_991001")
    assert progress == {
        "seed": 991001,
        "update": 958,
        "total_transitions": 735360,
        "transitions_per_second": 36.2,
        "latest_checkpoint": str(checkpoint),
    }


def test_validation_progress_counts_completed_pose_records(tmp_path):
    run_dir = tmp_path / "run"
    output = run_dir / "validation/update750/normal/seed_991001"
    output.mkdir(parents=True)
    (output / "pose_records.jsonl").write_text(
        "".join('{"pose_id": "p"}\n' for _ in range(12)), encoding="utf-8"
    )
    progress = supervisor._validation_progress(
        run_dir, "validate_update750_normal_seed_991001"
    )
    assert progress == {
        "update": 750,
        "condition": "normal",
        "seed": 991001,
        "poses_complete": 12,
        "poses_total": 20,
    }


def test_reused_b1_pipeline_skips_all_training_stages(tmp_path, monkeypatch):
    started = _start(tmp_path, monkeypatch, reuse_b1=True, delay="0.001")
    run_dir = Path(started["run_dir"])
    deadline = time.time() + 10
    state = None
    while time.time() < deadline:
        state = supervisor._read_json(run_dir / "status.json")
        if state.get("state") == "completed":
            break
        time.sleep(0.02)
    assert state is not None and state["state"] == "completed"
    events = [
        json.loads(line)
        for line in (run_dir / "fake_stage_events.jsonl").read_text().splitlines()
    ]
    assert events
    assert not any(row["stage"].startswith("train_blind_seed_") for row in events)
    assert events[0]["stage"] == "validate_update750_normal_seed_991001"
    for seed in (991001, 991002, 991003):
        record = state["stages"][f"train_blind_seed_{seed}"]
        assert record["state"] == "completed"
        assert record["reused"] is True
        assert record["attempts"] == 0


def test_start_returns_while_worker_remains_alive(tmp_path, monkeypatch):
    started = _start(tmp_path, monkeypatch, delay="1.0")
    assert started["state"] == "queued"
    pid = started["worker_pid"]
    try:
        assert supervisor._pid_alive(pid)
        assert Path(started["run_dir"], "status.json").is_file()
    finally:
        os.kill(pid, signal.SIGTERM)


def test_failure_pauses_without_retry_or_next_stage(tmp_path, monkeypatch):
    started = _start(
        tmp_path,
        monkeypatch,
        fail_stage="train_blind_seed_991002",
        delay="0.01",
    )
    run_dir = Path(started["run_dir"])
    deadline = time.time() + 10
    state = None
    while time.time() < deadline:
        state = supervisor._read_json(run_dir / "status.json")
        if state.get("state") == "paused_on_error":
            break
        time.sleep(0.05)
    assert state is not None and state["state"] == "paused_on_error"
    assert state["stages"]["train_blind_seed_991002"]["attempts"] == 1
    assert state["stages"]["train_blind_seed_991003"]["state"] == "queued"
