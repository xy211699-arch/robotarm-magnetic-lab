from __future__ import annotations

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
    for seed in (991001, 991002, 991003):
        for update in (750, 1000):
            path = root / "seeds" / f"seed_{seed}" / "training" / "checkpoints" / f"update_{update:04d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")
    return root


def _start(tmp_path, monkeypatch, *, fail_stage=None, delay="0.02"):
    monkeypatch.setenv("TASK010_VISUAL_DEPENDENCE_TEST_MODE", "1")
    monkeypatch.setenv("TASK010_VISUAL_DEPENDENCE_FAKE_DELAY", delay)
    if fail_stage is not None:
        monkeypatch.setenv("TASK010_VISUAL_DEPENDENCE_FAKE_FAIL_STAGE", fail_stage)
    args = supervisor._parser().parse_args(
        [
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
    )
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
