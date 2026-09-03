from __future__ import annotations

import os
from pathlib import Path
import signal
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
