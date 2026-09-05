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

import task010_blind_training_supervisor_v2 as supervisor  # noqa: E402


FAKE_CHILD = ROOT / "tests/fixtures/task010_blind_training_fake_child.py"


def _start(tmp_path, monkeypatch, *, delay="0.01", fail_seed=None, fail_mode=None):
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("TASK010_V2_TEST_MODE", "1")
    monkeypatch.setenv("TASK010_V2_FAKE_TRACE", str(trace))
    monkeypatch.setenv("TASK010_V2_FAKE_DELAY", delay)
    if fail_seed is not None:
        monkeypatch.setenv("TASK010_V2_FAKE_FAIL_SEED", str(fail_seed))
    if fail_mode is not None:
        monkeypatch.setenv("TASK010_V2_FAKE_FAIL_MODE", fail_mode)
    args = supervisor.parser().parse_args(
        [
            "start",
            "--base-config",
            str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
            "--visual-config",
            str(ROOT / "configs/task010/visual_dependence_v1.json"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--test-driver",
            str(FAKE_CHILD),
        ]
    )
    return supervisor.start(args), trace


def _wait_terminal(run_dir: Path, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if state["state"] in {"completed_training", "paused_on_error"}:
            return state
        time.sleep(0.02)
    raise AssertionError("supervisor did not reach a terminal state")


def _wait_pid_dead(pid: int, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not supervisor.pid_alive(pid):
            return
        time.sleep(0.01)
    raise AssertionError(f"PID {pid} did not exit")


def test_stage_order_contains_only_three_training_seeds():
    assert supervisor.SEEDS == (991001, 991002, 991003)
    assert supervisor.stage_names() == (
        "train_blind_seed_991001",
        "train_blind_seed_991002",
        "train_blind_seed_991003",
    )


def test_training_command_is_frozen_and_contains_no_validation(tmp_path):
    manifest = {
        "base_config": str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
        "git": {"commit": "a" * 40},
        "base_config_sha256": "b" * 64,
        "visual_config_sha256": "c" * 64,
        "dependency_audit": {"path": str(ROOT / "artifacts/task010_cnn_gru/gate0/prerequisites.json")},
        "test_driver": None,
    }
    command = supervisor.training_command(manifest, tmp_path, seed=991001, resume_checkpoint=None)
    joined = " ".join(command)
    assert command[command.index("--visual-condition") + 1] == "blind"
    assert command[command.index("--max-updates") + 1] == "1000"
    assert command[command.index("--save-interval") + 1] == "50"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert "validate_task010_checkpoint.py" not in joined


def test_start_returns_while_detached_worker_is_alive(tmp_path, monkeypatch):
    started, _trace = _start(tmp_path, monkeypatch, delay="1.0")
    try:
        assert started["state"] == "queued"
        assert supervisor.pid_alive(started["worker_pid"])
        assert Path(started["run_dir"], "manifest.json").is_file()
    finally:
        os.kill(started["worker_pid"], signal.SIGTERM)


def test_fake_pipeline_runs_seeds_sequentially_and_stops_before_validation(tmp_path, monkeypatch):
    started, trace = _start(tmp_path, monkeypatch)
    run_dir = Path(started["run_dir"])
    state = _wait_terminal(run_dir)
    assert state["state"] == "completed_training"
    assert [state["stages"][name]["state"] for name in supervisor.stage_names()] == [
        "completed",
        "completed",
        "completed",
    ]
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [(row["event"], row["seed"]) for row in rows] == [
        ("start", 991001),
        ("end", 991001),
        ("start", 991002),
        ("end", 991002),
        ("start", 991003),
        ("end", 991003),
    ]
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "validate_task010" not in events
    assert "summarize" not in events


def test_zero_exit_before_update_1000_pauses_and_does_not_start_next_seed(tmp_path, monkeypatch):
    started, trace = _start(
        tmp_path,
        monkeypatch,
        fail_seed=991002,
        fail_mode="incomplete_zero",
    )
    state = _wait_terminal(Path(started["run_dir"]))
    assert state["state"] == "paused_on_error"
    assert state["current_stage"] == "train_blind_seed_991002"
    assert state["error"]["error_type"] == "incomplete_zero_exit"
    assert state["error"]["observed_update"] == 437
    assert state["error"]["expected_update"] == 1000
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert not any(row["seed"] == 991003 for row in rows)


def test_nonzero_exit_pauses_with_child_exit_code(tmp_path, monkeypatch):
    started, _trace = _start(tmp_path, monkeypatch, fail_seed=991001, fail_mode="nonzero")
    state = _wait_terminal(Path(started["run_dir"]))
    assert state["state"] == "paused_on_error"
    assert state["error"]["error_type"] == "child_nonzero_exit"
    assert state["error"]["child_exit_code"] == 23


def test_start_rejects_a_second_active_run(tmp_path, monkeypatch):
    started, _trace = _start(tmp_path, monkeypatch, delay="1.0")
    try:
        args = supervisor.parser().parse_args(
            [
                "start",
                "--base-config",
                str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
                "--visual-config",
                str(ROOT / "configs/task010/visual_dependence_v1.json"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--test-driver",
                str(FAKE_CHILD),
            ]
        )
        with pytest.raises(RuntimeError, match="already active"):
            supervisor.start(args)
    finally:
        os.kill(started["worker_pid"], signal.SIGTERM)


def test_formal_start_rejects_dirty_tracked_worktree(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: " M docs/PROJECT_RUN_LOG.md\n",
    )
    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        supervisor.git_identity(require_clean=True)


def test_resume_command_from_update_0400_requests_exactly_600_updates(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "update_0400.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"placeholder")
    manifest = {
        "git": {"commit": "a" * 40},
        "base_config_sha256": "b" * 64,
        "visual_config_sha256": "c" * 64,
        "dependency_audit": {"sha256": "d" * 64},
        "test_driver": str(FAKE_CHILD),
    }
    command = supervisor.training_command(
        manifest,
        tmp_path,
        seed=991002,
        resume_checkpoint=checkpoint,
    )
    assert command[command.index("--max-updates") + 1] == "600"
    assert command[command.index("--resume-checkpoint") + 1].endswith("update_0400.pt")


def test_continue_resumes_failed_seed_and_never_restarts_completed_seed(tmp_path, monkeypatch):
    started, trace = _start(
        tmp_path,
        monkeypatch,
        fail_seed=991002,
        fail_mode="incomplete_zero",
    )
    run_dir = Path(started["run_dir"])
    paused = _wait_terminal(run_dir)
    _wait_pid_dead(started["worker_pid"])
    assert paused["stages"]["train_blind_seed_991001"]["state"] == "completed"
    monkeypatch.delenv("TASK010_V2_FAKE_FAIL_SEED")
    monkeypatch.delenv("TASK010_V2_FAKE_FAIL_MODE")
    args = supervisor.parser().parse_args(["continue", "--run-dir", str(run_dir)])
    continued = supervisor.continue_run(args)
    assert continued["state"] == "queued"
    completed = _wait_terminal(run_dir)
    assert completed["state"] == "completed_training"
    starts = [row for row in map(json.loads, trace.read_text().splitlines()) if row["event"] == "start"]
    assert [row["seed"] for row in starts] == [991001, 991002, 991002, 991003]
    assert starts[2]["resume_checkpoint"].endswith("update_0400.pt")
    assert starts[2]["max_updates"] == 600


def test_continue_refuses_non_paused_run(tmp_path, monkeypatch):
    started, _trace = _start(tmp_path, monkeypatch, delay="1.0")
    try:
        args = supervisor.parser().parse_args(["continue", "--run-dir", started["run_dir"]])
        with pytest.raises(RuntimeError, match="paused_on_error"):
            supervisor.continue_run(args)
    finally:
        os.kill(started["worker_pid"], signal.SIGTERM)


def test_continue_refuses_corrupt_latest_checkpoint(tmp_path, monkeypatch):
    started, _trace = _start(
        tmp_path,
        monkeypatch,
        fail_seed=991001,
        fail_mode="incomplete_zero",
    )
    run_dir = Path(started["run_dir"])
    paused = _wait_terminal(run_dir)
    _wait_pid_dead(started["worker_pid"])
    Path(paused["error"]["latest_checkpoint"]).write_bytes(b"corrupt")
    monkeypatch.delenv("TASK010_V2_FAKE_FAIL_SEED")
    monkeypatch.delenv("TASK010_V2_FAKE_FAIL_MODE")
    args = supervisor.parser().parse_args(["continue", "--run-dir", str(run_dir)])
    with pytest.raises(RuntimeError, match="checkpoint"):
        supervisor.continue_run(args)


def test_effective_status_reports_dead_coordinator_without_mutating_status(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    persisted = {
        "state": "training",
        "worker_pid": 999_999_999,
        "started_epoch_s": time.time() - 10.0,
        "heartbeat_epoch_s": time.time(),
        "error": None,
    }
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps(persisted), encoding="utf-8")
    observed = supervisor._effective_status(run_dir)
    assert observed["state"] == "paused_on_error"
    assert observed["error"]["error_type"] == "coordinator_stale"
    assert json.loads(status_path.read_text(encoding="utf-8")) == persisted


def test_continue_materializes_dead_coordinator_pause_before_resume(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "training/blind/seed_991001/checkpoints/update_0400.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parents[1] / "metrics.jsonl").write_text(
        json.dumps(
            {
                "update": 437,
                "time_ns": time.time_ns(),
                "transitions_per_second": 38.0,
                "all_finite": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {"test_driver": str(FAKE_CHILD), "git": {"commit": "a" * 40}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = supervisor._initial_state(run_dir, "test-run")
    stage = supervisor.stage_names()[0]
    state.update(state="training", current_stage=stage, worker_pid=999_999_999)
    state["stages"][stage]["state"] = "running"
    (run_dir / "status.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(supervisor, "_identity_matches", lambda value: True)
    monkeypatch.setattr(
        supervisor,
        "_validate_resume_checkpoint",
        lambda path, value, seed: {
            "path": str(path),
            "update": 400,
            "remaining_updates": 600,
            "sha256": "f" * 64,
            "seed": seed,
        },
    )
    monkeypatch.setattr(supervisor, "_spawn_worker", lambda *args, **kwargs: 12345)
    args = supervisor.parser().parse_args(["continue", "--run-dir", str(run_dir)])
    result = supervisor.continue_run(args)
    assert result["worker_pid"] == 12345
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events] == ["coordinator_stale_materialized", "continue_authorized"]
