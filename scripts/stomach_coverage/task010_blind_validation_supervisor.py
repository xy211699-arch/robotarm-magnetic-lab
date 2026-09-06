#!/usr/bin/env python3
"""Run three completed Blind-GRU checkpoints on the frozen twenty poses."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from summarize_task010_validation import VALIDATION_POSE_IDS  # noqa: E402
from validate_task010_checkpoint import validate_full_telemetry_record  # noqa: E402


SEEDS = (991001, 991002, 991003)
CONTROL_HZ = 10
COVERAGE_POINTS = 1201
DEFAULT_SOURCE_RUN = Path(
    "/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_blind_training_v2/latest"
)
DEFAULT_CONFIG = REPOSITORY / "configs/task010/cnn_gru_development_v1.json"
DEFAULT_EXPERIMENT_CONFIG = REPOSITORY / "configs/task010/visual_dependence_v1.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY / "artifacts/task010_blind_validation_v2"
ACTIVE_STATES = {"queued", "validating"}
STALE_AFTER_S = 60.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_event(run_dir: Path, payload: dict) -> None:
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_ns": time.time_ns(), **payload}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _pid_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        stat = Path(f"/proc/{int(pid)}/stat")
        return not stat.is_file() or stat.read_text(encoding="utf-8").split()[2] != "Z"
    except (ProcessLookupError, PermissionError, FileNotFoundError):
        return False


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPOSITORY, text=True).strip()


def git_drift_is_allowed(paths) -> bool:
    """Allow routine run-log bookkeeping without relaxing execution inputs."""
    return set(paths) <= {"docs/PROJECT_RUN_LOG.md"}


def _git_changed_paths(base_commit: str) -> list[str]:
    values: set[str] = set()
    for command in (
        ("git", "diff", "--name-only", f"{base_commit}..HEAD"),
        ("git", "diff", "--name-only"),
        ("git", "diff", "--cached", "--name-only"),
    ):
        output = subprocess.check_output(command, cwd=REPOSITORY, text=True)
        values.update(line.strip() for line in output.splitlines() if line.strip())
    return sorted(values)


def _require_clean_tracked_worktree() -> None:
    status = subprocess.check_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=no"),
        cwd=REPOSITORY,
        text=True,
    )
    if status.strip():
        raise RuntimeError("blind validation requires a clean tracked worktree")


def discover_checkpoints(source_run: Path) -> dict[int, Path]:
    source = Path(source_run).resolve(strict=True)
    status = _read_json(source / "status.json")
    if status.get("state") != "completed_training" or status.get("error") is not None:
        raise ValueError("source Blind-GRU training run is not completed_training")
    checkpoints: dict[int, Path] = {}
    for seed in SEEDS:
        stage = status.get("stages", {}).get(f"train_blind_seed_{seed}", {})
        audit = stage.get("completion_audit") or {}
        if stage.get("state") != "completed" or not audit.get("complete"):
            raise ValueError(f"seed {seed} does not have a completed training audit")
        if int(audit.get("observed_update", -1)) != 1000:
            raise ValueError(f"seed {seed} did not reach update 1000")
        checkpoint = Path(audit.get("latest_checkpoint", "")).resolve(strict=True)
        if checkpoint.name != "update_1000.pt":
            raise ValueError(f"seed {seed} final checkpoint is not update_1000.pt")
        if _sha256(checkpoint) != audit.get("checkpoint_sha256"):
            raise ValueError(f"seed {seed} final checkpoint hash mismatch")
        checkpoints[seed] = checkpoint
    return checkpoints


def initial_state(run_dir: Path, run_id: str) -> dict:
    now = time.time()
    return {
        "schema": "robotarm_magnetic_lab.task010_blind_validation_v2_status",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state": "queued",
        "current_seed": SEEDS[0],
        "worker_pid": -1,
        "child_pid": None,
        "started_epoch_s": now,
        "heartbeat_epoch_s": now,
        "finished_epoch_s": None,
        "error_summary": None,
        "execution": {"device": "cuda:0", "sequential": True, "max_parallel_children": 1},
        "seeds": {
            str(seed): {
                "state": "queued",
                "attempts": 0,
                "pose_progress": 0,
                "output_dir": str(Path(run_dir) / f"seed_{seed}"),
                "audit": None,
            }
            for seed in SEEDS
        },
    }


def validation_command(manifest: dict, run_dir: Path, *, seed: int, attempt: int = 1) -> list[str]:
    output = Path(run_dir) / f"seed_{seed}"
    if manifest.get("test_driver"):
        return [
            sys.executable,
            str(manifest["test_driver"]),
            "--seed",
            str(seed),
            "--attempt",
            str(attempt),
            "--output-dir",
            str(output),
            "--run-dir",
            str(run_dir),
        ]
    return [
        str(REPOSITORY / "run_isaaclab.sh"),
        "-p",
        "scripts/stomach_coverage/validate_task010_checkpoint.py",
        "--config",
        str(manifest["config"]),
        "--checkpoint",
        str(manifest["checkpoints"][str(seed)]),
        "--output-dir",
        str(output),
        "--visual-condition",
        "blind",
        "--experiment-config",
        str(manifest["experiment_config"]),
        "--training-seed",
        str(seed),
        "--save-full-telemetry",
        "--device",
        "cuda:0",
    ]


def audit_validation(validation_dir: Path) -> dict:
    output = Path(validation_dir)
    pose_records = _read_jsonl(output / "pose_records.jsonl")
    trajectories = _read_jsonl(output / "coverage_trajectories.jsonl")
    telemetry = _read_jsonl(output / "telemetry_10hz.jsonl")
    expected = set(VALIDATION_POSE_IDS)
    for label, rows in (
        ("pose records", pose_records),
        ("coverage trajectories", trajectories),
        ("telemetry records", telemetry),
    ):
        ids = [str(row.get("pose_id")) for row in rows]
        if len(rows) != 20 or len(set(ids)) != 20 or set(ids) != expected:
            raise ValueError(f"validation {label} must contain twenty unique frozen poses")
    trajectory_by_pose = {row["pose_id"]: row for row in trajectories}
    telemetry_by_pose = {row["pose_id"]: row for row in telemetry}
    curves = []
    for pose_id in VALIDATION_POSE_IDS:
        curve = [float(value) for value in trajectory_by_pose[pose_id].get("coverage_fraction", ())]
        if len(curve) != COVERAGE_POINTS:
            raise ValueError(f"{pose_id} coverage must contain 1201 points")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in curve):
            raise ValueError(f"{pose_id} coverage contains invalid values")
        if any(after < before - 1.0e-12 for before, after in zip(curve, curve[1:])):
            raise ValueError(f"{pose_id} cumulative coverage is not monotonic")
        checked = validate_full_telemetry_record(telemetry_by_pose[pose_id])
        if any(abs(a - b) > 1.0e-12 for a, b in zip(curve, checked["coverage_fraction"])):
            raise ValueError(f"{pose_id} telemetry and coverage trajectory differ")
        mask = output / "final_masks" / f"{pose_id}.npz"
        if not mask.is_file() or mask.stat().st_size == 0:
            raise ValueError(f"{pose_id} final mask is missing")
        curves.append(curve)
    means = [sum(values) / 20.0 for values in zip(*curves)]
    with (output / "mean_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "mean_reachable_coverage"))
        for step, value in enumerate(means):
            writer.writerow((step / CONTROL_HZ, value))
    result = {
        "pose_count": 20,
        "coverage_points": COVERAGE_POINTS,
        "telemetry_records": 20,
        "final_masks": 20,
        "mean_final_coverage": means[-1],
        "all_finite": True,
    }
    _atomic_json(output / "validation_audit.json", result)
    return result


def _resolve_run(run_dir: Path | None, output_root: Path) -> Path:
    return (Path(run_dir) if run_dir else Path(output_root) / "latest").resolve(strict=True)


def _update_link(link: Path, target: Path) -> None:
    temporary = link.with_name(link.name + f".{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _spawn_worker(run_dir: Path, state: dict, *, continuation: bool) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--run-dir", str(run_dir)]
    if continuation:
        command.append("--continuation")
    console = (run_dir / "coordinator.log").open("ab", buffering=0)
    child = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        stdin=subprocess.DEVNULL,
        stdout=console,
        stderr=console,
        start_new_session=True,
    )
    console.close()
    state.update(worker_pid=child.pid, heartbeat_epoch_s=time.time())
    _atomic_json(run_dir / "status.json", state)
    return child.pid


def _start(args) -> dict:
    test_mode = bool(args.test_driver)
    if test_mode and os.environ.get("TASK010_BLIND_VALIDATION_TEST_MODE") != "1":
        raise PermissionError("test driver requires TASK010_BLIND_VALIDATION_TEST_MODE=1")
    if not test_mode:
        _require_clean_tracked_worktree()
    source_run = Path(args.source_run).resolve(strict=True)
    checkpoints = discover_checkpoints(source_run)
    config = Path(args.config).resolve(strict=True)
    experiment = Path(args.experiment_config).resolve(strict=True)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        current = _status_payload(latest.resolve(strict=True))
        if current.get("state") in ACTIVE_STATES:
            raise RuntimeError("a Blind-GRU validation supervisor is already active")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / run_id
    run_dir.mkdir()
    manifest = {
        "schema": "robotarm_magnetic_lab.task010_blind_validation_v2_manifest",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_training_run": str(source_run),
        "git_head": _git_head(),
        "config": str(config),
        "config_sha256": _sha256(config),
        "experiment_config": str(experiment),
        "experiment_config_sha256": _sha256(experiment),
        "checkpoints": {str(seed): str(path) for seed, path in checkpoints.items()},
        "checkpoint_sha256": {str(seed): _sha256(path) for seed, path in checkpoints.items()},
        "pose_ids": list(VALIDATION_POSE_IDS),
        "control_hz": CONTROL_HZ,
        "coverage_points": COVERAGE_POINTS,
        "test_driver": str(Path(args.test_driver).resolve()) if args.test_driver else None,
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    state = initial_state(run_dir, run_id)
    _atomic_json(run_dir / "status.json", state)
    _append_event(run_dir, {"event": "queued", "seeds": list(SEEDS)})
    worker_pid = _spawn_worker(run_dir, state, continuation=False)
    _update_link(latest, run_dir)
    result = {"state": "queued", "run_id": run_id, "run_dir": str(run_dir), "worker_pid": worker_pid}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _check_identity(manifest: dict) -> None:
    changed_paths = [] if manifest.get("test_driver") else _git_changed_paths(manifest["git_head"])
    if not git_drift_is_allowed(changed_paths):
        raise RuntimeError(
            "execution-relevant Git paths changed during Blind-GRU validation: "
            + ", ".join(changed_paths)
        )
    for key in ("config", "experiment_config"):
        if _sha256(Path(manifest[key])) != manifest[f"{key}_sha256"]:
            raise RuntimeError(f"{key} changed during Blind-GRU validation")
    for seed in SEEDS:
        if _sha256(Path(manifest["checkpoints"][str(seed)])) != manifest["checkpoint_sha256"][str(seed)]:
            raise RuntimeError(f"seed {seed} checkpoint changed during Blind-GRU validation")


def _progress(state: dict) -> int:
    seed = state.get("current_seed")
    if seed is None:
        return 20
    return _line_count(Path(state["seeds"][str(seed)]["output_dir"]) / "telemetry_10hz.jsonl")


def _pause(run_dir: Path, state: dict, seed: int, error: BaseException | str) -> int:
    message = str(error)
    state["seeds"][str(seed)].update(state="paused_on_error", pose_progress=_progress(state))
    state.update(
        state="paused_on_error",
        current_seed=seed,
        child_pid=None,
        heartbeat_epoch_s=time.time(),
        error_summary=message,
    )
    _atomic_json(run_dir / "status.json", state)
    _append_event(run_dir, {"event": "paused_on_error", "seed": seed, "error": message})
    return 1


def _aggregate(run_dir: Path, state: dict) -> Path:
    curves = []
    for seed in SEEDS:
        path = Path(state["seeds"][str(seed)]["output_dir"]) / "mean_coverage.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != COVERAGE_POINTS:
            raise ValueError(f"seed {seed} mean curve must contain 1201 rows")
        curves.append([float(row["mean_reachable_coverage"]) for row in rows])
    output = run_dir / "three_seed_mean_coverage.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("time_s", "seed_991001_mean_coverage", "seed_991002_mean_coverage", "seed_991003_mean_coverage", "blind_mean_coverage", "blind_std_coverage")
        )
        for step, values in enumerate(zip(*curves)):
            average = sum(values) / 3.0
            std = math.sqrt(sum((value - average) ** 2 for value in values) / 3.0)
            writer.writerow((step / CONTROL_HZ, *values, average, std))
    return output


def _worker(run_dir: Path, continuation: bool) -> int:
    run_dir = Path(run_dir).resolve(strict=True)
    lock = (run_dir / ".coordinator.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 73
    manifest = _read_json(run_dir / "manifest.json")
    state = _read_json(run_dir / "status.json")
    state.update(worker_pid=os.getpid(), heartbeat_epoch_s=time.time(), error_summary=None)
    _atomic_json(run_dir / "status.json", state)
    _append_event(run_dir, {"event": "worker_started", "continuation": continuation})
    try:
        for seed in SEEDS:
            record = state["seeds"][str(seed)]
            if record["state"] == "validated":
                continue
            _check_identity(manifest)
            if continuation:
                try:
                    audit = audit_validation(Path(record["output_dir"]))
                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass
                else:
                    record.update(state="validated", pose_progress=20, audit=audit)
                    _atomic_json(run_dir / "status.json", state)
                    continue
            record["attempts"] += 1
            attempt = int(record["attempts"])
            record.update(state="validating", pose_progress=0)
            state.update(state="validating", current_seed=seed, error_summary=None)
            _atomic_json(run_dir / "status.json", state)
            command = validation_command(manifest, run_dir, seed=seed, attempt=attempt)
            log = run_dir / f"seed_{seed}_validation_attempt_{attempt:02d}.log"
            _append_event(run_dir, {"event": "validation_started", "seed": seed, "attempt": attempt, "command": command})
            with log.open("ab", buffering=0) as console:
                child = subprocess.Popen(command, cwd=REPOSITORY, stdin=subprocess.DEVNULL, stdout=console, stderr=console)
                state["child_pid"] = child.pid
                poll_interval_s = 0.02 if manifest.get("test_driver") else 5.0
                while child.poll() is None:
                    state["heartbeat_epoch_s"] = time.time()
                    record["pose_progress"] = _progress(state)
                    _atomic_json(run_dir / "status.json", state)
                    time.sleep(poll_interval_s)
            state["child_pid"] = None
            _append_event(run_dir, {"event": "validation_finished", "seed": seed, "attempt": attempt, "exit_code": child.returncode})
            if child.returncode != 0:
                return _pause(run_dir, state, seed, f"validation child exited with code {child.returncode}; see {log}")
            try:
                audit = audit_validation(Path(record["output_dir"]))
            except BaseException as error:
                return _pause(run_dir, state, seed, error)
            record.update(state="validated", pose_progress=20, audit=audit)
            state.update(current_seed=None, heartbeat_epoch_s=time.time())
            _atomic_json(run_dir / "status.json", state)
        aggregate = _aggregate(run_dir, state)
        state.update(
            state="completed",
            current_seed=None,
            child_pid=None,
            heartbeat_epoch_s=time.time(),
            finished_epoch_s=time.time(),
            error_summary=None,
            aggregate_csv=str(aggregate),
            aggregate_csv_sha256=_sha256(aggregate),
        )
        _atomic_json(run_dir / "status.json", state)
        _append_event(run_dir, {"event": "completed", "aggregate_csv": str(aggregate)})
        return 0
    except BaseException as error:
        with (run_dir / "coordinator.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        seed = int(state.get("current_seed") or next(seed for seed in SEEDS if state["seeds"][str(seed)]["state"] != "validated"))
        return _pause(run_dir, state, seed, error)
    finally:
        lock.close()


def _status_payload(run_dir: Path) -> dict:
    state = _read_json(Path(run_dir) / "status.json")
    result = json.loads(json.dumps(state))
    now = time.time()
    result["runtime_s"] = max(0.0, now - float(state["started_epoch_s"]))
    result["heartbeat_age_s"] = max(0.0, now - float(state["heartbeat_epoch_s"]))
    result["validation_pose_progress"] = _progress(state)
    if state["state"] in ACTIVE_STATES and (
        result["heartbeat_age_s"] > STALE_AFTER_S or not _pid_alive(state.get("worker_pid"))
    ):
        result["state"] = "paused_on_error"
        result["error_summary"] = "validation coordinator heartbeat is stale or worker exited"
    return result


def _status(args) -> dict:
    result = _status_payload(_resolve_run(args.run_dir, args.output_root))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _continue(args) -> dict:
    run_dir = _resolve_run(args.run_dir, args.output_root)
    state = _read_json(run_dir / "status.json")
    effective = _status_payload(run_dir)
    if state["state"] in ACTIVE_STATES and effective["state"] == "paused_on_error":
        state.update(state="paused_on_error", child_pid=None, error_summary=effective["error_summary"])
        _atomic_json(run_dir / "status.json", state)
    if state.get("state") != "paused_on_error":
        raise RuntimeError("continue requires paused_on_error")
    if _pid_alive(state.get("worker_pid")) or _pid_alive(state.get("child_pid")):
        raise RuntimeError("cannot continue while an old validation process is alive")
    _check_identity(_read_json(run_dir / "manifest.json"))
    previous_error = state.get("error_summary")
    state.update(state="queued", child_pid=None, heartbeat_epoch_s=time.time(), error_summary=None)
    _atomic_json(run_dir / "status.json", state)
    _append_event(
        run_dir,
        {"event": "manual_continue_requested", "seed": state.get("current_seed"), "previous_error": previous_error},
    )
    worker_pid = _spawn_worker(run_dir, state, continuation=True)
    result = {"state": "queued", "run_dir": str(run_dir), "worker_pid": worker_pid}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subparsers = command.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    start.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    start.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    start.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    start.add_argument("--test-driver", help=argparse.SUPPRESS)
    start.add_argument("--kit_args", help=argparse.SUPPRESS)
    for name in ("status", "continue"):
        item = subparsers.add_parser(name)
        item.add_argument("--run-dir", type=Path)
        item.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        item.add_argument("--kit_args", help=argparse.SUPPRESS)
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--continuation", action="store_true")
    worker.add_argument("--kit_args", help=argparse.SUPPRESS)
    return command


def main() -> int:
    args = parser().parse_args()
    if args.command == "start":
        _start(args)
        return 0
    if args.command == "status":
        _status(args)
        return 0
    if args.command == "continue":
        _continue(args)
        return 0
    return _worker(args.run_dir, args.continuation)


if __name__ == "__main__":
    raise SystemExit(main())
