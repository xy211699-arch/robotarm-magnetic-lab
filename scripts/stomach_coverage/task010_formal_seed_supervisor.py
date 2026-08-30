#!/usr/bin/env python3
"""Detached sequential training and validation for TASK-010 formal seeds."""

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
DEFAULT_CONFIG = REPOSITORY / "configs/task010/cnn_gru_development_v1.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY / "artifacts/task010_cnn_gru/formal_seeds"
FORMAL_SEEDS = (991001, 991002, 991003)
NUM_ENVS = 12
ROLLOUT_STEPS = 64
MAX_UPDATES = 1000
SAVE_INTERVAL = 50
FORMAL_STEPS = 1200
COVERAGE_POINTS = 1201
CONTROL_HZ = 10
HEARTBEAT_INTERVAL_S = 5.0
STALE_AFTER_S = 60.0
ACTIVE_STATES = {"queued", "training", "validating"}


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_ns": time.time_ns(), **payload}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pid_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        stat = Path(f"/proc/{int(pid)}/stat")
        if stat.is_file() and stat.read_text(encoding="utf-8").split()[2] == "Z":
            return False
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def _last_jsonl(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def _tail(path: Path, count: int = 8) -> str | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return " | ".join(lines[-count:])[-4000:] if lines else None


def _validate_frozen_config(path: Path) -> dict:
    resolved = Path(path).resolve(strict=True)
    config = _read_json(resolved)
    training = config.get("training", {})
    ppo = config.get("ppo", {})
    checkpoints = config.get("checkpoints", {})
    episode = config.get("episode", {})
    pose_ids = tuple(config.get("validation", {}).get("pose_ids", ()))
    expected = {
        "num_envs": (training.get("num_envs"), NUM_ENVS),
        "max_updates": (training.get("max_updates"), MAX_UPDATES),
        "rollout_steps": (ppo.get("rollout_steps"), ROLLOUT_STEPS),
        "save_interval": (checkpoints.get("rolling_interval"), SAVE_INTERVAL),
        "formal_steps": (episode.get("formal_steps"), FORMAL_STEPS),
        "coverage_points": (episode.get("coverage_points"), COVERAGE_POINTS),
        "validation_pose_count": (len(pose_ids), 20),
    }
    wrong = {key: value for key, value in expected.items() if value[0] != value[1]}
    if wrong:
        raise ValueError(f"TASK-010 frozen formal contract mismatch: {wrong}")
    if len(set(pose_ids)) != 20:
        raise ValueError("TASK-010 frozen validation pose IDs must be unique")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "pose_ids": list(pose_ids),
        "num_envs": NUM_ENVS,
        "rollout_steps": ROLLOUT_STEPS,
        "max_updates": MAX_UPDATES,
        "save_interval": SAVE_INTERVAL,
        "formal_steps": FORMAL_STEPS,
        "coverage_points": COVERAGE_POINTS,
    }


def _git_head() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY, text=True
    ).strip()


def _update_link(link: Path, target: Path) -> None:
    temporary = link.with_name(link.name + f".{os.getpid()}.tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _resolve_run(value: Path | None, output_root: Path) -> Path:
    candidate = Path(value) if value is not None else Path(output_root) / "latest"
    return candidate.resolve(strict=True)


def _seed_record(run_dir: Path, seed: int) -> dict:
    seed_dir = run_dir / "seeds" / f"seed_{seed}"
    return {
        "seed": seed,
        "state": "queued",
        "training_complete": False,
        "validation_complete": False,
        "training_attempts": 0,
        "validation_attempts": 0,
        "training_dir": str(seed_dir / "training"),
        "validation_dir": str(seed_dir / "validation"),
        "error_stage": None,
        "error_summary": None,
    }


def _initial_state(run_dir: Path, run_id: str) -> dict:
    now = time.time()
    return {
        "schema": "robotarm_magnetic_lab.task010_formal_seed_status",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state": "queued",
        "current_seed": FORMAL_SEEDS[0],
        "current_stage": None,
        "worker_pid": -1,
        "child_pid": None,
        "started_epoch_s": now,
        "heartbeat_epoch_s": now,
        "finished_epoch_s": None,
        "error_summary": None,
        "seeds": {str(seed): _seed_record(run_dir, seed) for seed in FORMAL_SEEDS},
    }


def _spawn_worker(run_dir: Path, state: dict, *, continuation: bool) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--run-dir",
        str(run_dir),
    ]
    if continuation:
        command.append("--continuation")
    console = (run_dir / "coordinator.log").open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY,
        stdin=subprocess.DEVNULL,
        stdout=console,
        stderr=console,
        start_new_session=True,
    )
    console.close()
    state["worker_pid"] = process.pid
    state["heartbeat_epoch_s"] = time.time()
    _atomic_json(run_dir / "status.json", state)
    return process.pid


def _start(args) -> dict:
    contract = _validate_frozen_config(args.config)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            existing = _status_payload(_resolve_run(latest, root))
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {"state": "unknown"}
        if existing.get("state") in ACTIVE_STATES:
            raise RuntimeError(f"TASK-010 formal coordinator already active: {latest}")
    if args.test_driver and os.environ.get("TASK010_FORMAL_TEST_MODE") != "1":
        raise PermissionError("test driver is disabled outside TASK010_FORMAL_TEST_MODE=1")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / run_id
    run_dir.mkdir(parents=False)
    manifest = {
        "schema": "robotarm_magnetic_lab.task010_formal_seed_manifest",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_epoch_s": time.time(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "config": contract,
        "seeds": list(FORMAL_SEEDS),
        "execution": {"gpu": "cuda:0", "sequential": True, "max_parallel_children": 1},
        "test_driver": str(Path(args.test_driver).resolve()) if args.test_driver else None,
        "test_plan": str(Path(args.test_plan).resolve()) if args.test_plan else None,
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    state = _initial_state(run_dir, run_id)
    _atomic_json(run_dir / "status.json", state)
    _append_jsonl(run_dir / "events.jsonl", {"event": "queued", "seeds": list(FORMAL_SEEDS)})
    pid = _spawn_worker(run_dir, state, continuation=False)
    _update_link(latest, run_dir)
    result = {"run_id": run_id, "run_dir": str(run_dir), "worker_pid": pid, "state": "queued"}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _training_command(manifest: dict, seed: int, seed_record: dict, attempt: int) -> list[str]:
    if manifest.get("test_driver"):
        return _test_command(manifest, seed, "training", seed_record, attempt)
    return [
        str(REPOSITORY / "run_isaaclab.sh"), "-p",
        "scripts/stomach_coverage/train_task010.py",
        "--config", manifest["config"]["path"],
        "--output-dir", seed_record["training_dir"],
        "--max-updates", str(MAX_UPDATES),
        "--save-interval", str(SAVE_INTERVAL),
        "--validation", "disabled",
        "--backend", "isaac",
        "--device", "cuda:0",
        "--seed", str(seed),
    ]


def _validation_command(manifest: dict, seed: int, seed_record: dict, attempt: int) -> list[str]:
    checkpoint = Path(seed_record["training_dir"]) / "checkpoints/update_1000.pt"
    if manifest.get("test_driver"):
        return _test_command(manifest, seed, "validating", seed_record, attempt)
    return [
        str(REPOSITORY / "run_isaaclab.sh"), "-p",
        "scripts/stomach_coverage/validate_task010_checkpoint.py",
        "--config", manifest["config"]["path"],
        "--checkpoint", str(checkpoint),
        "--output-dir", seed_record["validation_dir"],
        "--device", "cuda:0",
    ]


def _test_command(manifest: dict, seed: int, stage: str, seed_record: dict, attempt: int) -> list[str]:
    command = [
        sys.executable,
        manifest["test_driver"],
        "--stage", stage,
        "--seed", str(seed),
        "--training-dir", seed_record["training_dir"],
        "--validation-dir", seed_record["validation_dir"],
        "--config", manifest["config"]["path"],
        "--attempt", str(attempt),
        "--run-dir", manifest["run_dir"],
    ]
    if manifest.get("test_plan"):
        command += ["--plan", manifest["test_plan"]]
    return command


def _training_progress(record: dict) -> dict:
    training_dir = Path(record["training_dir"])
    metric = _last_jsonl(training_dir / "metrics.jsonl")
    checkpoints = sorted((training_dir / "checkpoints").glob("update_*.pt"))
    return {
        "training_update": int(metric.get("update", 0)) if metric else 0,
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
    }


def _validation_progress(record: dict) -> dict:
    validation_dir = Path(record["validation_dir"])
    return {"validation_pose_progress": _line_count(validation_dir / "pose_records.jsonl")}


def _progress(state: dict) -> dict:
    seed = state.get("current_seed")
    if seed is None:
        seed = FORMAL_SEEDS[-1]
    record = state["seeds"][str(seed)]
    return {**_training_progress(record), **_validation_progress(record)}


def _run_child(run_dir: Path, manifest: dict, state: dict, command: list[str], *, seed: int, stage: str, attempt: int) -> int:
    log = run_dir / "logs" / f"seed_{seed}_{stage}_attempt_{attempt:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _append_jsonl(run_dir / "events.jsonl", {"event": "stage_started", "seed": seed, "stage": stage, "attempt": attempt, "command": command})
    with log.open("ab", buffering=0) as console:
        child = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=console,
            start_new_session=False,
        )
        state["child_pid"] = child.pid
        next_heartbeat = 0.0
        while child.poll() is None:
            now = time.time()
            if now >= next_heartbeat:
                state["heartbeat_epoch_s"] = now
                state.update(_progress(state))
                _atomic_json(run_dir / "status.json", state)
                next_heartbeat = now + HEARTBEAT_INTERVAL_S
            time.sleep(0.1)
    code = int(child.returncode)
    state["child_pid"] = None
    state["heartbeat_epoch_s"] = time.time()
    state.update(_progress(state))
    _atomic_json(run_dir / "status.json", state)
    _append_jsonl(run_dir / "events.jsonl", {"event": "stage_finished", "seed": seed, "stage": stage, "attempt": attempt, "exit_code": code})
    if code != 0:
        state["error_summary"] = _tail(log) or f"{stage} exited with code {code}"
    return code


def _check_training(record: dict) -> Path:
    training_dir = Path(record["training_dir"])
    checkpoint = training_dir / "checkpoints/update_1000.pt"
    metric = _last_jsonl(training_dir / "metrics.jsonl")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"formal checkpoint missing: {checkpoint}")
    if metric is None or int(metric.get("update", -1)) != MAX_UPDATES:
        raise ValueError("formal training metrics did not reach update 1000")
    return checkpoint


def _load_validation_curves(manifest: dict, record: dict) -> tuple[list[str], list[list[float]]]:
    validation_dir = Path(record["validation_dir"])
    checkpoint = Path(record["training_dir"]) / "checkpoints/update_1000.pt"
    expected_checkpoint = _sha256(checkpoint)
    expected_config = manifest["config"]["sha256"]
    expected_ids = tuple(manifest["config"]["pose_ids"])
    records = [json.loads(line) for line in (validation_dir / "pose_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    trajectories = [json.loads(line) for line in (validation_dir / "coverage_trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    record_ids = [str(item.get("pose_id")) for item in records]
    trajectory_by_id = {str(item.get("pose_id")): item for item in trajectories}
    if len(records) != 20 or len(set(record_ids)) != 20 or set(record_ids) != set(expected_ids):
        raise ValueError("formal validation requires exactly 20 unique frozen pose records")
    if len(trajectories) != 20 or len(trajectory_by_id) != 20 or set(trajectory_by_id) != set(expected_ids):
        raise ValueError("formal validation requires exactly 20 unique raw coverage trajectories")
    curves = []
    for pose_id in expected_ids:
        record_row = records[record_ids.index(pose_id)]
        trajectory = trajectory_by_id[pose_id]
        if record_row.get("checkpoint_sha256") != expected_checkpoint or trajectory.get("checkpoint_sha256") != expected_checkpoint:
            raise ValueError(f"checkpoint mismatch for {pose_id}")
        if record_row.get("config_sha256") != expected_config or trajectory.get("config_sha256") != expected_config:
            raise ValueError(f"config mismatch for {pose_id}")
        values = [float(value) for value in trajectory.get("coverage_fraction", ())]
        if len(values) != COVERAGE_POINTS:
            raise ValueError(f"{pose_id} does not contain 1201 true 10 Hz coverage points")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{pose_id} contains non-finite or invalid reachable coverage")
        if any(after < before - 1.0e-12 for before, after in zip(values, values[1:])):
            raise ValueError(f"{pose_id} reachable cumulative coverage is non-monotonic")
        curves.append(values)
    return list(expected_ids), curves


def _write_seed_mean(validation_dir: Path, curves: list[list[float]]) -> list[float]:
    means = [sum(values) / len(values) for values in zip(*curves)]
    path = validation_dir / "mean_coverage.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "mean_coverage"))
        for index, value in enumerate(means):
            writer.writerow((index / CONTROL_HZ, value))
    return means


def _audit_validation(manifest: dict, record: dict) -> list[float]:
    _, curves = _load_validation_curves(manifest, record)
    return _write_seed_mean(Path(record["validation_dir"]), curves)


def _aggregate(run_dir: Path, state: dict) -> Path:
    seed_means = []
    for seed in FORMAL_SEEDS:
        path = Path(state["seeds"][str(seed)]["validation_dir"]) / "mean_coverage.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != COVERAGE_POINTS:
            raise ValueError(f"seed {seed} mean curve does not contain 1201 rows")
        seed_means.append([float(row["mean_coverage"]) for row in rows])
    output = run_dir / "summary/formal_three_seed_mean_coverage.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s",
            "seed_991001_mean_coverage",
            "seed_991002_mean_coverage",
            "seed_991003_mean_coverage",
            "formal_mean_coverage",
            "formal_std_coverage",
        ))
        for index, values in enumerate(zip(*seed_means)):
            mean = sum(values) / 3.0
            std = math.sqrt(sum((value - mean) ** 2 for value in values) / 3.0)
            writer.writerow((index / CONTROL_HZ, *values, mean, std))
    with output.open(newline="", encoding="utf-8") as stream:
        if sum(1 for _ in csv.DictReader(stream)) != COVERAGE_POINTS:
            raise RuntimeError("formal aggregate CSV row count verification failed")
    return output


def _pause(run_dir: Path, state: dict, seed: int, stage: str, error: BaseException | str) -> int:
    message = str(error)
    record = state["seeds"][str(seed)]
    record.update(state="paused_on_error", error_stage=stage, error_summary=message)
    state.update(
        state="paused_on_error",
        current_seed=seed,
        current_stage=stage,
        child_pid=None,
        heartbeat_epoch_s=time.time(),
        error_summary=message,
    )
    _atomic_json(run_dir / "status.json", state)
    _append_jsonl(run_dir / "events.jsonl", {"event": "paused_on_error", "seed": seed, "stage": stage, "error": message})
    return 1


def _worker(run_dir: Path, continuation: bool) -> int:
    run_dir = Path(run_dir).resolve(strict=True)
    lock_stream = (run_dir / ".coordinator.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 73
    manifest = _read_json(run_dir / "manifest.json")
    state = _read_json(run_dir / "status.json")
    state.update(worker_pid=os.getpid(), heartbeat_epoch_s=time.time(), error_summary=None)
    _atomic_json(run_dir / "status.json", state)
    _append_jsonl(run_dir / "events.jsonl", {"event": "worker_started", "continuation": continuation, "worker_pid": os.getpid()})
    try:
        if _sha256(Path(manifest["config"]["path"])) != manifest["config"]["sha256"]:
            raise RuntimeError("frozen TASK-010 config changed after start")
        _validate_frozen_config(Path(manifest["config"]["path"]))
        for seed in FORMAL_SEEDS:
            record = state["seeds"][str(seed)]
            if record["validation_complete"]:
                continue
            state["current_seed"] = seed
            if not record["training_complete"]:
                stage = "training"
                record["training_attempts"] += 1
                attempt = int(record["training_attempts"])
                record.update(state=stage, error_stage=None, error_summary=None)
                state.update(state=stage, current_stage=stage, error_summary=None)
                _atomic_json(run_dir / "status.json", state)
                command = _training_command(manifest, seed, record, attempt)
                if _run_child(run_dir, manifest, state, command, seed=seed, stage=stage, attempt=attempt) != 0:
                    return _pause(run_dir, state, seed, stage, state.get("error_summary") or "training process failed")
                try:
                    checkpoint = _check_training(record)
                except BaseException as error:
                    return _pause(run_dir, state, seed, stage, error)
                record.update(training_complete=True, latest_checkpoint=str(checkpoint), latest_checkpoint_sha256=_sha256(checkpoint))
                _atomic_json(run_dir / "status.json", state)

            stage = "validating"
            record["validation_attempts"] += 1
            attempt = int(record["validation_attempts"])
            record.update(state=stage, error_stage=None, error_summary=None)
            state.update(state=stage, current_stage=stage, error_summary=None)
            _atomic_json(run_dir / "status.json", state)
            command = _validation_command(manifest, seed, record, attempt)
            if _run_child(run_dir, manifest, state, command, seed=seed, stage=stage, attempt=attempt) != 0:
                return _pause(run_dir, state, seed, stage, state.get("error_summary") or "validation process failed")
            try:
                mean = _audit_validation(manifest, record)
            except BaseException as error:
                return _pause(run_dir, state, seed, stage, error)
            record.update(
                state="validated",
                validation_complete=True,
                error_stage=None,
                error_summary=None,
                validation_pose_progress=20,
                mean_curve_points=len(mean),
                mean_final_coverage=mean[-1],
            )
            state.update(state="validated", current_stage=None, heartbeat_epoch_s=time.time())
            _atomic_json(run_dir / "status.json", state)
            _append_jsonl(run_dir / "events.jsonl", {"event": "validated", "seed": seed, "mean_final_coverage": mean[-1]})

        output = _aggregate(run_dir, state)
        state.update(
            state="completed",
            current_seed=None,
            current_stage=None,
            child_pid=None,
            heartbeat_epoch_s=time.time(),
            finished_epoch_s=time.time(),
            error_summary=None,
            aggregate_csv=str(output),
            aggregate_csv_sha256=_sha256(output),
        )
        _atomic_json(run_dir / "status.json", state)
        _append_jsonl(run_dir / "events.jsonl", {"event": "completed", "aggregate_csv": str(output), "sha256": state["aggregate_csv_sha256"]})
        return 0
    except BaseException as error:
        with (run_dir / "coordinator.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        seed = int(state.get("current_seed") or FORMAL_SEEDS[0])
        stage = str(state.get("current_stage") or "training")
        return _pause(run_dir, state, seed, stage, error)
    finally:
        lock_stream.close()


def _status_payload(run_dir: Path) -> dict:
    state = _read_json(run_dir / "status.json")
    now = time.time()
    result = json.loads(json.dumps(state))
    result.update(_progress(state))
    result["runtime_s"] = max(0.0, now - float(state.get("started_epoch_s", now)))
    result["heartbeat_age_s"] = max(0.0, now - float(state.get("heartbeat_epoch_s", 0.0)))
    if state.get("state") in ACTIVE_STATES and (
        result["heartbeat_age_s"] > STALE_AFTER_S or not _pid_alive(state.get("worker_pid"))
    ):
        result["state"] = "paused_on_error"
        result["error_summary"] = (
            f"coordinator state is stale: heartbeat_age_s={result['heartbeat_age_s']:.3f}, "
            f"worker_pid={state.get('worker_pid')}"
        )
    return result


def _status(args) -> dict:
    run_dir = _resolve_run(args.run_dir, args.output_root)
    result = _status_payload(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _continue(args) -> dict:
    run_dir = _resolve_run(args.run_dir, args.output_root)
    state = _read_json(run_dir / "status.json")
    effective = _status_payload(run_dir)
    if state.get("state") in ACTIVE_STATES and effective.get("state") == "paused_on_error":
        state.update(
            state="paused_on_error",
            child_pid=None,
            heartbeat_epoch_s=time.time(),
            error_summary=effective["error_summary"],
        )
        seed = int(state.get("current_seed") or FORMAL_SEEDS[0])
        record = state["seeds"][str(seed)]
        record.update(
            state="paused_on_error",
            error_stage=state.get("current_stage") or "training",
            error_summary=effective["error_summary"],
        )
        _atomic_json(run_dir / "status.json", state)
        _append_jsonl(
            run_dir / "events.jsonl",
            {"event": "stale_state_acknowledged", "error": effective["error_summary"]},
        )
    if state.get("state") != "paused_on_error":
        raise RuntimeError("continue requires persisted state=paused_on_error")
    if _pid_alive(state.get("worker_pid")):
        raise RuntimeError("cannot continue while previous coordinator PID is alive")
    _append_jsonl(run_dir / "events.jsonl", {"event": "manual_continue_requested", "seed": state.get("current_seed"), "stage": state.get("current_stage")})
    pid = _spawn_worker(run_dir, state, continuation=True)
    result = {"run_id": state["run_id"], "run_dir": str(run_dir), "worker_pid": pid, "state": "paused_on_error", "retry_stage": state.get("current_stage")}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="后台启动三个正式种子的顺序训练、验证与汇总")
    start.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    start.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    start.add_argument("--test-driver", help=argparse.SUPPRESS)
    start.add_argument("--test-plan", help=argparse.SUPPRESS)
    start.add_argument("--kit_args", help=argparse.SUPPRESS)
    status = subparsers.add_parser("status", help="只读显示当前种子、阶段、进度、心跳与错误")
    status.add_argument("--run-dir", type=Path)
    status.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    status.add_argument("--kit_args", help=argparse.SUPPRESS)
    continuation = subparsers.add_parser("continue", help="人工检查错误后重试暂停种子的失败阶段")
    continuation.add_argument("--run-dir", type=Path)
    continuation.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    continuation.add_argument("--kit_args", help=argparse.SUPPRESS)
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--run-dir", type=Path, required=True)
    worker.add_argument("--continuation", action="store_true")
    worker.add_argument("--kit_args", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
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
