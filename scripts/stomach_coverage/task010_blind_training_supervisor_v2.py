#!/usr/bin/env python3
"""Detached, training-only supervisor for three formal TASK-010 Blind-GRU seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "source" / "robotarm_magnetic_lab"))

from robotarm_magnetic_lab.runtime.task010_training_health import (  # noqa: E402
    audit_completion,
    progress_snapshot,
    remaining_updates,
)


SEEDS = (991001, 991002, 991003)
TARGET_UPDATE = 1000
SAVE_INTERVAL = 50
HEARTBEAT_INTERVAL_S = 5.0
STALE_AFTER_S = 60.0
DEFAULT_BASE_CONFIG = REPOSITORY / "configs/task010/cnn_gru_development_v1.json"
DEFAULT_VISUAL_CONFIG = REPOSITORY / "configs/task010/visual_dependence_v1.json"
DEFAULT_DEPENDENCY_AUDIT = Path(
    "/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate0/prerequisites.json"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_blind_training_v2"
)
ACTIVE_STATES = {"queued", "training"}


def stage_names() -> tuple[str, ...]:
    return tuple(f"train_blind_seed_{seed}" for seed in SEEDS)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_identity(path: Path) -> dict:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_event(run_dir: Path, payload: dict) -> None:
    with (Path(run_dir) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_ns": time.time_ns(), **payload}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pid_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        stat = Path(f"/proc/{int(pid)}/stat")
        if stat.is_file() and stat.read_text(encoding="utf-8").split()[2] == "Z":
            return False
        return True
    except (ProcessLookupError, PermissionError, FileNotFoundError):
        return False


def git_identity(*, require_clean: bool = False) -> dict:
    status = subprocess.check_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=no"),
        cwd=REPOSITORY,
        text=True,
    )
    if require_clean and status.strip():
        raise RuntimeError("formal training requires a clean tracked worktree")
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPOSITORY, text=True).strip()
    branch = subprocess.check_output(
        ("git", "branch", "--show-current"), cwd=REPOSITORY, text=True
    ).strip()
    return {
        "commit": commit,
        "branch": branch,
        "tracked_status": status.splitlines(),
        "tracked_status_sha256": _text_sha256(status),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _gpu_identity() -> dict:
    try:
        output = subprocess.check_output(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            text=True,
            timeout=5.0,
            stderr=subprocess.STDOUT,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {"query_error": f"{type(error).__name__}: {error}"}
    return {"rows": [line.strip() for line in output.splitlines() if line.strip()]}


def _load_contract(base_config_path: Path, visual_config_path: Path) -> dict:
    base_path = Path(base_config_path).resolve(strict=True)
    visual_path = Path(visual_config_path).resolve(strict=True)
    base = _read_json(base_path)
    visual = _read_json(visual_path)
    violations: list[str] = []
    if base.get("training", {}).get("num_envs") != 12:
        violations.append("training.num_envs must be 12")
    if base.get("ppo", {}).get("rollout_steps") != 64:
        violations.append("ppo.rollout_steps must be 64")
    if base.get("training", {}).get("max_updates") != TARGET_UPDATE:
        violations.append("training.max_updates must be 1000")
    if base.get("checkpoints", {}).get("rolling_interval") != SAVE_INTERVAL:
        violations.append("checkpoints.rolling_interval must be 50")
    if visual.get("formal_seeds") != list(SEEDS):
        violations.append("visual formal_seeds mismatch")
    if visual.get("training_conditions") != ["blind"]:
        violations.append("visual training_conditions must contain only blind")
    recorded_base = visual.get("base_config", {})
    if recorded_base.get("sha256") != _sha256(base_path):
        violations.append("visual config base file hash mismatch")
    if violations:
        raise ValueError("; ".join(violations))
    return {
        "base": base,
        "visual": visual,
        "base_path": base_path,
        "visual_path": visual_path,
        "base_config_sha256": str(base["config_sha256"]),
        "visual_config_sha256": str(visual["config_sha256"]),
    }


def _training_dir(run_dir: Path, seed: int) -> Path:
    return Path(run_dir) / "training" / "blind" / f"seed_{seed}"


def training_command(
    manifest: dict,
    run_dir: Path,
    *,
    seed: int,
    resume_checkpoint: Path | None,
) -> list[str]:
    if resume_checkpoint is None:
        updates = TARGET_UPDATE
    else:
        from robotarm_magnetic_lab.runtime.task010_training_health import parse_checkpoint_update

        updates = remaining_updates(parse_checkpoint_update(resume_checkpoint), TARGET_UPDATE)
    output_dir = _training_dir(run_dir, seed)
    test_driver = manifest.get("test_driver")
    if test_driver:
        command = [
            sys.executable,
            str(test_driver),
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
            "--max-updates",
            str(updates),
            "--save-interval",
            str(SAVE_INTERVAL),
            "--git-commit",
            manifest["git"]["commit"],
            "--base-config-sha256",
            manifest["base_config_sha256"],
            "--visual-config-sha256",
            manifest["visual_config_sha256"],
            "--dependency-audit-sha256",
            manifest["dependency_audit"]["sha256"],
        ]
    else:
        command = [
            sys.executable,
            str(REPOSITORY / "scripts/stomach_coverage/train_task010.py"),
            "--config",
            str(manifest["base_config"]),
            "--visual-condition",
            "blind",
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
            "--max-updates",
            str(updates),
            "--save-interval",
            str(SAVE_INTERVAL),
            "--validation",
            "disabled",
            "--dependency-audit",
            str(manifest["dependency_audit"]["path"]),
            "--backend",
            "isaac",
            "--device",
            "cuda:0",
        ]
    if resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(Path(resume_checkpoint).resolve(strict=True))))
    return command


def _initial_state(run_dir: Path, run_id: str) -> dict:
    now = time.time()
    return {
        "schema": "robotarm_magnetic_lab.task010_blind_training_v2_status",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state": "queued",
        "current_stage": None,
        "worker_pid": -1,
        "child_pid": None,
        "started_epoch_s": now,
        "heartbeat_epoch_s": now,
        "finished_epoch_s": None,
        "progress": None,
        "error": None,
        "stages": {
            name: {
                "state": "queued",
                "attempts": 0,
                "started_epoch_s": None,
                "finished_epoch_s": None,
                "completion_audit": None,
            }
            for name in stage_names()
        },
    }


def _update_link(link: Path, target: Path) -> None:
    temporary = link.with_name(link.name + f".{os.getpid()}.tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _resolve_run(run_dir: Path | None, root: Path) -> Path:
    return (Path(run_dir) if run_dir is not None else Path(root) / "latest").resolve(strict=True)


def _spawn_worker(run_dir: Path, state: dict, *, continuation: bool = False) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--run-dir", str(run_dir)]
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


def _effective_status(run_dir: Path) -> dict:
    state = _read_json(Path(run_dir) / "status.json")
    result = json.loads(json.dumps(state))
    now = time.time()
    result["runtime_s"] = max(0.0, now - float(state.get("started_epoch_s", now)))
    result["heartbeat_age_s"] = max(0.0, now - float(state.get("heartbeat_epoch_s", 0.0)))
    if state.get("state") in ACTIVE_STATES and (
        result["heartbeat_age_s"] > STALE_AFTER_S or not pid_alive(state.get("worker_pid"))
    ):
        result["state"] = "paused_on_error"
        result["error"] = {
            "error_type": "coordinator_stale",
            "error_summary": "coordinator heartbeat is stale or worker PID is not alive",
        }
    return result


def start(args) -> dict:
    test_driver = Path(args.test_driver).resolve(strict=True) if args.test_driver else None
    if test_driver and os.environ.get("TASK010_V2_TEST_MODE") != "1":
        raise PermissionError("test driver requires TASK010_V2_TEST_MODE=1")
    contract = _load_contract(args.base_config, args.visual_config)
    git = git_identity(require_clean=not bool(test_driver))
    dependency_audit = _file_identity(args.dependency_audit)
    root = Path(args.artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 1024 * 1024 * 1024:
        raise OSError("formal artifact root has less than 1 GiB free")
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            current = _effective_status(latest.resolve(strict=True))
        except (FileNotFoundError, json.JSONDecodeError):
            current = {"state": "unknown"}
        if current.get("state") in ACTIVE_STATES:
            raise RuntimeError("a V2 training supervisor is already active")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = root / run_id
    run_dir.mkdir(parents=False)
    manifest = {
        "schema": "robotarm_magnetic_lab.task010_blind_training_v2_manifest",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "base_config": str(contract["base_path"]),
        "base_config_file": _file_identity(contract["base_path"]),
        "base_config_sha256": contract["base_config_sha256"],
        "visual_config": str(contract["visual_path"]),
        "visual_config_file": _file_identity(contract["visual_path"]),
        "visual_config_sha256": contract["visual_config_sha256"],
        "dependency_audit": dependency_audit,
        "training_contract": {
            "seeds": list(SEEDS),
            "visual_condition": "blind",
            "num_envs": 12,
            "rollout_steps": 64,
            "target_update": TARGET_UPDATE,
            "checkpoint_interval": SAVE_INTERVAL,
            "device": "cuda:0",
            "validation": "disabled",
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "packages": {
                "torch": _package_version("torch"),
                "isaaclab": _package_version("isaaclab"),
                "rsl_rl": _package_version("rsl-rl-lib"),
            },
            "gpu": _gpu_identity(),
        },
        "test_driver": str(test_driver) if test_driver else None,
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    state = _initial_state(run_dir, run_id)
    _atomic_json(run_dir / "status.json", state)
    _append_event(run_dir, {"event": "queued", "stages": list(stage_names())})
    worker_pid = _spawn_worker(run_dir, state)
    _update_link(latest, run_dir)
    _atomic_text(root / "latest_run_path.txt", str(run_dir) + "\n")
    result = {"run_id": run_id, "run_dir": str(run_dir), "worker_pid": worker_pid, "state": "queued"}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _identity_matches(manifest: dict) -> bool:
    if manifest.get("test_driver"):
        return True
    current = git_identity(require_clean=True)
    return (
        current["commit"] == manifest["git"]["commit"]
        and current["branch"] == manifest["git"]["branch"]
        and _sha256(Path(manifest["base_config"])) == manifest["base_config_file"]["sha256"]
        and _sha256(Path(manifest["visual_config"])) == manifest["visual_config_file"]["sha256"]
        and _sha256(Path(manifest["dependency_audit"]["path"]))
        == manifest["dependency_audit"]["sha256"]
    )


def _console_tail(path: Path, lines: int = 5) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def _run_child(run_dir: Path, manifest: dict, state: dict, stage: str, seed: int, attempt: int) -> dict:
    training_dir = _training_dir(run_dir, seed)
    resume = None
    if attempt > 1:
        snapshot = progress_snapshot(training_dir)
        if snapshot.get("latest_checkpoint"):
            resume = Path(snapshot["latest_checkpoint"])
    command = training_command(manifest, run_dir, seed=seed, resume_checkpoint=resume)
    log = run_dir / "logs" / f"{stage}_attempt_{attempt:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _append_event(run_dir, {"event": "stage_started", "stage": stage, "seed": seed, "attempt": attempt, "command": command})
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
                state["progress"] = progress_snapshot(training_dir, now_epoch_s=now)
                _atomic_json(run_dir / "status.json", state)
                next_heartbeat = now + HEARTBEAT_INTERVAL_S
            time.sleep(0.02 if manifest.get("test_driver") else 0.1)
    state["child_pid"] = None
    state["heartbeat_epoch_s"] = time.time()
    state["progress"] = progress_snapshot(training_dir)
    _atomic_json(run_dir / "status.json", state)
    _append_event(run_dir, {"event": "stage_finished", "stage": stage, "seed": seed, "attempt": attempt, "exit_code": int(child.returncode)})
    expected = {
        "seed": seed,
        "target_update": TARGET_UPDATE,
        "git_commit": manifest["git"]["commit"],
        "base_config_sha256": manifest["base_config_sha256"],
        "visual_dependence_config_sha256": manifest["visual_config_sha256"],
        "visual_condition": "blind",
        "dependency_audit_hash": manifest["dependency_audit"]["sha256"],
    }
    audit = audit_completion(training_dir, expected, child_exit_code=int(child.returncode))
    audit["console_tail"] = _console_tail(log)
    return audit


def _pause(run_dir: Path, state: dict, stage: str, audit: dict) -> int:
    state["stages"][stage].update(state="paused_on_error", completion_audit=audit)
    state.update(
        state="paused_on_error",
        current_stage=stage,
        child_pid=None,
        heartbeat_epoch_s=time.time(),
        error=audit,
    )
    _atomic_json(run_dir / "status.json", state)
    _append_event(run_dir, {"event": "paused_on_error", "stage": stage, "error": audit})
    return 1


def worker(run_dir: Path, continuation: bool = False) -> int:
    run_dir = Path(run_dir).resolve(strict=True)
    lock = (run_dir / ".coordinator.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return 73
    manifest = _read_json(run_dir / "manifest.json")
    state = _read_json(run_dir / "status.json")
    state.update(worker_pid=os.getpid(), heartbeat_epoch_s=time.time(), error=None)
    _atomic_json(run_dir / "status.json", state)
    _append_event(run_dir, {"event": "worker_started", "continuation": continuation, "worker_pid": os.getpid()})
    try:
        for seed, stage in zip(SEEDS, stage_names(), strict=True):
            record = state["stages"][stage]
            if record["state"] == "completed":
                continue
            if not _identity_matches(manifest):
                audit = {
                    "complete": False,
                    "error_type": "identity_mismatch",
                    "error_summary": "Git or frozen input identity changed during the run",
                }
                return _pause(run_dir, state, stage, audit)
            record["attempts"] += 1
            attempt = int(record["attempts"])
            record.update(state="running", started_epoch_s=time.time(), finished_epoch_s=None)
            state.update(state="training", current_stage=stage, error=None)
            _atomic_json(run_dir / "status.json", state)
            audit = _run_child(run_dir, manifest, state, stage, seed, attempt)
            if not audit["complete"]:
                return _pause(run_dir, state, stage, audit)
            record.update(
                state="completed",
                finished_epoch_s=time.time(),
                completion_audit=audit,
            )
            state.update(current_stage=None, heartbeat_epoch_s=time.time(), progress=None)
            _atomic_json(run_dir / "status.json", state)
        state.update(
            state="completed_training",
            current_stage=None,
            child_pid=None,
            heartbeat_epoch_s=time.time(),
            finished_epoch_s=time.time(),
            progress=None,
            error=None,
        )
        _atomic_json(run_dir / "status.json", state)
        _append_event(run_dir, {"event": "completed_training"})
        return 0
    except BaseException as error:
        with (run_dir / "coordinator.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        stage = state.get("current_stage") or next(
            name for name in stage_names() if state["stages"][name]["state"] != "completed"
        )
        audit = {
            "complete": False,
            "error_type": "supervisor_exception",
            "error_summary": f"{type(error).__name__}: {error}",
        }
        return _pause(run_dir, state, stage, audit)
    finally:
        lock.close()


def status(args) -> dict:
    run_dir = _resolve_run(args.run_dir, args.artifact_root)
    result = _effective_status(run_dir)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def watch(args) -> None:
    run_dir = _resolve_run(args.run_dir, args.artifact_root)
    try:
        while True:
            result = _effective_status(run_dir)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            if result.get("state") in {"completed_training", "paused_on_error"}:
                return
            time.sleep(max(1, int(args.interval)))
    except KeyboardInterrupt:
        print("watch stopped; training supervisor was not signalled", flush=True)


def diagnose(args) -> dict:
    run_dir = _resolve_run(args.run_dir, args.artifact_root)
    state = _effective_status(run_dir)
    result = {
        "run_dir": str(run_dir),
        "manifest": _read_json(run_dir / "manifest.json"),
        "status": state,
        "coordinator_log_tail": _console_tail(run_dir / "coordinator.log", 20),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _validate_resume_checkpoint(checkpoint: Path, manifest: dict, seed: int) -> dict:
    checkpoint = Path(checkpoint).resolve(strict=True)
    from robotarm_magnetic_lab.runtime.task010_training_health import parse_checkpoint_update

    update = parse_checkpoint_update(checkpoint)
    if update >= TARGET_UPDATE:
        raise RuntimeError(f"resume checkpoint update {update} is already at the target")
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except BaseException as error:
        raise RuntimeError(
            f"resume checkpoint cannot be loaded: {type(error).__name__}: {error}"
        ) from error
    metadata = payload.get("experiment_metadata") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(metadata, dict):
        raise RuntimeError("resume checkpoint has an invalid payload")
    expected = {
        "current_update": update,
        "git_commit": manifest["git"]["commit"],
        "base_config_sha256": manifest["base_config_sha256"],
        "visual_dependence_config_sha256": manifest["visual_config_sha256"],
        "visual_condition": "blind",
        "dependency_audit_hash": manifest["dependency_audit"]["sha256"],
    }
    actual = {
        "current_update": payload.get("current_update"),
        "git_commit": payload.get("git_commit"),
        "base_config_sha256": payload.get("base_config_sha256", payload.get("config_hash")),
        "visual_dependence_config_sha256": payload.get("visual_dependence_config_sha256"),
        "visual_condition": metadata.get("visual_condition"),
        "dependency_audit_hash": payload.get("dependency_audit_hash"),
    }
    mismatches = [name for name, value in actual.items() if value != expected[name]]
    if mismatches:
        raise RuntimeError("resume checkpoint identity mismatch: " + ", ".join(mismatches))
    return {
        "path": str(checkpoint),
        "update": update,
        "remaining_updates": remaining_updates(update, TARGET_UPDATE),
        "sha256": _sha256(checkpoint),
        "seed": int(seed),
    }


def continue_run(args) -> dict:
    run_dir = _resolve_run(args.run_dir, args.artifact_root)
    manifest = _read_json(run_dir / "manifest.json")
    state = _read_json(run_dir / "status.json")
    effective = _effective_status(run_dir)
    if state.get("state") in ACTIVE_STATES and effective.get("state") == "paused_on_error":
        stage = state.get("current_stage")
        audit = dict(effective["error"])
        audit.update(state.get("progress") or {})
        if stage in state.get("stages", {}):
            state["stages"][stage].update(state="paused_on_error", completion_audit=audit)
        state.update(
            state="paused_on_error",
            child_pid=None,
            heartbeat_epoch_s=time.time(),
            error=audit,
        )
        _atomic_json(run_dir / "status.json", state)
        _append_event(run_dir, {"event": "coordinator_stale_materialized", "error": audit})
    if state.get("state") != "paused_on_error":
        raise RuntimeError("continue requires persisted state paused_on_error")
    if pid_alive(state.get("worker_pid")) or pid_alive(state.get("child_pid")):
        raise RuntimeError("cannot continue while the previous worker or child PID is alive")
    stage = state.get("current_stage")
    if stage not in stage_names():
        raise RuntimeError("paused run does not identify a valid current training stage")
    seed = SEEDS[stage_names().index(stage)]
    if not _identity_matches(manifest):
        raise RuntimeError("cannot continue because Git or frozen input identity changed")
    snapshot = progress_snapshot(_training_dir(run_dir, seed))
    latest = snapshot.get("latest_checkpoint")
    if not latest:
        raise RuntimeError("cannot continue because no complete checkpoint is available")
    checkpoint_audit = _validate_resume_checkpoint(Path(latest), manifest, seed)
    previous_error = state.get("error")
    state["stages"][stage]["state"] = "queued"
    state.update(
        state="queued",
        child_pid=None,
        heartbeat_epoch_s=time.time(),
        progress=snapshot,
        error=None,
    )
    _atomic_json(run_dir / "status.json", state)
    _append_event(
        run_dir,
        {
            "event": "continue_authorized",
            "stage": stage,
            "seed": seed,
            "previous_error": previous_error,
            "resume_checkpoint": checkpoint_audit,
        },
    )
    worker_pid = _spawn_worker(run_dir, state, continuation=True)
    result = {
        "run_id": state["run_id"],
        "run_dir": str(run_dir),
        "worker_pid": worker_pid,
        "state": "queued",
        "stage": stage,
        "resume_checkpoint": checkpoint_audit,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subparsers = command.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    start_parser.add_argument("--visual-config", type=Path, default=DEFAULT_VISUAL_CONFIG)
    start_parser.add_argument("--dependency-audit", type=Path, default=DEFAULT_DEPENDENCY_AUDIT)
    start_parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    start_parser.add_argument("--test-driver", help=argparse.SUPPRESS)
    start_parser.add_argument("--kit_args", help=argparse.SUPPRESS)

    for name in ("status", "watch", "diagnose"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--run-dir", type=Path)
        sub.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
        sub.add_argument("--kit_args", help=argparse.SUPPRESS)
        if name == "watch":
            sub.add_argument("--interval", type=int, default=60)

    continuation = subparsers.add_parser("continue")
    continuation.add_argument("--run-dir", type=Path, required=True)
    continuation.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    continuation.add_argument("--kit_args", help=argparse.SUPPRESS)

    internal = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    internal.add_argument("--run-dir", type=Path, required=True)
    internal.add_argument("--continuation", action="store_true")
    internal.add_argument("--kit_args", help=argparse.SUPPRESS)
    return command


def main() -> int:
    args = parser().parse_args()
    if args.command == "start":
        start(args)
        return 0
    if args.command == "status":
        status(args)
        return 0
    if args.command == "watch":
        watch(args)
        return 0
    if args.command == "diagnose":
        diagnose(args)
        return 0
    if args.command == "continue":
        continue_run(args)
        return 0
    return worker(args.run_dir, continuation=args.continuation)


if __name__ == "__main__":
    raise SystemExit(main())
