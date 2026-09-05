"""Pure artifact health and completion checks for TASK-010 training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping


TARGET_UPDATE = 1000
TRANSITIONS_PER_UPDATE = 768
DEGRADED_TPS = 10.0
SUSPECTED_STALL_S = 300.0
CRITICAL_STALL_S = 900.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoint_update(path: Path) -> int:
    path = Path(path)
    stem = path.stem
    if path.suffix != ".pt" or not stem.startswith("update_"):
        raise ValueError(f"invalid checkpoint filename: {path.name}")
    suffix = stem.removeprefix("update_")
    if len(suffix) != 4 or not suffix.isdigit():
        raise ValueError(f"invalid checkpoint filename: {path.name}")
    return int(suffix)


def read_complete_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return []
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        if b"\n" not in data:
            return []
        data = data.rsplit(b"\n", 1)[0] + b"\n"
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid complete JSONL row {line_number} in {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} in {path} is not an object")
        rows.append(row)
    return rows


def _latest_checkpoint(training_dir: Path) -> tuple[Path | None, int | None]:
    candidates: list[tuple[int, Path]] = []
    for path in (Path(training_dir) / "checkpoints").glob("update_*.pt"):
        try:
            candidates.append((parse_checkpoint_update(path), path))
        except ValueError:
            continue
    if not candidates:
        return None, None
    update, path = max(candidates, key=lambda item: item[0])
    return path, update


def remaining_updates(checkpoint_update: int, target_update: int = TARGET_UPDATE) -> int:
    checkpoint_update = int(checkpoint_update)
    target_update = int(target_update)
    if checkpoint_update < 0 or checkpoint_update > target_update:
        raise ValueError("checkpoint update is outside the formal training range")
    return target_update - checkpoint_update


def progress_snapshot(training_dir: Path, now_epoch_s: float | None = None) -> dict[str, Any]:
    training_dir = Path(training_dir)
    rows = read_complete_jsonl(training_dir / "metrics.jsonl")
    checkpoint, checkpoint_update = _latest_checkpoint(training_dir)
    now = float(time.time() if now_epoch_s is None else now_epoch_s)
    if not rows:
        return {
            "health": "starting",
            "observed_update": 0,
            "remaining_updates": TARGET_UPDATE,
            "last_metric_epoch_s": None,
            "metric_age_s": None,
            "current_tps": None,
            "median_tps_last_10": None,
            "eta_s": None,
            "latest_checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_update": checkpoint_update,
        }

    last = rows[-1]
    observed_update = int(last.get("update", 0))
    last_metric_epoch_s = float(last.get("time_ns", 0)) / 1.0e9
    metric_age_s = max(0.0, now - last_metric_epoch_s)
    recent_tps = [
        float(row["transitions_per_second"])
        for row in rows[-10:]
        if isinstance(row.get("transitions_per_second"), (int, float))
        and math.isfinite(float(row["transitions_per_second"]))
        and float(row["transitions_per_second"]) > 0.0
    ]
    median_tps = float(statistics.median(recent_tps)) if recent_tps else None
    if metric_age_s > CRITICAL_STALL_S:
        health = "critical_stall"
    elif metric_age_s > SUSPECTED_STALL_S:
        health = "suspected_stall"
    elif median_tps is not None and median_tps < DEGRADED_TPS:
        health = "degraded_performance"
    else:
        health = "healthy"
    remaining = max(0, TARGET_UPDATE - observed_update)
    eta_s = (
        float(remaining * TRANSITIONS_PER_UPDATE / median_tps)
        if median_tps is not None and median_tps > 0.0
        else None
    )
    return {
        "health": health,
        "observed_update": observed_update,
        "remaining_updates": remaining,
        "last_metric_epoch_s": last_metric_epoch_s,
        "metric_age_s": metric_age_s,
        "current_tps": float(last.get("transitions_per_second", 0.0)),
        "median_tps_last_10": median_tps,
        "eta_s": eta_s,
        "latest_checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_update": checkpoint_update,
    }


def _finite_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_value(item) for item in value)
    return True


def _audit_base(training_dir: Path, expected: Mapping[str, Any], child_exit_code: int) -> dict[str, Any]:
    snapshot = progress_snapshot(training_dir)
    return {
        "complete": False,
        "error_type": None,
        "error_summary": None,
        "child_exit_code": int(child_exit_code),
        "expected_update": int(expected.get("target_update", TARGET_UPDATE)),
        "observed_update": int(snapshot["observed_update"]),
        "latest_checkpoint": snapshot["latest_checkpoint"],
        "checkpoint_update": snapshot["checkpoint_update"],
        "checkpoint_sha256": None,
    }


def _fail(audit: dict[str, Any], error_type: str, summary: str) -> dict[str, Any]:
    audit["error_type"] = error_type
    audit["error_summary"] = summary
    return audit


def audit_completion(
    training_dir: Path,
    expected: Mapping[str, Any],
    *,
    child_exit_code: int,
) -> dict[str, Any]:
    training_dir = Path(training_dir)
    audit = _audit_base(training_dir, expected, child_exit_code)
    if child_exit_code != 0:
        return _fail(audit, "child_nonzero_exit", f"training child exited with code {child_exit_code}")

    metrics = read_complete_jsonl(training_dir / "metrics.jsonl")
    if any(row.get("all_finite") is not True or not _finite_value(row) for row in metrics):
        return _fail(audit, "non_finite_metric", "one or more complete metric rows are non-finite")

    target = int(expected.get("target_update", TARGET_UPDATE))
    if audit["observed_update"] != target:
        return _fail(
            audit,
            "incomplete_zero_exit",
            f"training exited with code 0 at update {audit['observed_update']}; expected {target}",
        )

    checkpoint = training_dir / "checkpoints" / f"update_{target:04d}.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return _fail(audit, "checkpoint_missing", f"final checkpoint is missing or empty: {checkpoint}")
    audit["latest_checkpoint"] = str(checkpoint)
    audit["checkpoint_update"] = target
    audit["checkpoint_sha256"] = _sha256(checkpoint)

    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except BaseException as error:
        return _fail(audit, "checkpoint_corrupt", f"cannot load final checkpoint: {type(error).__name__}: {error}")
    if not isinstance(payload, Mapping) or int(payload.get("current_update", -1)) != target:
        return _fail(audit, "checkpoint_corrupt", "final checkpoint current_update mismatch")

    events = read_complete_jsonl(training_dir / "events.jsonl")
    initialized = next((row for row in events if row.get("event") == "runner_initialized"), None)
    if initialized is None:
        return _fail(audit, "identity_mismatch", "runner_initialized event is missing")
    metadata = payload.get("experiment_metadata") or {}
    checks = {
        "seed": (initialized.get("seed"), expected.get("seed")),
        "git_commit": (payload.get("git_commit"), expected.get("git_commit")),
        "base_config_sha256": (
            payload.get("base_config_sha256", payload.get("config_hash")),
            expected.get("base_config_sha256"),
        ),
        "visual_dependence_config_sha256": (
            payload.get("visual_dependence_config_sha256"),
            expected.get("visual_dependence_config_sha256"),
        ),
        "visual_condition": (metadata.get("visual_condition"), expected.get("visual_condition")),
        "dependency_audit_hash": (
            payload.get("dependency_audit_hash"),
            expected.get("dependency_audit_hash"),
        ),
    }
    mismatches = [name for name, (actual, wanted) in checks.items() if actual != wanted]
    if mismatches:
        return _fail(audit, "identity_mismatch", "checkpoint identity mismatch: " + ", ".join(mismatches))

    audit["complete"] = True
    return audit
