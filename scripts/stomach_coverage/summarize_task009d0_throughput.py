#!/usr/bin/env python3
"""Strictly aggregate TASK-009D0 Gate 5 manifests and freeze num_envs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

from robotarm_magnetic_lab.coverage.entry_pose_library import manifest_hash  # noqa: E402
from robotarm_magnetic_lab.runtime.task009d0_config import (  # noqa: E402
    TASK009D0_CONFIG_PATH,
    TASK009D0_FROZEN_SCHEMA,
    load_task009d0_config,
)


SCHEMA = "robotarm_magnetic_lab.task009d0_throughput_manifest"
SUMMARY_SCHEMA = "robotarm_magnetic_lab.task009d0_throughput_summary"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_manifest(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if record.get("schema") != SCHEMA or int(record.get("version", 0)) != 1:
        raise ValueError("throughput manifest schema/version mismatch")
    if record.get("status") != "pass" or record.get("faults"):
        raise ValueError("throughput manifest is faulted")
    if record.get("branch") != "feature/TASK-009D0-vectorized-training-infrastructure":
        raise ValueError("throughput manifest branch mismatch")
    if not isinstance(record.get("commit"), str) or len(record["commit"]) != 40:
        raise ValueError("throughput manifest commit is not a full Git hash")
    if record.get("config_sha256") != config["config_sha256"]:
        raise ValueError("throughput manifest config mismatch")
    num_envs = int(record.get("num_envs", 0))
    repeat = int(record.get("repeat_index", -1))
    if num_envs not in config["num_env_candidates"] or repeat not in range(3):
        raise ValueError("throughput candidate/repeat mismatch")
    if record.get("clocks") != config["clocks"]:
        raise ValueError("throughput clock mismatch")
    benchmark = config["benchmark"]
    if int(record.get("warmup_steps", -1)) != benchmark["warmup_steps"]:
        raise ValueError("throughput warmup count mismatch")
    if int(record.get("measured_steps", -1)) != benchmark["measured_steps"]:
        raise ValueError("throughput measured count mismatch")
    if len(record.get("measurements", ())) != benchmark["measured_steps"]:
        raise ValueError("throughput measurement vector length mismatch")
    if not str(record.get("device", "")).startswith("cuda:0"):
        raise ValueError("throughput manifest is not CUDA GPU evidence")
    if float(record.get("environment_transitions_per_second", 0.0)) <= 0.0:
        raise ValueError("throughput must be positive")
    minimum_free = min(float(row["gpu_free_fraction"]) for row in record["measurements"])
    if abs(minimum_free - float(record["minimum_gpu_free_fraction"])) > 1.0e-12:
        raise ValueError("minimum GPU free-memory summary mismatch")
    return record


def aggregate_manifests(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    validated = [validate_manifest(record, config) for record in records]
    commits = {record["commit"] for record in validated}
    if len(commits) != 1:
        raise ValueError("all throughput manifests must share one implementation commit")
    rows = []
    expected = {(candidate, repeat) for candidate in config["num_env_candidates"] for repeat in range(3)}
    observed = {(int(record["num_envs"]), int(record["repeat_index"])) for record in validated}
    if observed != expected or len(validated) != len(expected):
        raise ValueError("exactly three unique manifests are required for every candidate")
    for candidate in config["num_env_candidates"]:
        subset = [record for record in validated if int(record["num_envs"]) == candidate]
        rows.append({
            "num_envs": candidate,
            "repeat_throughputs": [float(record["environment_transitions_per_second"]) for record in subset],
            "median_environment_transitions_per_second": statistics.median(
                float(record["environment_transitions_per_second"]) for record in subset
            ),
            "minimum_gpu_free_fraction": min(float(record["minimum_gpu_free_fraction"]) for record in subset),
            "fault_count": sum(len(record["faults"]) for record in subset),
        })
    return rows


def select_num_envs(
    rows: list[dict[str, Any]], *, near_tie_fraction: float, minimum_free: float
) -> int:
    eligible = [
        row for row in rows
        if int(row.get("fault_count", 0)) == 0
        and float(row["minimum_gpu_free_fraction"]) >= float(minimum_free)
    ]
    if not eligible:
        raise ValueError("no throughput candidate satisfies fault and GPU-memory gates")
    fastest = max(float(row["median_environment_transitions_per_second"]) for row in eligible)
    near = [
        row for row in eligible
        if fastest - float(row["median_environment_transitions_per_second"])
        <= fastest * float(near_tie_fraction)
    ]
    return min(int(row["num_envs"]) for row in near)


def summarize(artifact_root: Path, frozen_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_task009d0_config(TASK009D0_CONFIG_PATH)
    paths = sorted(Path(artifact_root).glob("task009d0_throughput_env*_repeat*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = aggregate_manifests(records, config)
    selected = select_num_envs(
        rows,
        near_tie_fraction=config["benchmark"]["near_tie_fraction"],
        minimum_free=config["benchmark"]["minimum_free_memory_fraction"],
    )
    evidence = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in paths
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "version": 1,
        "status": "pass",
        "implementation_commit": records[0]["commit"],
        "config_sha256": config["config_sha256"],
        "candidates": rows,
        "selected_num_envs": selected,
        "source_manifests": evidence,
    }
    summary_path = Path(artifact_root) / "task009d0_throughput_summary.json"
    _atomic_json(summary_path, summary)
    frozen = dict(config)
    frozen["schema"] = TASK009D0_FROZEN_SCHEMA
    frozen["selected_num_envs"] = selected
    frozen["selection"] = {
        "implementation_commit": records[0]["commit"],
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "source_manifest_sha256": [item["sha256"] for item in evidence],
        "rule": "highest median throughput; choose smaller num_envs within 10 percent",
    }
    frozen["config_sha256"] = manifest_hash({key: value for key, value in frozen.items() if key != "config_sha256"})
    _atomic_json(frozen_path, frozen)
    load_task009d0_config(frozen_path, frozen=True)
    return summary, frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--write_frozen_config", type=Path, required=True)
    args = parser.parse_args()
    summary, frozen = summarize(args.artifact_root, args.write_frozen_config)
    print("TASK009D0_GATE5_SUMMARY", json.dumps({
        "status": summary["status"],
        "selected_num_envs": frozen["selected_num_envs"],
        "summary": str((args.artifact_root / "task009d0_throughput_summary.json").resolve()),
        "frozen_config": str(args.write_frozen_config.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
