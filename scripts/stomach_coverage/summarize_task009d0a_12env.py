#!/usr/bin/env python3
"""Summarize the incremental 12-env candidate and write frozen config v2."""

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
    TASK009D0_FROZEN_SCHEMA,
    load_task009d0_config,
)


RARE_C0_ABORT = "RuntimeError: TASK-009D0 reset produced non-positive initial C0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def decide(
    candidate: dict[str, Any], throughputs: list[float], minimum_free: float, fault_count: int,
    *, observed_process_count: int = 3,
) -> int:
    if observed_process_count != int(candidate["independent_process_repeats"]):
        raise ValueError("exactly three 12-env process results are required")
    if fault_count:
        return 8
    if len(throughputs) != observed_process_count:
        raise ValueError("fault-free 12-env results must all provide throughput")
    eligible = fault_count == 0 and minimum_free >= float(candidate["minimum_gpu_free_memory_fraction"])
    threshold = float(candidate["baseline_median_environment_transitions_per_second"]) * (
        1.0 + float(candidate["minimum_relative_improvement"])
    )
    return 12 if eligible and statistics.median(throughputs) > threshold else 8


def is_accepted_rare_c0_abort(records: list[dict[str, Any]]) -> bool:
    """Recognize only the explicitly accepted, pre-measurement C0 reset abort."""
    failed = [record for record in records if record.get("status") == "fail"]
    return (
        len(failed) == 1
        and failed[0].get("faults") == [RARE_C0_ABORT]
        and failed[0].get("measurements") == []
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--candidate_config", type=Path, required=True)
    parser.add_argument("--base_frozen_config", type=Path, required=True)
    parser.add_argument("--write_frozen_config", type=Path, required=True)
    parser.add_argument(
        "--accept_rare_nonpositive_c0_abort",
        action="store_true",
        help="Apply the explicit 2026-08-28 user waiver for one known pre-measurement C0 reset abort.",
    )
    parser.add_argument("--kit_args", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    candidate = json.loads(args.candidate_config.read_text(encoding="utf-8"))
    paths = sorted(args.artifact_root.glob("task009d0_throughput_env12_repeat*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(records) != 3 or {int(row["repeat_index"]) for row in records} != {0, 1, 2}:
        raise ValueError("12-env results must contain unique repeats 0, 1 and 2")
    if any(int(row["num_envs"]) != 12 or row.get("smoke") for row in records):
        raise ValueError("formal 12-env manifests are required")
    if len({row["commit"] for row in records}) != 1:
        raise ValueError("all 12-env runs must share one implementation commit")
    if any(int(row["warmup_steps"]) != 50 or int(row["measured_steps"]) != 300 for row in records):
        raise ValueError("12-env Gate 5 timing protocol mismatch")
    successful = [row for row in records if row.get("status") == "pass"]
    throughputs = [float(row["environment_transitions_per_second"]) for row in successful]
    minimum_free = min((float(row["minimum_gpu_free_fraction"]) for row in successful), default=0.0)
    fault_count = sum(len(row.get("faults", ())) for row in records)
    if any(row.get("status") not in ("pass", "fail") for row in records):
        raise ValueError("12-env manifest status is invalid")
    threshold = float(candidate["baseline_median_environment_transitions_per_second"]) * 1.10
    rare_abort_waived = bool(
        args.accept_rare_nonpositive_c0_abort
        and is_accepted_rare_c0_abort(records)
        and candidate.get("accepted_rare_reset_abort")
        == "TASK-009D0 reset produced non-positive initial C0"
    )
    if rare_abort_waived:
        selected = (
            12
            if len(successful) >= 2
            and statistics.median(throughputs) > threshold
            and minimum_free >= float(candidate["minimum_gpu_free_memory_fraction"])
            else 8
        )
    else:
        selected = decide(
            candidate, throughputs, minimum_free, fault_count, observed_process_count=len(records)
        )
    evidence = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in paths
    ]
    summary = {
        "schema": "robotarm_magnetic_lab.task009d0a_12env_throughput_summary",
        "version": 1,
        "status": (
            "candidate_selected_with_user_waiver"
            if selected == 12 and rare_abort_waived
            else "candidate_selected" if selected == 12 else "candidate_rejected"
        ),
        "implementation_commit": records[0]["commit"],
        "candidate_config_path": str(args.candidate_config.resolve()),
        "candidate_config_sha256": file_sha256(args.candidate_config),
        "baseline_num_envs": 8,
        "baseline_median_environment_transitions_per_second": float(
            candidate["baseline_median_environment_transitions_per_second"]
        ),
        "required_strictly_greater_than": threshold,
        "candidate_num_envs": 12,
        "repeat_throughputs": throughputs,
        "successful_process_count": len(successful),
        "failed_process_count": len(records) - len(successful),
        "median_environment_transitions_per_second": statistics.median(throughputs) if throughputs else None,
        "relative_improvement_over_8env": (
            statistics.median(throughputs) /
            float(candidate["baseline_median_environment_transitions_per_second"]) - 1.0
            if throughputs else None
        ),
        "minimum_gpu_free_fraction": minimum_free,
        "fault_count": fault_count,
        "rare_nonpositive_c0_abort_user_waiver": rare_abort_waived,
        "waiver_policy": (
            "abort the affected run; no resampling and no pose-library modification"
            if rare_abort_waived else None
        ),
        "selected_num_envs": selected,
        "source_manifests": evidence,
    }
    summary_path = args.artifact_root / "task009d0a_12env_throughput_summary.json"
    _atomic_json(summary_path, summary)

    frozen = load_task009d0_config(args.base_frozen_config, frozen=True)
    frozen["num_env_candidates"] = [1, 2, 4, 8, 12]
    frozen["selected_num_envs"] = selected
    frozen["selection"] = {
        "implementation_commit": records[0]["commit"],
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "source_manifest_sha256": [item["sha256"] for item in evidence],
        "rule": (
            "user-authorized 2026-08-28 waiver: accept one known pre-measurement non-positive-C0 reset abort; "
            "successful median > 1.10 * 30.310967 and minimum free GPU fraction >= 0.20"
            if rare_abort_waived
            else "select 12 only if median > 1.10 * 30.310967, zero faults, and minimum free GPU fraction >= 0.20"
        ),
        "rare_nonpositive_c0_abort_user_waiver": rare_abort_waived,
        "runtime_policy": "abort an affected run; do not resample or modify the pose library",
    }
    frozen["config_sha256"] = manifest_hash({key: value for key, value in frozen.items() if key != "config_sha256"})
    _atomic_json(args.write_frozen_config, frozen)
    load_task009d0_config(args.write_frozen_config, frozen=True)
    print("TASK009D0A_GATE5_SUMMARY", json.dumps({
        "selected_num_envs": selected,
        "median_throughput": summary["median_environment_transitions_per_second"],
        "minimum_gpu_free_fraction": minimum_free,
        "summary": str(summary_path.resolve()),
        "frozen_config": str(args.write_frozen_config.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
