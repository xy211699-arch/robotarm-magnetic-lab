#!/usr/bin/env python3
"""Validate and summarize TASK-009C random baseline artifacts without repair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(ROOT / "scripts"))

from _artifact_paths import artifact_root
from robotarm_magnetic_lab.baselines.random_policies import POLICY_IDS, load_random_baseline_config
from robotarm_magnetic_lab.coverage.entry_pose_library import file_sha256
from robotarm_magnetic_lab.runtime.task009c_episode_runner import (
    EpisodeProtocolError,
    EpisodeSpec,
    read_episode_jsonl,
    summarize_episode,
    validate_episode_records,
)


ARTIFACT_ROOT = artifact_root(ROOT) / "task009c_random_baseline_preexperiment"
parser = argparse.ArgumentParser(description=__doc__)
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--latest_smoke", action="store_true")
source.add_argument("--latest_formal", action="store_true")
parser.add_argument("--validate_only", action="store_true")
parser.add_argument("--write_figures", action="store_true")
parser.add_argument(
    "--config", type=Path, default=ROOT / "configs/task009c/random_baseline_preexperiment_v1.json"
)
parser.add_argument("--artifact_root", type=Path, default=ARTIFACT_ROOT)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_latest_manifest(root: Path, kind: str) -> tuple[dict, Path, list[dict]]:
    pointer_path = Path(root) / f"latest_{kind}_manifest.json"
    if not pointer_path.is_file():
        raise EpisodeProtocolError(f"latest {kind} pointer is missing: {pointer_path}")
    pointer = _json(pointer_path)
    if pointer.get("kind") != kind:
        raise EpisodeProtocolError("stable pointer kind mismatch")
    manifest_path = Path(pointer["manifest_path"])
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != int(pointer["manifest_bytes"])
        or file_sha256(manifest_path) != pointer["manifest_sha256"]
    ):
        raise EpisodeProtocolError("stable pointer manifest path, size, or hash mismatch")
    return pointer, manifest_path, _jsonl(manifest_path)


def validate_manifest(
    manifest_rows: list[dict], config: dict, kind: str
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    if not manifest_rows or manifest_rows[0].get("record_type") != "run_start":
        raise EpisodeProtocolError("run manifest is missing its append-only start record")
    start = manifest_rows[0]
    if start.get("kind") != kind or start.get("config_sha256") != config["config_sha256"]:
        raise EpisodeProtocolError("run manifest kind or configuration hash mismatch")
    if manifest_rows[-1].get("record_type") != "run_complete":
        raise EpisodeProtocolError("run manifest is not complete")
    expected_records = config["smoke_episodes" if kind == "smoke" else "formal_episodes"]
    expected = {record["episode_id"]: EpisodeSpec.from_record(record) for record in expected_records}
    entries = [row for row in manifest_rows if row.get("record_type") == "episode"]
    if len(entries) != len(expected) or {row["episode_id"] for row in entries} != set(expected):
        raise EpisodeProtocolError("run manifest does not contain the exact configured episode set")
    curves: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for entry in entries:
        episode_id = entry["episode_id"]
        if entry.get("status") != "pass":
            raise EpisodeProtocolError(f"episode is not valid: {episode_id}")
        path = Path(entry["boundary_log_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["boundary_log_bytes"])
            or file_sha256(path) != entry["boundary_log_sha256"]
        ):
            raise EpisodeProtocolError(f"episode log hash mismatch: {episode_id}")
        spec = expected[episode_id]
        records = validate_episode_records(
            read_episode_jsonl(path),
            expected_cycles=spec.action_cycles,
            expected_episode_id=episode_id,
        )
        if records[0]["policy_id"] != spec.policy_id or records[0]["pose_id"] != spec.pose_id:
            raise EpisodeProtocolError(f"episode pairing mismatch: {episode_id}")
        summary = summarize_episode(records)
        summaries[episode_id] = summary
        curves[episode_id] = records
    return curves, summaries


def aggregate_formal(curves: dict[str, list[dict]], config: dict) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    expected_counts = {**{policy: 5 for policy in POLICY_IDS}, "HOLD": 2}
    for policy, count in expected_counts.items():
        selected = [rows for rows in curves.values() if rows[0]["policy_id"] == policy]
        if len(selected) != count:
            raise EpisodeProtocolError(f"{policy} requires exactly {count} aligned formal episodes")
        times = np.asarray([row["sim_time_s"] for row in selected[0]], dtype=np.float64)
        if len(times) != 3001:
            raise EpisodeProtocolError(f"{policy} curve does not contain 3001 points")
        for rows in selected[1:]:
            other = np.asarray([row["sim_time_s"] for row in rows], dtype=np.float64)
            if not np.array_equal(times, other):
                raise EpisodeProtocolError(f"{policy} episode times are not exactly aligned")
        reachable = np.asarray(
            [[row["reachable_coverage_fraction"] for row in rows] for rows in selected],
            dtype=np.float64,
        )
        raw = np.asarray(
            [[row["raw_coverage_fraction"] for row in rows] for rows in selected], dtype=np.float64
        )
        delta = reachable - reachable[:, :1]
        grouped[policy] = {
            "time_s": times,
            "reachable_mean": reachable.mean(axis=0),
            "reachable_std": reachable.std(axis=0),
            "reachable_min": reachable.min(axis=0),
            "reachable_max": reachable.max(axis=0),
            "delta_mean": delta.mean(axis=0),
            "delta_std": delta.std(axis=0),
            "delta_min": delta.min(axis=0),
            "delta_max": delta.max(axis=0),
            "raw_mean": raw.mean(axis=0),
        }
    return grouped


def write_formal_outputs(output: Path, grouped: dict, config: dict, summaries: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    curves_path = output / "mean_curves_10hz.csv"
    policies = [*POLICY_IDS, "HOLD"]
    fields = ["time_s"]
    metrics = (
        "reachable_mean", "reachable_std", "reachable_min", "reachable_max",
        "delta_mean", "delta_std", "delta_min", "delta_max", "raw_mean",
    )
    for policy in policies:
        fields.extend(f"{policy}_{metric}" for metric in metrics)
    with curves_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, time_s in enumerate(grouped["R1"]["time_s"]):
            row = {"time_s": float(time_s)}
            for policy in policies:
                for metric in metrics:
                    row[f"{policy}_{metric}"] = float(grouped[policy][metric][index])
            writer.writerow(row)
    candidate_path = output / "candidate_times.csv"
    with candidate_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["policy_id", "n", "time_s", "mean", "std", "minimum", "maximum"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for policy in policies:
            times = grouped[policy]["time_s"]
            for candidate in config["candidate_times_s"]:
                matches = np.flatnonzero(times == float(candidate))
                if len(matches) != 1:
                    raise EpisodeProtocolError(f"candidate time is not an exact boundary: {candidate}")
                index = int(matches[0])
                writer.writerow(
                    {
                        "policy_id": policy,
                        "n": 2 if policy == "HOLD" else 5,
                        "time_s": candidate,
                        "mean": grouped[policy]["reachable_mean"][index],
                        "std": grouped[policy]["reachable_std"][index],
                        "minimum": grouped[policy]["reachable_min"][index],
                        "maximum": grouped[policy]["reachable_max"][index],
                    }
                )
    summaries_path = output / "episode_summaries.json"
    _write_json(summaries_path, summaries)
    return {
        "mean_curves_csv": _artifact(curves_path),
        "candidate_times_csv": _artifact(candidate_path),
        "episode_summaries": _artifact(summaries_path),
    }


def write_figures(output: Path, grouped: dict, config: dict) -> dict:
    import matplotlib.pyplot as plt

    policies = [*POLICY_IDS, "HOLD"]
    figure, axis = plt.subplots(figsize=(12, 7))
    audit, audit_axis = plt.subplots(figsize=(12, 7))
    for policy in policies:
        style = config["curve_styles"][policy]
        sample = slice(None, None, 10)
        label = f"{policy} (n={2 if policy == 'HOLD' else 5})"
        kwargs = {
            "color": style["color"],
            "linestyle": style["linestyle"],
            "marker": style["marker"],
            "markevery": 10,
            "markersize": 4,
            "label": label,
        }
        axis.plot(
            grouped[policy]["time_s"][sample],
            100.0 * grouped[policy]["reachable_mean"][sample],
            **kwargs,
        )
        audit_axis.plot(
            grouped[policy]["time_s"][sample],
            100.0 * grouped[policy]["delta_mean"][sample],
            **kwargs,
        )
    for target, ylabel in (
        (axis, "Reachable area-weighted cumulative coverage (%)"),
        (audit_axis, "Coverage gain from C0 (percentage points)"),
    ):
        target.set_xlabel("Simulation time (s)")
        target.set_ylabel(ylabel)
        target.grid(True, alpha=0.25)
        target.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.tight_layout()
    audit.tight_layout()
    png = output / "random_baseline_mean_coverage.png"
    svg = output / "random_baseline_mean_coverage.svg"
    delta_png = output / "random_baseline_mean_delta.png"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    audit.savefig(delta_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    plt.close(audit)
    return {"main_png": _artifact(png), "main_svg": _artifact(svg), "delta_png": _artifact(delta_png)}


def _artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def main() -> int:
    args = parser.parse_args()
    kind = "smoke" if args.latest_smoke else "formal"
    config = load_random_baseline_config(args.config)
    pointer, manifest_path, manifest_rows = load_latest_manifest(args.artifact_root, kind)
    curves, summaries = validate_manifest(manifest_rows, config, kind)
    result = {
        "status": "pass",
        "kind": kind,
        "config_sha256": config["config_sha256"],
        "run_id": pointer["run_id"],
        "manifest": _artifact(manifest_path),
        "episode_count": len(curves),
    }
    if kind == "formal":
        grouped = aggregate_formal(curves, config)
        if args.write_figures:
            output = manifest_path.parent / "summary"
            result["tables"] = write_formal_outputs(output, grouped, config, summaries)
            result["figures"] = write_figures(output, grouped, config)
    print("TASK009C_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
