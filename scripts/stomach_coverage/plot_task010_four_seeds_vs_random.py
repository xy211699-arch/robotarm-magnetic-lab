#!/usr/bin/env python3
"""Plot four update-1000 seed curves against 20-pose R1 and R3 baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
POSE_IDS = (
    "validation-0006", "validation-0011", "validation-0015", "validation-0017",
    "validation-0019", "validation-0035", "validation-0040", "validation-0042",
    "validation-0045", "validation-0046", "validation-0051", "validation-0058",
    "validation-0060", "validation-0063", "validation-0067", "validation-0068",
    "validation-0069", "validation-0092", "validation-0095", "validation-0097",
)
POINTS = 1201
LEGACY_POSE_IDS = POSE_IDS[:5]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_mean(path: Path) -> np.ndarray:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 20 or tuple(row.get("pose_id") for row in rows) != POSE_IDS:
        raise ValueError(f"model curve must contain the frozen ordered 20 poses: {path}")
    curves = np.asarray([row["coverage_fraction"] for row in rows], dtype=np.float64)
    if curves.shape != (20, POINTS) or not np.isfinite(curves).all():
        raise ValueError(f"model curve must be finite 20x1201: {path}")
    if np.any(np.diff(curves, axis=1) < -1.0e-12):
        raise ValueError(f"model coverage is not monotonic: {path}")
    return curves.mean(axis=0)


def load_random_mean(
    run_directory: Path,
    policy_id: str,
    *,
    pose_ids: tuple[str, ...] = POSE_IDS,
    episode_prefix: str = "comparison-",
) -> tuple[np.ndarray, list[Path]]:
    files = [
        Path(run_directory) / "episodes" / f"{episode_prefix}{pose_id}-{policy_id.lower()}.jsonl"
        for pose_id in pose_ids
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(
            f"{policy_id} does not yet have all {len(pose_ids)} completed episodes; missing {len(missing)}"
        )
    curves = []
    for pose_id, path in zip(pose_ids, files, strict=True):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != 3001:
            raise ValueError(f"random episode must contain C0+3000 boundaries: {path}")
        if any(
            row.get("policy_id") != policy_id
            or row.get("pose_id") != pose_id
            or int(row.get("boundary_index", -1)) != index
            or not np.isclose(float(row.get("sim_time_s", np.nan)), index / 10.0, atol=1e-9)
            for index, row in enumerate(rows)
        ):
            raise ValueError(f"random episode identity or 10 Hz boundary mismatch: {path}")
        curve = np.asarray(
            [row["reachable_coverage_fraction"] for row in rows[:POINTS]], dtype=np.float64
        )
        if not np.isfinite(curve).all() or np.any(np.diff(curve) < -1.0e-12):
            raise ValueError(f"random coverage is non-finite or non-monotonic: {path}")
        curves.append(curve)
    return np.asarray(curves).mean(axis=0), files


def latest_random_run(output_root: Path) -> Path:
    pointer_path = Path(output_root) / "latest_formal_manifest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest = Path(pointer["manifest_path"])
    if (
        not manifest.is_file()
        or manifest.stat().st_size != int(pointer["manifest_bytes"])
        or file_sha256(manifest) != pointer["manifest_sha256"]
    ):
        raise ValueError("random comparison latest pointer failed size/SHA-256 validation")
    return manifest.parent


def default_model_paths() -> dict[str, Path]:
    development = (ROOT / "artifacts/task010_cnn_gru/development_seed_991000/latest").resolve()
    candidates = sorted(development.glob("validation_curve_final_*/update_1000/coverage_trajectories.jsonl"))
    if not candidates:
        raise FileNotFoundError("development seed update-1000 trajectory is missing")
    formal = (ROOT / "artifacts/task010_cnn_gru/formal_seeds/latest").resolve()
    return {
        "Seed 991000 / update 1000": candidates[-1],
        **{
            f"Seed {seed} / update 1000": formal / f"seeds/seed_{seed}/validation/coverage_trajectories.jsonl"
            for seed in (991001, 991002, 991003)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-output-root", type=Path,
        default=ROOT / "artifacts/task009c_random_baseline_20pose_comparison",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=None,
    )
    parser.add_argument(
        "--legacy-five-pose-random", action="store_true",
        help="temporarily compare against the original five-pose TASK-009C R1/R3 run",
    )
    parser.add_argument(
        "--legacy-random-run-dir", type=Path,
        default=ROOT.parent / "robotarm_magnetic_lab/artifacts/task009c_random_baseline_preexperiment/formal-20260827_092532_645812Z",
    )
    args = parser.parse_args()

    model_paths = default_model_paths()
    curves = {label: load_model_mean(path) for label, path in model_paths.items()}
    sample_sizes = {label: 20 for label in curves}
    if args.legacy_five_pose_random:
        random_run = args.legacy_random_run_dir
        random_pose_ids = LEGACY_POSE_IDS
        episode_prefix = "formal-"
        random_suffix = " / legacy 5-pose"
    else:
        random_run = latest_random_run(args.random_output_root)
        random_pose_ids = POSE_IDS
        episode_prefix = "comparison-"
        random_suffix = ""
    source_paths = list(model_paths.values())
    for policy_id in ("R1", "R3"):
        curve, paths = load_random_mean(
            random_run, policy_id, pose_ids=random_pose_ids, episode_prefix=episode_prefix
        )
        label = f"Random {policy_id}{random_suffix}"
        curves[label] = curve
        sample_sizes[label] = len(random_pose_ids)
        source_paths.extend(paths)

    output = args.output_dir or ROOT / "artifacts/task010_cnn_gru/comparisons" / (
        "update1000_four_seeds_vs_r1_r3_legacy5"
        if args.legacy_five_pose_random
        else "update1000_four_seeds_vs_r1_r3"
    )
    output.mkdir(parents=True, exist_ok=True)
    time_s = np.arange(POINTS, dtype=np.float64) / 10.0
    csv_path = output / "update1000_four_seeds_vs_r1_r3.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", *(label.lower().replace(" ", "_") for label in curves)))
        for index, timestamp in enumerate(time_s):
            writer.writerow((timestamp, *(curve[index] for curve in curves.values())))

    styles = (
        ("#0072B2", "-", "o"), ("#E69F00", "-", "s"),
        ("#009E73", "-", "^"), ("#CC79A7", "-", "D"),
        ("#D55E00", "--", "v"), ("#4D4D4D", "--", "P"),
    )
    figure, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    for (label, curve), (color, linestyle, marker) in zip(curves.items(), styles, strict=True):
        axis.plot(
            time_s, 100.0 * curve, color=color, linestyle=linestyle, linewidth=2.2,
            marker=marker, markevery=60, markersize=5, label=f"{label} (n={sample_sizes[label]})",
        )
    axis.set_xlabel("Simulation time (s)", fontsize=13)
    axis.set_ylabel("Mean reachable area-weighted cumulative coverage (%)", fontsize=13)
    axis.set_xlim(0.0, 120.0)
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.28)
    axis.legend(loc="lower right", fontsize=10)
    png_path = output / "update1000_four_seeds_vs_r1_r3.png"
    svg_path = output / "update1000_four_seeds_vs_r1_r3.svg"
    figure.savefig(png_path, dpi=300)
    figure.savefig(svg_path)
    plt.close(figure)

    artifacts = [csv_path, png_path, svg_path]
    manifest = {
        "schema": "robotarm_magnetic_lab.task010_four_seeds_vs_random",
        "version": 1,
        "time_range_s": [0.0, 120.0],
        "frequency_hz": 10,
        "points_per_curve": POINTS,
        "pose_ids": list(POSE_IDS),
        "curves": list(curves),
        "curve_sample_sizes": sample_sizes,
        "random_pose_ids": list(random_pose_ids),
        "legacy_five_pose_random": bool(args.legacy_five_pose_random),
        "random_run_directory": str(random_run.resolve()),
        "sources": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in source_paths
        ],
        "outputs": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in artifacts
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
