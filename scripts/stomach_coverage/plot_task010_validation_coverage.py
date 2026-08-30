#!/usr/bin/env python3
"""Plot mean frozen-pose coverage trajectories for TASK-010 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_mean(path: Path) -> np.ndarray:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 20 or len({row["pose_id"] for row in records}) != 20:
        raise ValueError(f"expected twenty unique validation poses: {path}")
    curves = np.asarray([row["coverage_fraction"] for row in records], dtype=np.float64)
    if curves.shape != (20, 1201) or not np.isfinite(curves).all():
        raise ValueError(f"invalid 20x1201 coverage trajectories: {path}")
    if np.any(np.diff(curves, axis=1) < -1.0e-12):
        raise ValueError(f"coverage trajectory is not monotonic: {path}")
    return curves.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    updates = (250, 500, 750, 1000)
    means = {
        update: load_mean(args.validation_dir / f"update_{update:04d}" / "coverage_trajectories.jsonl")
        for update in updates
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "checkpoint_mean_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("simulation_time_s", *(f"update_{update:04d}_mean_coverage_fraction" for update in updates)))
        for index in range(1201):
            writer.writerow((index / 10.0, *(means[update][index] for update in updates)))

    styles = (("#0072B2", "o"), ("#E69F00", "s"), ("#009E73", "^"), ("#CC79A7", "D"))
    fig, axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    time_s = np.arange(1201, dtype=np.float64) / 10.0
    for (update, curve), (color, marker) in zip(means.items(), styles, strict=True):
        axis.plot(time_s, 100.0 * curve, color=color, linewidth=2.2, marker=marker,
                  markevery=60, markersize=5, label=f"Update {update} (n=20)")
    axis.set_xlabel("Simulation time (s)", fontsize=13)
    axis.set_ylabel("Mean reachable area-weighted cumulative coverage (%)", fontsize=13)
    axis.set_xlim(0.0, 120.0)
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, alpha=0.28)
    axis.legend(loc="lower right", fontsize=11)
    fig.savefig(args.output_dir / "checkpoint_mean_coverage.png", dpi=220)
    fig.savefig(args.output_dir / "checkpoint_mean_coverage.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
