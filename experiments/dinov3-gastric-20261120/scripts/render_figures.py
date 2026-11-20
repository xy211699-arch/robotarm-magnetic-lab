"""Render the paper figures from the published run records and original camera frames."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("normal", "raised", "ulcerated")
TITLES = ("Normal", "Raised", "Ulcerated")
BLUE, TEAL, CLAY = "#285680", "#14847C", "#B46248"
GRAY, INK, GRID = "#687582", "#182632", "#E5E9ED"
HEAT = LinearSegmentedColormap.from_list("count", ["#F6F8FA", "#B5CFDF", BLUE])
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def records(relative):
    with (ROOT / relative).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def data(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def panel(ax, letter, title):
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-18, 8), textcoords="offset points",
                fontweight="bold", fontsize=10, va="baseline")


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=360, facecolor="white", pil_kwargs={"optimize": True})
    fig.savefig(output / f"{name}.pdf", dpi=320, facecolor="white")
    fig.savefig(output / f"{name}.svg", dpi=320, facecolor="white")
    plt.close(fig)


def lesion_data(output, image_root):
    samples = records("lesion/figure-samples.csv")
    original_split = [
        row for row in records("lesion/split-manifest.csv") if row["sample_id"] == row["parent_sample_id"]
    ]
    counts = Counter((row["label"], row["split"]) for row in original_split)
    fig = plt.figure(figsize=(7.2, 2.85), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.48, 1), wspace=0.1)
    photo_grid = grid[0].subgridspec(2, 3, wspace=0.045, hspace=0.04)
    for column, (label, title) in enumerate(zip(CLASSES, TITLES, strict=True)):
        for row_index, sample in enumerate(r for r in samples if r["label"] == label):
            path = image_root / "lesion" / sample["image_path"]
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sample["sha256"]:
                    raise ValueError(f"Original frame SHA-256 mismatch: {path}")
            ax = fig.add_subplot(photo_grid[row_index, column])
            with Image.open(path) as frame:
                ax.imshow(frame.convert("RGB"), interpolation="antialiased")
            ax.set_axis_off()
            if row_index == 0:
                ax.set_title(title, fontweight="bold")
                if column == 0:
                    ax.annotate("a", (0, 1), xycoords="axes fraction", xytext=(-7, 4),
                                textcoords="offset points", fontweight="bold", fontsize=10)
            ax.text(0.5, -0.025, f'{float(sample["distance_mm"]):.1f} mm',
                    transform=ax.transAxes, ha="center", va="top", fontsize=7)
    ax = fig.add_subplot(grid[1])
    panel(ax, "b", "Original frames by split")
    left = np.zeros(len(CLASSES))
    for split, text, color in (("train", "Train", BLUE), ("val", "Validation", TEAL), ("test", "Test", CLAY)):
        values = np.array([counts[label, split] for label in CLASSES])
        ax.barh(TITLES, values, left=left, height=0.49, color=color, label=text, zorder=3)
        for pos, start, value in zip(range(len(CLASSES)), left, values, strict=True):
            ax.text(start + value / 2, pos, str(value), color="white", ha="center", va="center", fontsize=7)
        left += values
    ax.set(xlim=(0, 107), xlabel="Original frames", xticks=[0, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=3, handlelength=1, columnspacing=1)
    save(fig, output, "lesion-data")


def distance_data(output):
    capped = records("wall-distance/cap-bins.csv")
    augmented = records("wall-distance/augmentation-bins.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
    edges = [int(row["lower_mm"]) for row in capped] + [int(capped[-1]["upper_mm"])]
    panel(left, "a", "Candidate selection")
    left.stairs([int(r["before"]) for r in capped], edges, color=GRAY, linestyle="--",
                linewidth=1.05, label="Candidates (411)")
    left.stairs([int(r["after"]) for r in capped], edges, color=BLUE, fill=True,
                alpha=0.22, label="Retained (312)")
    left.stairs([int(r["after"]) for r in capped], edges, color=BLUE, linewidth=1.1)
    left.axhline(40, color=CLAY, lw=0.8, ls=":", label="Cap: 40 frames")
    left.set(ylabel="Original frames per 2 mm", ylim=(0, 105))
    left.legend(loc="upper left", ncol=1)
    panel(right, "b", "Training-label reweighting")
    original = np.array([int(r["source_train"]) for r in augmented])
    final = np.array([int(r["actual_train"]) for r in augmented])
    target = [100 * float(r["target_probability"]) for r in augmented]
    right.stairs(100 * original / original.sum(), edges, color=GRAY, linestyle="--",
                 linewidth=1.05, label="Original (208)")
    right.bar(edges[:-1], 100 * final / final.sum(), align="edge", width=2, color="#B4D8D4",
              edgecolor="white", linewidth=0.4, label="With augmentation (2,000)")
    right.plot(np.array(edges[:-1]) + 1, target, "o", color=TEAL, markersize=2.6,
               label="Truncated-normal target")
    right.set(ylabel="Share of training frames (%)", ylim=(0, 18.5))
    right.legend(loc="upper left", ncol=1)
    for ax in (left, right):
        ax.set(xlim=(30, 60), xlabel="Ground-truth distance (mm)", xticks=[30, 40, 50, 60])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.55)
    save(fig, output, "wall-distance-data")


def confusion(ax, metrics, letter, title, vmax):
    matrix = np.array(metrics["confusion_matrix"])
    panel(ax, letter, f"{title}  |  n = {matrix.sum()}")
    ax.imshow(matrix, cmap=HEAT, vmin=0, vmax=vmax)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=TITLES, yticklabels=TITLES)
    ax.set(xlabel="Predicted class", ylabel="Capture label")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                color="white" if matrix[i, j] > 0.62 * vmax else INK)


def results(output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.65), layout="constrained")
    validation = data("lesion/frozen-selection.json")["validation"]
    test = data("lesion/test-metrics.json")
    vmax = max(max(row) for m in (validation, test) for row in m["confusion_matrix"])
    confusion(axes[0, 0], validation, "a", "Lesion validation", vmax)
    confusion(axes[0, 1], test, "b", "Lesion test", vmax)

    ax = axes[1, 0]
    panel(ax, "c", "Wall-distance test predictions")
    readings = records("wall-distance/test-predictions.csv")
    truth = np.array([float(r["distance_true_mm"]) for r in readings])
    prediction = np.array([float(r["distance_predicted_mm"]) for r in readings])
    bounds = (min(25, 5 * np.floor(prediction.min() / 5)), max(65, 5 * np.ceil(prediction.max() / 5)))
    ax.plot(bounds, bounds, linestyle="--", color=GRAY, lw=0.85, label="Identity")
    ax.scatter(truth, prediction, color=BLUE, edgecolors="white", linewidths=0.35, s=17, alpha=0.8, zorder=3)
    ax.set(xlim=bounds, ylim=bounds, xlabel="Ground truth (mm)", ylabel="Prediction (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    panel(ax, "d", "Wall-distance error by interval")
    wall = data("wall-distance/test-metrics.json")
    groups = wall["per_interval"]
    intervals = [f'{r["lower_mm"]}–{r["upper_mm"]} mm  ·  n = {r["n"]}' for r in groups]
    y = np.arange(len(groups))
    mae = [r["mae"] for r in groups]
    baseline = [r["mean_baseline"]["mae"] for r in groups]
    ax.hlines(y, np.minimum(mae, baseline), np.maximum(mae, baseline), color="#ADB7BF", lw=1.2)
    ax.scatter(mae, y, s=32, color=BLUE, label="DINOv3 + Ridge", zorder=3)
    ax.scatter(baseline, y, s=32, marker="s", facecolor="white", edgecolor=GRAY, lw=1, label="Train-mean baseline", zorder=3)
    ax.set(xlim=(0, 11), ylim=(-0.6, 2.6), yticks=y, yticklabels=intervals, xlabel="Mean absolute error (mm)")
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, columnspacing=1)
    save(fig, output, "held-out-results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True, help="Root produced by restore_originals.py")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lesion_data(args.output, args.image_root)
    distance_data(args.output)
    results(args.output)


if __name__ == "__main__":
    main()
