"""Render the wall-distance comparison from the verified result tables."""

from collections import defaultdict
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parents[1]
MODELS = ("dinov3", "radio-v2.5-b", "siglip2-base")
NAMES = ("DINOv3", "RADIOv2.5-B", "SigLIP2-B")
COLORS = ("#285680", "#14847C", "#B46248")
GRAY, INK, GRID = "#687582", "#182632", "#E5E9ED"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def rows(name):
    with (HERE / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=8, fontweight="bold")
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-16, 8), textcoords="offset points",
                fontweight="bold", fontsize=10, va="baseline")


def main():
    results = {r["encoder"]: r for r in rows("summary.csv") if r["task"] == "wall-distance"}
    intervals = defaultdict(list)
    for row in rows("wall-distance-intervals.csv"):
        intervals[row["encoder"]].append(row)
    fig = plt.figure(figsize=(7.2, 2.9), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.24), height_ratios=(1, 0.18))
    left, right = (fig.add_subplot(grid[0, column]) for column in range(2))
    n = results[MODELS[0]]["test_n"]
    panel(left, "a", f"All held-out frames  |  n = {n}")
    for i, (model, color) in enumerate(zip(MODELS, COLORS, strict=True)):
        mae, rmse = (float(results[model][f"test_{key}_mm"]) for key in ("mae", "rmse"))
        left.hlines(i, mae, rmse, color=color, linewidth=0.8)
        left.scatter(mae, i, color=color, s=25)
        left.scatter(rmse, i, facecolor="white", edgecolor=color, s=25)
    left.scatter([], [], color=GRAY, s=23, label="MAE")
    left.scatter([], [], facecolor="white", edgecolor=GRAY, s=23, label="RMSE")
    upper = 1.09 * max(float(r["test_rmse_mm"]) for r in results.values())
    left.set(xlim=(0, upper), ylim=(-0.5, 2.5), yticks=range(3), yticklabels=NAMES, xlabel="Error (mm)")
    left.invert_yaxis()
    left.xaxis.set_major_locator(MaxNLocator(nbins=5))
    left.grid(axis="x", color=GRID, linewidth=0.55)

    panel(right, "b", "Absolute error by distance interval")
    first = intervals[MODELS[0]]
    x = np.arange(len(first))
    for model, title, color, mark in zip(MODELS, NAMES, COLORS, ("o", "s", "^"), strict=True):
        right.plot(x, [float(r["mae"]) for r in intervals[model]], marker=mark, color=color,
                   markersize=4, linewidth=1.1, label=title)
    right.plot(x, [float(r["mean_baseline_mae_mm"]) for r in first], "--D", color=GRAY,
               mfc="white", markersize=3.6, linewidth=0.9, label="Train mean")
    ticks = [f'{r["lower_mm"]}–{r["upper_mm"]}\n(n = {r["n"]})' for r in first]
    right.set(xticks=x, xticklabels=ticks, ylim=(0, None), ylabel="MAE (mm)", xlabel="Ground-truth distance (mm)")
    right.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    right.grid(axis="y", color=GRID, linewidth=0.55)
    for column, axis in enumerate((left, right)):
        axis.set_axisbelow(True)
        legend = fig.add_subplot(grid[1, column])
        legend.set_axis_off()
        legend.legend(*axis.get_legend_handles_labels(), loc="center", ncol=2)
    fig.savefig(HERE / "wall-distance-comparison.png", dpi=360, facecolor="white")
    fig.savefig(HERE / "wall-distance-comparison.pdf", dpi=360, facecolor="white")
    fig.savefig(HERE / "wall-distance-comparison.svg", dpi=360, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
