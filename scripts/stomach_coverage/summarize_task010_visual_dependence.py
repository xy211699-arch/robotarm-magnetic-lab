#!/usr/bin/env python3
"""Paired visual-dependence metrics and hierarchical bootstrap summarizer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robotarm_magnetic_lab.runtime.task010_visual_dependence_config import (
    VisualDependenceConfig,
)


THRESHOLDS = (0.80, 0.90, 0.95)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_curve(coverage: Sequence[float]) -> np.ndarray:
    values = np.asarray(coverage, dtype=np.float64)
    if values.shape != (1201,):
        raise ValueError("visual-dependence coverage curve must contain 1201 points")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("visual-dependence coverage curve is invalid")
    if np.any(np.diff(values) < -1.0e-12):
        raise ValueError("visual-dependence coverage curve is not monotonic")
    return values


def episode_metrics(coverage: Sequence[float]) -> Mapping[str, float | None]:
    curve = _finite_curve(coverage)
    metrics: dict[str, float | None] = {
        "nAUC_120": float(np.trapezoid(curve, dx=0.1) / 120.0),
        "C30": float(curve[300]),
        "C60": float(curve[600]),
        "C120": float(curve[1200]),
    }
    for threshold in THRESHOLDS:
        reached = np.flatnonzero(curve >= threshold)
        if reached.size:
            metrics[f"time_to_{int(threshold * 100)}"] = float(reached[0]) / 10.0
            metrics[f"reached_{int(threshold * 100)}"] = True
        else:
            metrics[f"time_to_{int(threshold * 100)}"] = None
            metrics[f"reached_{int(threshold * 100)}"] = False
    return metrics


def _paired_differences(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], float]:
    conditions = {str(row["condition"]) for row in rows}
    if "normal" not in conditions:
        raise ValueError("paired bootstrap requires normal baseline rows")
    comparisons = sorted(conditions - {"normal"})
    if len(comparisons) != 1:
        raise ValueError("paired bootstrap requires exactly one non-normal condition")
    comparison = comparisons[0]
    by_key: dict[tuple[int, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["pose_id"]))
        by_key.setdefault(key, {}).setdefault(str(row["condition"]), list(row["coverage_fraction"]))
    differences: dict[tuple[int, str], float] = {}
    for key, condition_curves in by_key.items():
        if set(condition_curves) != {"normal", comparison}:
            raise ValueError(f"paired bootstrap key is missing a condition: {key}")
        baseline = episode_metrics(condition_curves["normal"])["nAUC_120"]
        treatment = episode_metrics(condition_curves[comparison])["nAUC_120"]
        assert baseline is not None and treatment is not None
        differences[key] = float(baseline - treatment)
    return differences


def _per_seed_means(
    differences: Mapping[tuple[int, str], float],
) -> dict[int, float]:
    by_seed: dict[int, list[float]] = {}
    for (seed, _), value in differences.items():
        by_seed.setdefault(seed, []).append(value)
    return {seed: float(np.mean(values)) for seed, values in by_seed.items()}


def _leave_one_pose_out_means(
    rows: Sequence[Mapping[str, Any]],
    comparison_condition: str,
) -> list[float]:
    baseline = {
        (row["seed"], row["pose_id"]): row["coverage_fraction"]
        for row in rows
        if row["condition"] == "normal"
    }
    treatment = {
        (row["seed"], row["pose_id"]): row["coverage_fraction"]
        for row in rows
        if row["condition"] == comparison_condition
    }
    if set(baseline) != set(treatment):
        raise ValueError("leave-one-pose-out condition keys differ")
    keys = sorted(baseline)
    pose_ids = sorted({pose_id for _, pose_id in keys})
    means = []
    for excluded_pose in pose_ids:
        values = []
        for key in keys:
            if key[1] == excluded_pose:
                continue
            base = episode_metrics(baseline[key])["nAUC_120"]
            treat = episode_metrics(treatment[key])["nAUC_120"]
            assert base is not None and treat is not None
            values.append(float(base - treat))
        if values:
            means.append(float(np.mean(values)))
    return means


def hierarchical_paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> Mapping[str, float]:
    differences = _paired_differences(rows)
    seeds = sorted({key[0] for key in differences})
    pose_ids_by_seed = {
        current_seed: sorted({pose_id for (item_seed, pose_id) in differences if item_seed == current_seed})
        for current_seed in seeds
    }
    rng = np.random.default_rng(seed)
    sampled_means = []
    for _ in range(int(replicates)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        values = []
        for sampled_seed in sampled_seeds:
            pose_ids = pose_ids_by_seed[int(sampled_seed)]
            sampled_poses = rng.choice(pose_ids, size=len(pose_ids), replace=True)
            values.extend(differences[(int(sampled_seed), str(pose_id))] for pose_id in sampled_poses)
        sampled_means.append(float(np.mean(values)))
    sampled_means = np.asarray(sampled_means, dtype=np.float64)
    per_seed = _per_seed_means(differences)
    return {
        "independent_seed_count": len(seeds),
        "paired_pose_count_per_seed": len(pose_ids_by_seed[seeds[0]]),
        "per_seed_means": {str(key): value for key, value in per_seed.items()},
        "mean": float(np.mean(sampled_means)),
        "std": float(np.std(sampled_means, ddof=0)),
        "ci95_low": float(np.percentile(sampled_means, 2.5)),
        "ci95_high": float(np.percentile(sampled_means, 97.5)),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _condition_and_update_from_path(path: Path) -> tuple[str, int, int] | None:
    parts = path.parts
    if "validation" not in parts:
        return None
    validation_index = parts.index("validation")
    if len(parts) < validation_index + 5:
        return None
    update_part = parts[validation_index + 1]
    condition = parts[validation_index + 2]
    seed_part = parts[validation_index + 3]
    if not update_part.startswith("update") or not seed_part.startswith("seed_"):
        return None
    try:
        update = int(update_part.removeprefix("update"))
        seed = int(seed_part.removeprefix("seed_"))
    except ValueError:
        return None
    return condition, update, seed


def _load_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    input_path = run_dir / "summary_input.jsonl"
    if input_path.is_file():
        return [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    rows = []
    for trajectory_path in sorted(run_dir.glob("validation/*/*/seed_*/coverage_trajectories.jsonl")):
        parsed = _condition_and_update_from_path(trajectory_path)
        if parsed is None:
            raise ValueError(f"cannot parse visual-dependence path: {trajectory_path}")
        condition, update, seed = parsed
        trajectories = [
            json.loads(line)
            for line in trajectory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for trajectory in trajectories:
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "pose_id": trajectory["pose_id"],
                    "update": update,
                    "coverage_fraction": trajectory["coverage_fraction"],
                }
            )
    return rows


def _validate_summary_rows(rows: list[dict[str, Any]], config: VisualDependenceConfig) -> None:
    required = {"condition", "seed", "pose_id", "update", "coverage_fraction"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("visual-dependence summary rows are missing required fields")
    keys = [
        (row["condition"], int(row["seed"]), str(row["pose_id"]), int(row["update"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("visual-dependence summary contains duplicate condition/seed/pose/update rows")
    primary = [row for row in rows if int(row["update"]) == config.primary_update]
    sensitivity = [row for row in rows if int(row["update"]) == config.sensitivity_update]
    if len(primary) != 240 or len(sensitivity) != 120:
        raise ValueError(
            f"expected 240 primary and 120 sensitivity rows, got {len(primary)}/{len(sensitivity)}"
        )
    primary_conditions = {row["condition"] for row in primary}
    sensitivity_conditions = {row["condition"] for row in sensitivity}
    if primary_conditions != set(config.primary_conditions):
        raise ValueError("primary visual-dependence condition set mismatch")
    if sensitivity_conditions != set(config.sensitivity_conditions):
        raise ValueError("sensitivity visual-dependence condition set mismatch")
    for row in rows:
        episode_metrics(row["coverage_fraction"])


def summarize_visual_dependence(
    run_dir: Path,
    config: VisualDependenceConfig,
) -> Mapping[str, object]:
    run_dir = Path(run_dir)
    rows = _load_summary_rows(run_dir)
    if not rows:
        raise FileNotFoundError("visual-dependence summary input is missing")
    _validate_summary_rows(rows, config)

    primary_rows = [row for row in rows if int(row["update"]) == config.primary_update]
    sensitivity_rows = [row for row in rows if int(row["update"]) == config.sensitivity_update]
    if len(primary_rows) != 240 or len(sensitivity_rows) != 120:
        raise ValueError(
            f"expected 240 primary and 120 sensitivity rows, got {len(primary_rows)}/{len(sensitivity_rows)}"
        )

    metric_rows = []
    for row in rows:
        metric_row = dict(row)
        metric_row.pop("coverage_fraction")
        metric_row.update(episode_metrics(row["coverage_fraction"]))
        metric_rows.append(metric_row)
    _write_csv(run_dir / "condition_metrics.csv", metric_rows)

    primary_normal = [row for row in primary_rows if row["condition"] == "normal"]
    primary_blind = [row for row in primary_rows if row["condition"] == "blind"]
    primary_donor = [row for row in primary_rows if row["condition"] == "donor"]
    primary_first_frame = [row for row in primary_rows if row["condition"] == "first_frame"]
    effects = {}
    for name, comparison_rows in (("B0-B1", primary_blind), ("B0-I1", primary_donor)):
        bootstrap = hierarchical_paired_bootstrap(
            primary_normal + comparison_rows,
            seed=config.bootstrap_seed,
            replicates=config.bootstrap_replicates,
        )
        comparison_condition = comparison_rows[0]["condition"]
        leave_one = _leave_one_pose_out_means(
            primary_normal + comparison_rows,
            comparison_condition,
        )
        effects[name] = {
            "confirmatory": True,
            "bootstrap": bootstrap,
            "all_seed_directions_positive": all(
                value > 0.0 for value in bootstrap["per_seed_means"].values()
            ),
            "ci_excludes_zero": bootstrap["ci95_low"] > 0.0,
            "all_leave_one_pose_out_positive": all(value > 0.0 for value in leave_one),
            "leave_one_pose_out_means": leave_one,
        }
        effects[name]["claim_gate_passed"] = bool(
            effects[name]["all_seed_directions_positive"]
            and effects[name]["ci_excludes_zero"]
            and effects[name]["all_leave_one_pose_out_positive"]
        )

    paired_differences = []
    for condition, comparison_rows in (("blind", primary_blind), ("donor", primary_donor)):
        normal_by_key = {
            (row["seed"], row["pose_id"]): row["coverage_fraction"]
            for row in primary_normal
        }
        for row in comparison_rows:
            key = (row["seed"], row["pose_id"])
            base = episode_metrics(normal_by_key[key])["nAUC_120"]
            treat = episode_metrics(row["coverage_fraction"])["nAUC_120"]
            assert base is not None and treat is not None
            paired_differences.append(
                {
                    "comparison": f"B0-{condition.upper() if condition != 'blind' else 'B1'}",
                    "seed": key[0],
                    "pose_id": key[1],
                    "update": config.primary_update,
                    "difference_nAUC_120": float(base - treat),
                }
            )
    _write_csv(run_dir / "paired_episode_differences.csv", paired_differences)

    mean_curves = []
    for condition in config.primary_conditions:
        for update, rows_subset in (
            (config.primary_update, primary_rows),
            (config.sensitivity_update, sensitivity_rows),
        ):
            condition_rows = [row for row in rows_subset if row["condition"] == condition]
            if not condition_rows:
                continue
            curves = np.asarray([row["coverage_fraction"] for row in condition_rows], dtype=np.float64)
            mean = curves.mean(axis=0)
            for index, value in enumerate(mean):
                mean_curves.append(
                    {
                        "condition": condition,
                        "update": update,
                        "time_s": index / 10.0,
                        "mean_coverage": float(value),
                        "analysis_kind": (
                            "primary" if update == config.primary_update else "secondary"
                        ),
                    }
                )
    _write_csv(run_dir / "mean_curves_10hz.csv", mean_curves)
    _write_json(run_dir / "confirmatory_effects.json", effects)

    output_paths = [
        run_dir / "condition_metrics.csv",
        run_dir / "paired_episode_differences.csv",
        run_dir / "confirmatory_effects.json",
        run_dir / "mean_curves_10hz.csv",
    ]
    audit = {
        "schema": "robotarm_magnetic_lab.task010_visual_dependence_summary_audit",
        "primary_update": config.primary_update,
        "sensitivity_update": config.sensitivity_update,
        "primary_rows": len(primary_rows),
        "sensitivity_rows": len(sensitivity_rows),
        "artifacts": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in output_paths
        ],
    }
    _write_json(run_dir / "artifact_audit.json", audit)
    return {
        "status": "summarized",
        "effects": effects,
        "artifact_audit": audit,
    }


def main() -> int:
    import argparse

    from robotarm_magnetic_lab.runtime.task010_visual_dependence_config import (
        load_visual_dependence_config,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_visual_dependence_config(args.config)
    result = summarize_visual_dependence(args.run_dir, config)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
