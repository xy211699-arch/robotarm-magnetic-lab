#!/usr/bin/env python3
"""Rank contact sweep candidates against measured physical targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _target_terms(targets: dict, trials: list[dict]) -> list[tuple[str, float, float, float]]:
    by_scenario = {trial["scenario"]: trial["metrics"] for trial in trials}
    terms = []
    drop_metrics = by_scenario.get("drop", {}).get("drops", [])
    drop_by_height = {str(int(item["height_mm"])): item for item in drop_metrics}
    for height, expected in targets.get("drop", {}).items():
        observed = drop_by_height.get(str(height))
        if observed is None:
            continue
        for metric, specification in expected.items():
            if specification is None or specification.get("value") is None:
                continue
            terms.append(
                (
                    f"drop.{height}.{metric}",
                    float(observed[metric]),
                    float(specification["value"]),
                    float(specification.get("scale", abs(specification["value"]) or 1.0)),
                )
            )
    slide_metrics = by_scenario.get("incline_slide", {}).get("regions", [])
    slide_by_region = {item["name"]: item for item in slide_metrics}
    for region, expected in targets.get("incline_slide", {}).items():
        observed = slide_by_region.get(region)
        if observed is None:
            continue
        for metric, specification in expected.items():
            if specification is None or specification.get("value") is None:
                continue
            terms.append(
                (
                    f"incline_slide.{region}.{metric}",
                    float(observed[metric]),
                    float(specification["value"]),
                    float(specification.get("scale", abs(specification["value"]) or 1.0)),
                )
            )
    return terms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_index", type=Path)
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "configs"
        / "calibration"
        / "contact_targets.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sweep = json.loads(args.sweep_index.read_text(encoding="utf-8"))
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    ranking = []
    for candidate in sweep["candidates"]:
        if not candidate.get("completed"):
            continue
        terms = _target_terms(targets["targets"], candidate["trials"])
        if not terms:
            raise SystemExit(
                "No numeric target is configured. Fill contact_targets.json "
                "with measured values before fitting."
            )
        squared = [((observed - expected) / scale) ** 2 for _, observed, expected, scale in terms]
        ranking.append(
            {
                "parameters": candidate["parameters"],
                "rmse": math.sqrt(sum(squared) / len(squared)),
                "term_count": len(terms),
                "residuals": [
                    {
                        "name": name,
                        "observed": observed,
                        "target": expected,
                        "normalized_error": (observed - expected) / scale,
                    }
                    for name, observed, expected, scale in terms
                ],
            }
        )
    ranking.sort(key=lambda item: item["rmse"])
    result = {
        "schema_version": "1.0.0",
        "source_sweep": str(args.sweep_index),
        "source_targets": str(args.targets),
        "ranking": ranking,
        "recommended": ranking[0] if ranking else None,
    }
    output = args.output or args.sweep_index.with_name("fit_result.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[CONTACT_FIT] result={output}")
    if ranking:
        print(json.dumps(ranking[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
