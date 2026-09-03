#!/usr/bin/env python3
"""Machine-readable V0-V2 gate evidence validator for the visual-dependence study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _required_true(evidence: Mapping[str, Any], gate: str, field: str) -> None:
    if evidence.get(field) is not True:
        raise RuntimeError(f"{gate}.{field} is required")


def validate_gate_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise RuntimeError("gate evidence must be an object")
    v0 = evidence.get("v0")
    v1 = evidence.get("v1")
    v2 = evidence.get("v2")
    v3 = evidence.get("v3")
    if not isinstance(v0, dict):
        raise RuntimeError("v0 evidence is missing")
    for field in (
        "critic_isolation",
        "blind_visual_projection",
        "identical_trainable_parameters",
        "actor_observation_schema",
        "resnet_forward_count",
    ):
        _required_true(v0, "v0", field)
    if not isinstance(v1, dict):
        raise RuntimeError("v1 evidence is missing")
    _required_true(v1, "v1", "blind_forward_backward_save_restore")
    if not isinstance(v2, dict):
        raise RuntimeError("v2 evidence is missing")
    for field in (
        "donor_mapping",
        "first_frame_repeat",
        "target_previous_action",
        "unique_variables",
        "curve_length",
    ):
        _required_true(v2, "v2", field)
    if not isinstance(v3, dict):
        raise RuntimeError("v3 evidence is missing")
    if v3.get("status") != "awaiting_manual_start":
        raise RuntimeError("v3 status must remain awaiting_manual_start")
    return {
        "schema": "robotarm_magnetic_lab.task010_visual_dependence_gate_report",
        "status": "passed",
        "v0": v0,
        "v1": v1,
        "v2": v2,
        "v3": v3,
    }


def _load_evidence(args: argparse.Namespace) -> dict[str, Any]:
    path = args.evidence
    if path is None and args.run_dir is not None:
        path = Path(args.run_dir) / "gates" / "evidence.json"
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("gate evidence file is missing")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("gate evidence must be a JSON object")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    report = validate_gate_evidence(_load_evidence(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
