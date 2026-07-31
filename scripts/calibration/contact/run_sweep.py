#!/usr/bin/env python3
"""Run repeatable stomach/capsule contact-parameter sweeps.

This orchestration script uses the system Python. Each trial launches the
existing Isaac Lab contact test in a fresh process with temporary USD override
layers, so PhysX always receives the requested material parameters at startup.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path("/mnt/isaac-linux/robotarm_magnetic_lab")
RUNNER = PROJECT / "run_isaaclab.sh"
CONTACT_TESTS = PROJECT / "scripts" / "contact_physics"
BASE_SCENE = PROJECT / "assets" / "robotarm_magnetic_stomach_training.usda"
BASE_STOMACH = PROJECT / "assets" / "stomach" / "stomach_environment_lab.usda"
DEFAULT_GRID = (
    {"name": "baseline", "static_friction": 0.20, "dynamic_friction": 0.15,
     "stiffness": 25000.0, "damping": 300.0},
    {"name": "low", "static_friction": 0.15, "dynamic_friction": 0.10,
     "stiffness": 12000.0, "damping": 180.0},
    {"name": "high_friction", "static_friction": 0.30, "dynamic_friction": 0.25,
     "stiffness": 25000.0, "damping": 300.0},
    {"name": "soft_damped", "static_friction": 0.20, "dynamic_friction": 0.15,
     "stiffness": 12000.0, "damping": 400.0},
    {"name": "stiff_damped", "static_friction": 0.20, "dynamic_friction": 0.15,
     "stiffness": 50000.0, "damping": 600.0},
)
SCENARIOS = {
    "drop": CONTACT_TESTS / "test_02_drop_heights.py",
    "incline_slide": CONTACT_TESTS / "test_03_incline_slide.py",
}


def _scene_layer(path: Path, candidate: dict) -> None:
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "RobotarmMagneticStomachScene"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
)

def Xform "RobotarmMagneticStomachScene" (
    prepend references = @{BASE_SCENE}@</RobotarmMagneticStomachScene>
)
{{
    over "MagneticDemo"
    {{
        over "PhysicsMaterials"
        {{
            over "capsule_ground"
            {{
                float physics:staticFriction = {candidate["static_friction"]}
                float physics:dynamicFriction = {candidate["dynamic_friction"]}
                float physics:restitution = 0
                bool physxMaterial:compliantContactAccelerationSpring = true
                float physxMaterial:compliantContactStiffness = {candidate["stiffness"]}
                float physxMaterial:compliantContactDamping = {candidate["damping"]}
            }}
        }}
    }}
}}
""",
        encoding="utf-8",
    )


def _stomach_layer(path: Path, candidate: dict) -> None:
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "StomachEnvironment"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
)

def Xform "StomachEnvironment" (
    prepend references = @{BASE_STOMACH}@</StomachEnvironment>
)
{{
    over "ConvertedSource"
    {{
        over "Physics_Materials"
        {{
            over "Stomach_Wall_Material"
            {{
                float physics:staticFriction = {candidate["static_friction"]}
                float physics:dynamicFriction = {candidate["dynamic_friction"]}
                float physics:restitution = 0
            }}
        }}
    }}
}}
""",
        encoding="utf-8",
    )


def _latest_summary(result_root: Path, scenario: str) -> Path:
    summaries = sorted((result_root / scenario).glob("*/summary.json"))
    if not summaries:
        raise RuntimeError(f"No summary produced for {scenario} in {result_root}")
    return summaries[-1]


def _run_trial(run_dir: Path, candidate: dict, scenario: str) -> dict:
    candidate_dir = run_dir / candidate["name"]
    overlay_dir = candidate_dir / "overlays"
    result_root = candidate_dir / "results"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    scene_layer = overlay_dir / "scene.usda"
    stomach_layer = overlay_dir / "stomach.usda"
    _scene_layer(scene_layer, candidate)
    _stomach_layer(stomach_layer, candidate)

    env = os.environ.copy()
    env["ROBOTARM_STOMACH_SCENE_USD"] = str(scene_layer)
    env["ROBOTARM_STOMACH_ASSET_USD"] = str(stomach_layer)
    env["ROBOTARM_CONTACT_RESULT_ROOT"] = str(result_root)
    command = [
        str(RUNNER), "-p", str(SCENARIOS[scenario]),
        "--visualizer", "none", "--no-realtime", "--log_every", "100",
    ]
    log_path = candidate_dir / f"{scenario}.log"
    print(
        f"[CALIBRATION] candidate={candidate['name']} scenario={scenario} "
        f"mu=({candidate['static_friction']},{candidate['dynamic_friction']}) "
        f"k={candidate['stiffness']} c={candidate['damping']}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=PROJECT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(
            f"{scenario} failed with code {completed.returncode}\n{tail}"
        )
    summary_path = _latest_summary(result_root, scenario)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "scenario": scenario,
        "summary_path": str(summary_path),
        "passed": summary["passed"],
        "metrics": summary["metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=("drop", "incline_slide", "both"), default="both"
    )
    parser.add_argument(
        "--grid",
        type=Path,
        help="Optional JSON array replacing the built-in five-point grid.",
    )
    parser.add_argument("--max_candidates", type=int)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT / "logs" / "contact_calibration",
    )
    args = parser.parse_args()

    candidates = (
        json.loads(args.grid.read_text(encoding="utf-8"))
        if args.grid
        else list(DEFAULT_GRID)
    )
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    scenarios = tuple(SCENARIOS) if args.scenario == "both" else (args.scenario,)
    run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "grid.json").write_text(
        json.dumps(candidates, indent=2), encoding="utf-8"
    )

    index = {"schema_version": "1.0.0", "run_dir": str(run_dir), "candidates": []}
    for candidate in candidates:
        item = {"parameters": candidate, "trials": []}
        try:
            for scenario in scenarios:
                item["trials"].append(_run_trial(run_dir, candidate, scenario))
            item["completed"] = True
        except Exception as exc:
            item["completed"] = False
            item["error"] = str(exc)
            print(f"[CALIBRATION_ERROR] {exc}", file=sys.stderr, flush=True)
        index["candidates"].append(item)
        (run_dir / "sweep_index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )
    successful = sum(item["completed"] for item in index["candidates"])
    print(
        f"[CALIBRATION_DONE] successful={successful}/{len(candidates)} "
        f"index={run_dir / 'sweep_index.json'}",
        flush=True,
    )
    return 0 if successful == len(candidates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
