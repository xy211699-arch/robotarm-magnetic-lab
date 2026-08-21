"""Deterministic TASK-008 calibration manifests and metric evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force_macro import (
    DynamicForceMacroActionId,
    move_projected_displacement_m,
    up_elevation_and_crossing,
    view_signed_angle_deg,
)


@dataclass(frozen=True)
class TrialSpec:
    split: str
    action_id: int
    trial_index: int
    seed: int
    x_offset_m: float
    y_offset_m: float
    yaw_rad: float
    roll_rad: float


def make_manifest(split: str, samples: int, seed: int, actions=(1, 2, 3, 4, 5)) -> list[TrialSpec]:
    rows = []
    for action in actions:
        for index in range(samples):
            value = int(hashlib.sha256(f"{split}:{seed}:{action}:{index}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(value)
            rows.append(TrialSpec(split, action, index, value, rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)))
    return rows


def replacement_trial(spec: TrialSpec, attempt: int) -> TrialSpec:
    """Deterministically replace an invalid reset without consuming its slot."""
    if attempt <= 0:
        raise ValueError("replacement attempt must be positive")
    value = int(
        hashlib.sha256(
            f"{spec.split}:{spec.seed}:{spec.action_id}:{spec.trial_index}:replacement:{attempt}".encode()
        ).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(value)
    return TrialSpec(
        spec.split,
        spec.action_id,
        spec.trial_index,
        value,
        rng.uniform(-0.01, 0.01),
        rng.uniform(-0.01, 0.01),
        rng.uniform(-np.pi, np.pi),
        rng.uniform(-np.pi, np.pi),
    )


def manifest_sha256(rows: list[TrialSpec]) -> str:
    return hashlib.sha256(json.dumps([asdict(row) for row in rows], sort_keys=True).encode()).hexdigest()


def coarse_candidates(initial=0.9, growth=1.25, maximum=3.0) -> list[float]:
    values = [float(initial)]
    while values[-1] * growth < maximum:
        values.append(values[-1] * growth)
    if values[-1] != maximum:
        values.append(float(maximum))
    return values


def midpoint_refinements(lower_failure: float, first_pass: float, rounds=3) -> list[float]:
    lower, upper = float(lower_failure), float(first_pass)
    result = []
    for _ in range(rounds):
        middle = 0.5 * (lower + upper)
        result.append(middle)
        upper = middle
    return result


def evaluate_trace(action_id: int, trace) -> tuple[bool, dict]:
    action = DynamicForceMacroActionId(action_id)
    if len(trace) != 240:
        return False, {"reason": "wrong_substep_count"}
    onset = trace[48] if action != DynamicForceMacroActionId.UP else trace[0]
    end = trace[-1]
    if action in (DynamicForceMacroActionId.MOVE_POS, DynamicForceMacroActionId.MOVE_NEG):
        value = move_projected_displacement_m(onset.com_world, end.com_world, onset.lateral_direction_world)
        return value >= 0.005, {"projected_displacement_m": value}
    if action in (DynamicForceMacroActionId.VIEW_POS, DynamicForceMacroActionId.VIEW_NEG):
        direction = np.asarray(onset.lateral_direction_world)
        value = view_signed_angle_deg(onset.camera_axis_world, end.camera_axis_world, direction)
        return value >= 15.0, {"signed_view_angle_deg": value}
    elevation, crossed = up_elevation_and_crossing(
        end.camera_axis_world,
        np.asarray(onset.camera_axis_world) - np.array([0.0, 0.0, onset.camera_axis_world[2]]),
        [item.camera_axis_world for item in trace],
    )
    return elevation >= 45.0 and not crossed, {"camera_elevation_deg": elevation, "crossed_vertical": crossed}


def write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
