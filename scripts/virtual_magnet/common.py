"""Deterministic protocols and result utilities for TASK-007 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ACTION_NAMES = (
    "HOLD_VIEW",
    "VIEW_UP",
    "VIEW_UP_RIGHT",
    "VIEW_RIGHT",
    "VIEW_DOWN_RIGHT",
    "VIEW_DOWN",
    "VIEW_DOWN_LEFT",
    "VIEW_LEFT",
    "VIEW_UP_LEFT",
    "MOVE_SIDE_POS",
    "MOVE_SIDE_NEG",
)
KEY_TO_ACTION = {
    "KEY_0": 0,
    "KEY_1": 1,
    "KEY_2": 2,
    "KEY_3": 3,
    "KEY_4": 4,
    "KEY_5": 5,
    "KEY_6": 6,
    "KEY_7": 7,
    "KEY_8": 8,
    "KEY_9": 9,
    "KEY_MINUS": 10,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "-": 10,
}


def key_name_to_action(key: str) -> int | None:
    normalized = str(key).upper().strip().replace("KEY_", "")
    aliases = {"MINUS": "-", "SUBTRACT": "-", "NUMPAD_SUBTRACT": "-"}
    normalized = aliases.get(normalized, normalized)
    if normalized.startswith("NUMPAD") and normalized[-1:].isdigit():
        normalized = normalized[-1]
    return KEY_TO_ACTION.get(normalized)


@dataclass(frozen=True)
class TrialSpec:
    split: str
    action_id: int
    trial_index: int
    seed: int
    valid_move: bool = True
    region_id: str = "flat_center"


def generate_manifest(
    split: str,
    trials_per_action: int,
    *,
    base_seed: int,
    action_ids: Iterable[int] = range(11),
    valid_move: bool = True,
    region_ids: tuple[str, ...] = ("flat_center",),
) -> list[TrialSpec]:
    if trials_per_action <= 0:
        raise ValueError("trials_per_action must be positive")
    # Separate split namespaces make development and held-out seed sets
    # disjoint even when callers use the same base seed.
    split_offset = int.from_bytes(hashlib.sha256(split.encode()).digest()[:4], "little")
    result = []
    for action_id in action_ids:
        if not 0 <= int(action_id) <= 10:
            raise ValueError(f"invalid action ID: {action_id}")
        for trial_index in range(trials_per_action):
            seed = int(base_seed + split_offset + 1009 * int(action_id) + trial_index)
            result.append(
                TrialSpec(
                    split=split,
                    action_id=int(action_id),
                    trial_index=trial_index,
                    seed=seed,
                    valid_move=valid_move,
                    region_id=region_ids[trial_index % len(region_ids)],
                )
            )
    return result


def manifest_digest(specs: Iterable[TrialSpec]) -> str:
    payload = json.dumps([asdict(item) for item in specs], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def terminal_pass(action_id: int, audit: dict, *, invalid_move: bool = False) -> bool:
    action_substeps = audit.get("action_substeps", audit.get("physics_substeps"))
    if action_substeps != 240 or audit.get("result") == "fault":
        return False
    stable = (
        float(audit.get("linear_speed_m_s", float("inf"))) <= 0.002
        and float(audit.get("angular_speed_rad_s", float("inf"))) <= 0.1
    )
    if action_id in (9, 10):
        if invalid_move:
            return audit.get("result") == "rejected" and stable
        displacement = float(audit.get("move_signed_displacement_m", 0.0))
        return audit.get("result") == "completed" and 0.004 <= displacement <= 0.006 and stable
    return (
        audit.get("result") == "completed"
        and float(audit.get("optical_axis_error_deg", float("inf"))) <= 3.0
        and float(audit.get("tangent_drift_m", float("inf"))) <= 0.002
        and stable
    )


def summarize_trials(rows: list[dict]) -> dict:
    classes = {}
    for row in rows:
        key = row.get("class", ACTION_NAMES[int(row["action_id"])])
        item = classes.setdefault(key, {"total": 0, "passed": 0, "fault": 0, "rejected": 0})
        item["total"] += 1
        item["passed"] += int(bool(row.get("pass")))
        item["fault"] += int(row.get("result") == "fault")
        item["rejected"] += int(row.get("result") == "rejected")
    for item in classes.values():
        item["rate"] = item["passed"] / max(item["total"], 1)
        item["gate_16_of_20"] = item["total"] >= 20 and item["passed"] >= 16
    return {
        "classes": classes,
        "all_classes_pass": bool(classes) and all(item["gate_16_of_20"] for item in classes.values()),
    }


def write_json(path: str | Path, payload: dict | list) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def file_evidence(path: str | Path) -> dict:
    source = Path(path)
    return {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def apply_trial_reset(env, spec: TrialSpec) -> None:
    """Apply a deterministic reset-only pose perturbation; never used during actions."""
    import torch
    from scipy.spatial.transform import Rotation

    env.reset(seed=spec.seed)
    unwrapped = env.unwrapped
    capsule = unwrapped.scene["capsule"]
    pose = capsule.data.root_pose_w.torch.clone()
    rng = np.random.default_rng(spec.seed)
    pose[0, :3] += torch.as_tensor(
        [rng.uniform(-0.001, 0.001), rng.uniform(-0.001, 0.001), rng.uniform(0.0001, 0.0005)],
        device=unwrapped.device,
        dtype=torch.float32,
    )
    yaw = rng.uniform(-0.05, 0.05)
    base = Rotation.from_quat([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    quaternion = (Rotation.from_rotvec([0.0, 0.0, yaw]) * base).as_quat().astype(np.float32)
    if not spec.valid_move and spec.action_id in (9, 10):
        # Invalid MOVE keeps a geometrically valid side pose but starts in free
        # space, so the exact missing predicate is recent sidewall contact.
        # This avoids conflating rejection behavior with an unstable upright
        # magnetic equilibrium.
        pose[0, 2] += 0.025
    pose[0, 3:7] = torch.as_tensor(quaternion, device=unwrapped.device)
    capsule.write_root_pose_to_sim_index(root_pose=pose)
    capsule.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=unwrapped.device)
    )
    # The pose perturbation is written after env.reset(), while the bridge's
    # reset event has already sampled the pre-perturbation pose. Re-synchronize
    # the virtual magnet so every trial starts at the same capsule-relative
    # nominal pose rather than a stale world pose.
    unwrapped._virtual_magnet_bridge.reset(
        env_ids=torch.tensor([0], device=unwrapped.device, dtype=torch.long)
    )


def run_live_trial(env, spec: TrialSpec) -> dict:
    """Run a settled one-second trial and return privileged terminal audit."""
    import torch

    apply_trial_reset(env, spec)
    unwrapped = env.unwrapped
    if spec.valid_move or spec.action_id <= 8:
        # Establish deterministic contact history without issuing a public
        # HOLD request or moving the controller away from its nominal branch.
        env.step(torch.tensor([[-1.0]], device=unwrapped.device))
    env.step(torch.tensor([[float(spec.action_id)]], device=unwrapped.device))
    audit = {}
    for key, value in unwrapped._virtual_magnet_bridge.audit.items():
        if isinstance(value, np.ndarray):
            audit[key] = value.tolist()
        elif isinstance(value, np.generic):
            audit[key] = value.item()
        else:
            audit[key] = value
    audit.update(
        seed=spec.seed,
        split=spec.split,
        action_id=spec.action_id,
        action_name=ACTION_NAMES[spec.action_id],
        trial_index=spec.trial_index,
        region_id=spec.region_id,
        valid_move=spec.valid_move,
    )
    audit["pass"] = terminal_pass(spec.action_id, audit, invalid_move=not spec.valid_move)
    audit["class"] = (
        f"INVALID_{ACTION_NAMES[spec.action_id]}" if not spec.valid_move else ACTION_NAMES[spec.action_id]
    )
    return audit
