"""Pure synchronous random policies frozen by the TASK-009C contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    ParameterizedForceMode,
)


CONFIG_SCHEMA = "robotarm_magnetic_lab.task009c_random_baseline_preexperiment"
CONFIG_VERSION = 1
COMPARISON_CONFIG_SCHEMA = "robotarm_magnetic_lab.task009c_random_baseline_20pose_comparison"
COMPARISON_CONFIG_VERSION = 1
FROZEN_VALIDATION_POSE_IDS = (
    "validation-0006", "validation-0011", "validation-0015", "validation-0017",
    "validation-0019", "validation-0035", "validation-0040", "validation-0042",
    "validation-0045", "validation-0046", "validation-0051", "validation-0058",
    "validation-0060", "validation-0063", "validation-0067", "validation-0068",
    "validation-0069", "validation-0092", "validation-0095", "validation-0097",
)
COMPARISON_SNAPSHOT_TIMES_S = tuple(range(0, 301, 30))
POLICY_IDS = tuple(f"R{index}" for index in range(1, 8))
ALL_MODE_VALUES = np.asarray([int(mode) for mode in ParameterizedForceMode], dtype=np.int64)
MOVE_BIASED_PROBABILITIES = np.asarray(
    [0.05, 0.30, 0.30, 0.125, 0.125, 0.10], dtype=np.float64
)


@dataclass(frozen=True)
class PolicyAction:
    """One policy output; policies never accept environment observations."""

    mode: ParameterizedForceMode
    alpha: float

    def as_pair(self) -> tuple[int, float]:
        return int(self.mode), float(self.alpha)


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_comparison_config(source: Path, record: dict[str, Any]) -> dict[str, Any]:
    if record.get("version") != COMPARISON_CONFIG_VERSION:
        raise ValueError("20-pose comparison configuration version mismatch")
    expected = str(record.get("config_sha256", ""))
    payload = {key: value for key, value in record.items() if key != "config_sha256"}
    if _hash_payload(payload) != expected:
        raise ValueError("20-pose comparison configuration hash mismatch")
    root = source.resolve().parents[2]
    base_path = root / str(record["base_config_path"])
    base = load_random_baseline_config(base_path)
    if base["config_sha256"] != record["base_config_sha256"]:
        raise ValueError("20-pose comparison base configuration hash mismatch")
    pose_ids = tuple(str(value) for value in record["validation_pose_ids"])
    if pose_ids != FROZEN_VALIDATION_POSE_IDS:
        raise ValueError("20-pose comparison must use the frozen validation pose order")
    policy_ids = tuple(str(value).upper() for value in record["policy_ids"])
    if policy_ids != POLICY_IDS:
        raise ValueError("20-pose comparison must contain R1 through R7 exactly once")
    snapshot_times = tuple(int(value) for value in record["snapshot_times_s"])
    if snapshot_times != COMPARISON_SNAPSHOT_TIMES_S:
        raise ValueError("comparison snapshots must be taken every 30 seconds from 0 to 300")
    duration_s = float(record["episode_duration_s"])
    if duration_s != 300.0:
        raise ValueError("20-pose comparison episodes must remain 300 seconds")
    expanded = dict(base)
    expanded.update(record)
    expanded["environment_seeds"] = {
        pose_id: 950000 + int(pose_id.rsplit("-", 1)[1]) for pose_id in pose_ids
    }
    expanded["candidate_times_s"] = list(snapshot_times)
    expanded["smoke_episodes"] = []
    expanded["formal_episodes"] = []
    for pose_id in pose_ids:
        suffix = int(pose_id.rsplit("-", 1)[1])
        for policy_id in policy_ids:
            expanded["formal_episodes"].append(
                {
                    "episode_id": f"comparison-{pose_id}-{policy_id.lower()}",
                    "kind": "formal",
                    "policy_id": policy_id,
                    "pose_id": pose_id,
                    "environment_seed": 950000 + suffix,
                    "policy_seed": 960000 + 1000 * int(policy_id[1:]) + suffix,
                    "duration_s": duration_s,
                    "action_cycles": 3000,
                }
            )
    if len(expanded["formal_episodes"]) != 140:
        raise AssertionError("20-pose comparison expansion did not produce 140 episodes")
    return expanded


def load_random_baseline_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the single versioned TASK-009C configuration."""
    source = Path(path)
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("schema") == COMPARISON_CONFIG_SCHEMA:
        return _load_comparison_config(source, record)
    if record.get("schema") != CONFIG_SCHEMA or record.get("version") != CONFIG_VERSION:
        raise ValueError("TASK-009C configuration schema/version mismatch")
    expected = str(record.get("config_sha256", ""))
    payload = {key: value for key, value in record.items() if key != "config_sha256"}
    if _hash_payload(payload) != expected:
        raise ValueError("TASK-009C configuration hash mismatch")
    if record["clocks"] != {
        "control_hz": 10,
        "physics_hz": 240,
        "physics_steps_per_action": 24,
    }:
        raise ValueError("TASK-009C clocks must remain 240/10 Hz with 24 substeps")
    if len(record["formal_episodes"]) != 37 or len(record["smoke_episodes"]) != 8:
        raise ValueError("TASK-009C must contain 37 formal and 8 smoke episodes")
    return record


class _Policy:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng: np.random.Generator
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _action(mode: ParameterizedForceMode, alpha: float) -> PolicyAction:
        return PolicyAction(mode, 0.0 if mode == ParameterizedForceMode.HOLD else float(alpha))


class IndependentPolicy(_Policy):
    def act(self) -> PolicyAction:
        mode = ParameterizedForceMode(int(self.rng.choice(ALL_MODE_VALUES)))
        return self._action(mode, self.rng.random())


class BlockPolicy(_Policy):
    def __init__(self, seed: int, block_cycles: int) -> None:
        self.block_cycles = int(block_cycles)
        if self.block_cycles <= 0:
            raise ValueError("block_cycles must be positive")
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()
        self.remaining = 0
        self.current = PolicyAction(ParameterizedForceMode.HOLD, 0.0)

    def act(self) -> PolicyAction:
        if self.remaining == 0:
            mode = ParameterizedForceMode(int(self.rng.choice(ALL_MODE_VALUES)))
            self.current = self._action(mode, self.rng.random())
            self.remaining = self.block_cycles
        self.remaining -= 1
        return self.current


class MarkovPolicy(_Policy):
    def __init__(self, seed: int, stay_probability: float, alpha_sigma: float) -> None:
        self.stay_probability = float(stay_probability)
        self.alpha_sigma = float(alpha_sigma)
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()
        self.current: PolicyAction | None = None

    def act(self) -> PolicyAction:
        if self.current is None:
            mode = ParameterizedForceMode(int(self.rng.choice(ALL_MODE_VALUES)))
            self.current = self._action(mode, self.rng.random())
            return self.current
        previous = self.current.mode
        if self.rng.random() < self.stay_probability:
            mode = previous
        else:
            alternatives = ALL_MODE_VALUES[ALL_MODE_VALUES != int(previous)]
            mode = ParameterizedForceMode(int(self.rng.choice(alternatives)))
        if mode == ParameterizedForceMode.HOLD:
            alpha = 0.0
        elif previous == ParameterizedForceMode.HOLD:
            alpha = float(self.rng.random())
        else:
            alpha = float(np.clip(self.current.alpha + self.rng.normal(0.0, self.alpha_sigma), 0.0, 1.0))
        self.current = self._action(mode, alpha)
        return self.current


class MoveBiasedPersistentPolicy(_Policy):
    def __init__(self, seed: int, hold_probability: float, fixed_alpha: float | None) -> None:
        self.hold_probability = float(hold_probability)
        self.fixed_alpha = None if fixed_alpha is None else float(fixed_alpha)
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()
        self.current: PolicyAction | None = None

    def _sample(self) -> PolicyAction:
        mode = ParameterizedForceMode(
            int(self.rng.choice(ALL_MODE_VALUES, p=MOVE_BIASED_PROBABILITIES))
        )
        alpha = self.fixed_alpha if self.fixed_alpha is not None else float(self.rng.random())
        return self._action(mode, alpha)

    def act(self) -> PolicyAction:
        if self.current is None or self.rng.random() >= self.hold_probability:
            self.current = self._sample()
        return self.current


class AlternatingPolicy(_Policy):
    def __init__(self, seed: int, minimum_cycles: int, maximum_cycles: int) -> None:
        self.minimum_cycles = int(minimum_cycles)
        self.maximum_cycles = int(maximum_cycles)
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()
        self.stage = "MOVE"
        self.remaining = 0
        self.current: PolicyAction | None = None

    def _start_stage(self) -> None:
        if self.stage == "MOVE":
            modes = (ParameterizedForceMode.MOVE_POS, ParameterizedForceMode.MOVE_NEG)
        else:
            modes = (
                ParameterizedForceMode.VIEW_POS,
                ParameterizedForceMode.VIEW_NEG,
                ParameterizedForceMode.UP,
            )
        mode = modes[int(self.rng.integers(0, len(modes)))]
        self.current = self._action(mode, self.rng.random())
        self.remaining = int(self.rng.integers(self.minimum_cycles, self.maximum_cycles + 1))

    def act(self) -> PolicyAction:
        if self.remaining == 0:
            if self.current is not None:
                self.stage = "OBSERVE" if self.stage == "MOVE" else "MOVE"
            self._start_stage()
        self.remaining -= 1
        assert self.current is not None
        return self.current


class HoldPolicy(_Policy):
    def act(self) -> PolicyAction:
        return PolicyAction(ParameterizedForceMode.HOLD, 0.0)


def build_policy(policy_id: str, seed: int, config: dict[str, Any]) -> _Policy:
    """Build one policy using only its RNG and internal action history."""
    policy = str(policy_id).upper()
    parameters = config["policies"].get(policy)
    if parameters is None and policy != "HOLD":
        raise ValueError(f"unknown random baseline policy {policy_id!r}")
    if policy == "R1":
        return IndependentPolicy(seed)
    if policy == "R2":
        return BlockPolicy(seed, parameters["block_cycles"])
    if policy == "R3":
        return BlockPolicy(seed, parameters["block_cycles"])
    if policy == "R4":
        return MarkovPolicy(seed, parameters["stay_probability"], parameters["alpha_sigma"])
    if policy == "R5":
        return MoveBiasedPersistentPolicy(seed, parameters["stay_probability"], None)
    if policy == "R6":
        return MoveBiasedPersistentPolicy(
            seed, parameters["stay_probability"], parameters["fixed_alpha"]
        )
    if policy == "R7":
        return AlternatingPolicy(
            seed, parameters["minimum_stage_cycles"], parameters["maximum_stage_cycles"]
        )
    if policy == "HOLD":
        return HoldPolicy(seed)
    raise ValueError(f"unknown random baseline policy {policy_id!r}")
