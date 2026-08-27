from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robotarm_magnetic_lab.baselines.random_policies import (
    POLICY_IDS,
    PolicyAction,
    build_policy,
    load_random_baseline_config,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    ParameterizedForceMode,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task009c/random_baseline_preexperiment_v1.json"


def _config():
    return load_random_baseline_config(CONFIG_PATH)


def _pairs(policy, count):
    return [policy.act().as_pair() for _ in range(count)]


def test_versioned_config_expands_every_episode_and_seed():
    config = _config()
    assert len(config["formal_episodes"]) == 37
    assert len(config["smoke_episodes"]) == 8
    assert config["validation_pose_ids"] == [
        "validation-0006",
        "validation-0011",
        "validation-0015",
        "validation-0017",
        "validation-0019",
    ]
    assert config["scheduling"]["policy_order_per_pose"] == [
        "R2", "R5", "R6", "R7", "R1", "R4", "R3"
    ]
    random_episodes = [e for e in config["formal_episodes"] if e["policy_id"] != "HOLD"]
    assert len(random_episodes) == 35
    assert len({e["episode_id"] for e in config["formal_episodes"]}) == 37
    for episode in random_episodes:
        suffix = int(episode["pose_id"].split("-")[1])
        index = int(episode["policy_id"][1:])
        assert episode["policy_seed"] == 960000 + 1000 * index + suffix
    assert [(e["pose_id"], e["policy_seed"]) for e in config["formal_episodes"][-2:]] == [
        ("validation-0006", 960006),
        ("validation-0019", 960019),
    ]


def test_config_hash_is_mandatory(tmp_path):
    record = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    record["formal_episodes"][0]["policy_seed"] += 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_random_baseline_config(path)


@pytest.mark.parametrize("policy_id", (*POLICY_IDS, "HOLD"))
def test_policy_seed_reproducibility_and_reset(policy_id):
    first = build_policy(policy_id, 961006, _config())
    second = build_policy(policy_id, 961006, _config())
    expected = _pairs(first, 200)
    assert _pairs(second, 200) == expected
    first.reset()
    assert _pairs(first, 200) == expected


def test_r1_mode_distribution_and_hold_alpha():
    policy = build_policy("R1", 961006, _config())
    actions = [policy.act() for _ in range(120_000)]
    counts = np.bincount([int(action.mode) for action in actions], minlength=6) / len(actions)
    assert np.max(np.abs(counts - 1.0 / 6.0)) < 0.006
    assert all(action.alpha == 0.0 for action in actions if action.mode == ParameterizedForceMode.HOLD)


@pytest.mark.parametrize(("policy_id", "block"), (("R2", 5), ("R3", 10)))
def test_block_policies_hold_mode_and_alpha_for_exact_block(policy_id, block):
    actions = _pairs(build_policy(policy_id, 962006, _config()), block * 30)
    for offset in range(0, len(actions), block):
        assert len(set(actions[offset : offset + block])) == 1


class _FakeRng:
    def __init__(self, *, random_values=(), choice_values=(), normal_values=(), integer_values=()):
        self.random_values = iter(random_values)
        self.choice_values = iter(choice_values)
        self.normal_values = iter(normal_values)
        self.integer_values = iter(integer_values)

    def random(self):
        return float(next(self.random_values))

    def choice(self, values, p=None):
        requested = int(next(self.choice_values))
        assert requested in [int(value) for value in values]
        return requested

    def normal(self, _mean, _sigma):
        return float(next(self.normal_values))

    def integers(self, low, high=None):
        value = int(next(self.integer_values))
        assert value >= low and (high is None or value < high)
        return value


def test_r4_switches_to_different_mode_and_walks_alpha():
    policy = build_policy("R4", 964006, _config())
    policy.current = PolicyAction(ParameterizedForceMode.MOVE_POS, 0.4)
    policy.rng = _FakeRng(random_values=[0.9], choice_values=[2], normal_values=[0.05])
    action = policy.act()
    assert action.mode == ParameterizedForceMode.MOVE_NEG
    assert action.alpha == pytest.approx(0.45)


def test_r4_hold_transition_resets_and_restarts_alpha():
    policy = build_policy("R4", 964006, _config())
    policy.current = PolicyAction(ParameterizedForceMode.MOVE_POS, 0.4)
    policy.rng = _FakeRng(random_values=[0.9], choice_values=[0])
    assert policy.act() == PolicyAction(ParameterizedForceMode.HOLD, 0.0)
    policy.rng = _FakeRng(random_values=[0.9, 0.7], choice_values=[3])
    assert policy.act() == PolicyAction(ParameterizedForceMode.VIEW_POS, 0.7)


def test_r5_resample_may_return_same_mode():
    policy = build_policy("R5", 965006, _config())
    policy.current = PolicyAction(ParameterizedForceMode.MOVE_POS, 0.25)
    policy.rng = _FakeRng(random_values=[0.95, 0.75], choice_values=[1])
    action = policy.act()
    assert action.mode == ParameterizedForceMode.MOVE_POS
    assert action.alpha == 0.75


def test_r5_long_run_mode_distribution():
    policy = build_policy("R5", 965006, _config())
    actions = [policy.act() for _ in range(200_000)]
    counts = np.bincount([int(action.mode) for action in actions], minlength=6) / len(actions)
    expected = np.asarray([0.05, 0.30, 0.30, 0.125, 0.125, 0.10])
    assert np.max(np.abs(counts - expected)) < 0.012


def test_r6_non_hold_alpha_is_fixed_midpoint():
    actions = [build_policy("R6", 966006, _config()).act() for _ in range(500)]
    assert all(action.alpha in (0.0, 0.5) for action in actions)


def test_r7_starts_move_then_strictly_alternates_stage_classes():
    policy = build_policy("R7", 967006, _config())
    policy.rng = _FakeRng(
        random_values=[0.2, 0.8, 0.4],
        integer_values=[0, 5, 2, 6, 1, 7],
    )
    actions = [policy.act() for _ in range(18)]
    assert all(action.mode in (ParameterizedForceMode.MOVE_POS, ParameterizedForceMode.MOVE_NEG) for action in actions[:5])
    assert all(action.mode in (ParameterizedForceMode.VIEW_POS, ParameterizedForceMode.VIEW_NEG, ParameterizedForceMode.UP) for action in actions[5:11])
    assert all(action.mode in (ParameterizedForceMode.MOVE_POS, ParameterizedForceMode.MOVE_NEG) for action in actions[11:18])
    assert all(action.mode != ParameterizedForceMode.HOLD for action in actions)


def test_hold_policy_always_outputs_zero_actor_command():
    assert set(_pairs(build_policy("HOLD", 960006, _config()), 100)) == {(0, 0.0)}
