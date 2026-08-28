from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "stomach_coverage"
sys.path.insert(0, str(SCRIPTS))

from validate_task009d0_long_soak import SCHEMA, validate_soak  # noqa: E402


def _valid_manifest() -> dict:
    episodes = []
    for episode_index in range(2):
        episodes.append({
            "episode_index": episode_index,
            "formal_steps": 1200,
            "formal_physics_substeps": 28800,
            "inter_episode_hold_substeps": 240,
            "post_reset_episode_length": [0, 0],
            "terminal_observation_present": True,
            "envs": [
                {
                    "env_id": env_id,
                    "coverage_points": 1201,
                    "reachable_coverage": [0.1 + episode_index * 0.01] * 1201,
                    "raw_coverage": [0.2 + episode_index * 0.01] * 1201,
                    "frame_ids": list(range(episode_index * 2000, episode_index * 2000 + 1201)),
                    "initial_mask_sha256": f"initial-{episode_index}-{env_id}",
                    "final_mask_sha256": f"final-{episode_index}-{env_id}",
                    "state_finite": True,
                    "rgb_finite": True,
                }
                for env_id in range(2)
            ],
        })
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": "pass",
        "selected_num_envs": 2,
        "clocks": {
            "physics_hz": 240,
            "control_hz": 10,
            "physics_steps_per_action": 24,
        },
        "episodes": episodes,
        "faults": [],
    }


def test_two_episode_manifest_requires_exact_counts():
    assert validate_soak(_valid_manifest())["status"] == "pass"
    broken = copy.deepcopy(_valid_manifest())
    broken["episodes"][1]["envs"][0]["coverage_points"] = 1200
    with pytest.raises(ValueError, match="1201 coverage points"):
        validate_soak(broken)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("formal_physics_substeps", 28776, "28800"),
        ("inter_episode_hold_substeps", 216, "240 HOLD"),
        ("post_reset_episode_length", [1, 0], "zero after reset"),
        ("terminal_observation_present", False, "terminal observation"),
    ),
)
def test_episode_contract_rejects_wrong_boundary_state(field, value, message):
    broken = copy.deepcopy(_valid_manifest())
    broken["episodes"][0][field] = value
    with pytest.raises(ValueError, match=message):
        validate_soak(broken)


def test_manifest_rejects_inherited_mask_nonmonotonic_coverage_and_repeated_rgb():
    inherited = copy.deepcopy(_valid_manifest())
    inherited["episodes"][1]["envs"][0]["initial_mask_sha256"] = inherited["episodes"][0]["envs"][0]["final_mask_sha256"]
    with pytest.raises(ValueError, match="inherited"):
        validate_soak(inherited)

    decreasing = copy.deepcopy(_valid_manifest())
    decreasing["episodes"][0]["envs"][0]["reachable_coverage"][600] = 0.09
    with pytest.raises(ValueError, match="decreased"):
        validate_soak(decreasing)

    repeated = copy.deepcopy(_valid_manifest())
    repeated["episodes"][0]["envs"][0]["frame_ids"][600] = repeated["episodes"][0]["envs"][0]["frame_ids"][599]
    with pytest.raises(ValueError, match="RGB frames"):
        validate_soak(repeated)
