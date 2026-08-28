from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.stomach_coverage.validate_task009d0_two_env_isolation import (
    validate_isolation_manifest,
)


def _manifest():
    return {
        "phase_a": [{
            "local_positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "local_quaternions": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        }],
        "phase_b_env1_replay": [{"position_error_m": 0.0, "quaternion_alignment": 1.0}],
        "row_reset": {
            "coverage_mask_hash_before": "same",
            "coverage_mask_hash_after": "same",
            "local_pose_before": [0.0] * 7,
            "local_pose_after": [0.0] * 7,
            "frame_before": 20,
            "frame_after": 20,
            "episode_index_before": 2,
            "episode_index_after": 2,
            "previous_action_before": [0.0] * 7,
            "previous_action_after": [0.0] * 7,
            "reward_before": 0.0,
            "reward_after": 0.0,
        },
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda m: m["row_reset"].update(coverage_mask_hash_after="changed"), "coverage"),
        (lambda m: m["row_reset"]["local_pose_after"].__setitem__(0, 1.0), "pose"),
        (lambda m: m["row_reset"].update(frame_after=21), "frame"),
        (lambda m: m["row_reset"].update(episode_index_after=3), "episode"),
        (lambda m: m["row_reset"]["previous_action_after"].__setitem__(0, 1.0), "previous action"),
        (lambda m: m["row_reset"].update(reward_after=1.0), "reward"),
        (lambda m: m["phase_a"][0]["local_positions"][1].__setitem__(0, 2e-6), "trajectory"),
        (lambda m: m["phase_a"][0]["local_quaternions"][1].__setitem__(3, 0.99), "quaternion"),
    ),
)
def test_isolation_protocol_rejects_cross_row_changes(mutate, message):
    value = deepcopy(_manifest())
    mutate(value)
    with pytest.raises(ValueError, match=message):
        validate_isolation_manifest(value)


def test_isolation_protocol_accepts_exact_fixture():
    assert validate_isolation_manifest(_manifest())["status"] == "pass"
