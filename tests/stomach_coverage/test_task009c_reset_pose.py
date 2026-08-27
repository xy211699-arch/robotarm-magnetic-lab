from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.task009b_training_env import (
    RESET_HOLD_CYCLES,
    TASK009C_CONFIG_PATH,
    TASK009C_OPTION_KEY,
    _load_task009c_pose_records,
    _validate_task009c_pose_record,
)


ROOT = Path(__file__).resolve().parents[2]


def _frozen():
    return _load_task009c_pose_records(TASK009C_CONFIG_PATH)


def test_frozen_pose_input_is_available_and_exactly_five_validation_records():
    config, manifest, allowed = _frozen()
    assert RESET_HOLD_CYCLES == 10
    assert set(allowed) == set(config["validation_pose_ids"])
    assert len(allowed) == 5
    assert all(record["split"] == "validation" for record in allowed.values())
    assert manifest["config_sha256"] == config["pose_library"]["manifest_config_sha256"]


def test_valid_pose_record_is_accepted_without_coordinate_changes():
    config, manifest, allowed = _frozen()
    frozen = allowed["validation-0006"]
    request = {
        "pose_id": frozen["pose_id"],
        "split": frozen["split"],
        "pose_world_xyzw": frozen["pose_world_xyzw"],
        "pose_library_manifest_config_sha256": manifest["config_sha256"],
    }
    pose_id, pose = _validate_task009c_pose_record(
        request, config=config, manifest=manifest, allowed=allowed
    )
    assert pose_id == "validation-0006"
    np.testing.assert_array_equal(pose, np.asarray(frozen["pose_world_xyzw"]))


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda record: record.pop("pose_id"), "missing fields"),
        (lambda record: record.update(split="train"), "not in the frozen validation set"),
        (lambda record: record.update(pose_id="validation-9999"), "not in the frozen validation set"),
        (
            lambda record: record.update(pose_library_manifest_config_sha256="bad"),
            "manifest hash mismatch",
        ),
        (lambda record: record.update(pose_world_xyzw=[0.0] * 6), "finite seven-vector"),
        (
            lambda record: record["pose_world_xyzw"].__setitem__(0, record["pose_world_xyzw"][0] + 1e-4),
            "differ from the frozen pose library",
        ),
    ),
)
def test_invalid_pose_records_are_rejected(mutation, match):
    config, manifest, allowed = _frozen()
    frozen = allowed["validation-0006"]
    request = {
        "pose_id": frozen["pose_id"],
        "split": frozen["split"],
        "pose_world_xyzw": list(frozen["pose_world_xyzw"]),
        "pose_library_manifest_config_sha256": manifest["config_sha256"],
    }
    mutation(request)
    with pytest.raises(ValueError, match=match):
        _validate_task009c_pose_record(
            request, config=config, manifest=manifest, allowed=allowed
        )


def test_reset_implementation_writes_pose_before_ten_hold_steps():
    source = (
        ROOT
        / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/"
        "robotarm_magnetic_lab/task009b_training_env.py"
    ).read_text(encoding="utf-8")
    base_reset = source.index("super().reset(seed=seed")
    pose_write = source.index("write_root_pose_to_sim_index")
    hold_loop = source.index("for cycle in range(RESET_HOLD_CYCLES)")
    assert base_reset < pose_write < hold_loop
    assert "permanent_wrench_composer.reset()" in source
    assert "write_root_velocity_to_sim_index" in source
    assert "self.episode_length_buf.zero_()" in source
    assert f'options.get(TASK009C_OPTION_KEY)' in source
    assert TASK009C_OPTION_KEY == "task009c_initial_pose"
