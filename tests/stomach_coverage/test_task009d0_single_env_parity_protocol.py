from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.stomach_coverage.validate_task009d0_single_env_parity import (
    validate_parity_records,
)


def _record():
    return {
        "pose_id": "validation-0006",
        "boundary": 1,
        "physics_substeps": 24,
        "scalar_frame_id": 10,
        "vector_frame_id": 10,
        "scalar_camera_force": [1.0, 0.0, 0.0],
        "vector_camera_force": [1.0, 0.0, 0.0],
        "scalar_other_force": [1.0, 0.0, 0.0],
        "vector_other_force": [1.0, 0.0, 0.0],
        "scalar_current_mask": [True, False, True],
        "vector_current_mask": [True, False, True],
        "scalar_cumulative_mask": [True, False, True],
        "vector_cumulative_mask": [True, False, True],
        "area_error_m2": 0.0,
        "finite": True,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda row: row["vector_current_mask"].__setitem__(1, True), "current mask"),
        (lambda row: row.update(vector_frame_id=11), "frame"),
        (lambda row: row["vector_camera_force"].__setitem__(0, 1.1), "force"),
        (lambda row: row.update(physics_substeps=23), "24 substeps"),
    ),
)
def test_protocol_rejects_exact_parity_mutations(mutate, message):
    row = deepcopy(_record())
    mutate(row)
    with pytest.raises(ValueError, match=message):
        validate_parity_records([row], expected_count=1)


def test_protocol_accepts_exact_record():
    summary = validate_parity_records([_record()], expected_count=1)
    assert summary["status"] == "pass"
    assert summary["boundary_count"] == 1
