"""Static contract tests for the TASK-007 Isaac Lab action adapter."""

import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.virtual_magnet_action import (
    VirtualMagnetRequestAction,
)


@pytest.mark.parametrize("action_id", range(11))
def test_all_public_action_ids_are_accepted(action_id):
    assert VirtualMagnetRequestAction.validate_action_id(float(action_id)) == action_id


@pytest.mark.parametrize("value", [-1.0, 11.0, 1.25, float("nan"), float("inf")])
def test_internal_or_invalid_action_ids_are_rejected(value):
    with pytest.raises(ValueError):
        VirtualMagnetRequestAction.validate_action_id(value)


def test_public_action_dimension_is_one_without_joint_names():
    source = VirtualMagnetRequestAction.__dict__
    assert source["action_dim"].fget(None) == 1
    assert "joint_names" not in " ".join(VirtualMagnetRequestAction.__dict__)

