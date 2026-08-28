from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    ParameterizedForceConfig,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.vectorized_parameterized_force_action import (
    VectorizedParameterizedForceAction,
)


class _Proxy:
    def __init__(self, tensor):
        self.torch = tensor


class _Composer:
    def __init__(self):
        self.reset_calls = []
        self.set_calls = []

    def reset(self, env_ids=None):
        self.reset_calls.append(None if env_ids is None else env_ids.clone())

    def set_forces_and_torques_index(self, **kwargs):
        self.set_calls.append(kwargs)


def _fake_action(num_envs=4):
    action = object.__new__(VectorizedParameterizedForceAction)
    action._env = SimpleNamespace(num_envs=num_envs, device="cpu")
    action.config = ParameterizedForceConfig()
    action.camera_center_local = torch.tensor([0.0, 0.0, 0.006])
    action.other_center_local = torch.tensor([0.0, 0.0, -0.006])
    action.mass_kg = torch.full((num_envs,), 0.005735)
    action._raw_actions = torch.zeros((num_envs, 2))
    action._processed_actions = torch.zeros((num_envs, 2))
    action._modes = torch.zeros(num_envs, dtype=torch.int64)
    action._alpha = torch.zeros(num_envs)
    action._previous_action_features = torch.zeros((num_envs, 7))
    action._all_env_ids = torch.arange(num_envs, dtype=torch.int64)
    action.last_camera_positions_world = torch.zeros((num_envs, 3))
    action.last_other_positions_world = torch.zeros((num_envs, 3))
    action.last_directions_world = torch.zeros((num_envs, 3))
    action.last_resultant_forces_world = torch.zeros((num_envs, 3))
    action.last_resultant_torques_world = torch.zeros((num_envs, 3))
    composer = _Composer()
    data = SimpleNamespace(
        root_link_pos_w=_Proxy(torch.zeros((num_envs, 3))),
        root_link_quat_w=_Proxy(
            # Isaac Lab uses xyzw. Rotate the local capsule axis onto world +X
            # so MOVE/VIEW have a well-defined horizontal lateral direction.
            torch.tensor([[0.0, 2.0**-0.5, 0.0, 2.0**-0.5]]).repeat(num_envs, 1)
        ),
        root_com_pos_w=_Proxy(torch.zeros((num_envs, 3))),
    )
    action.capsule = SimpleNamespace(data=data, permanent_wrench_composer=composer)
    return action, composer


def test_previous_action_features_mask_hold_strength():
    action, _ = _fake_action(2)
    action.process_actions(torch.tensor([[0.0, 0.8], [5.0, 0.25]]))
    expected = torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0, 1, 0.25]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(action.previous_action_features.cpu(), expected)
    assert action.processed_actions[0, 1].item() == 0.0


def test_action_term_requires_exact_vector_shape_and_integral_modes():
    action, _ = _fake_action(2)
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        action.process_actions(torch.zeros((1, 2)))
    with pytest.raises(ValueError, match="integral"):
        action.process_actions(torch.tensor([[1.25, 0.5], [0.0, 0.0]]))


def test_mixed_batch_writes_only_intended_rows_and_up_uses_position():
    action, composer = _fake_action(4)
    action.process_actions(
        torch.tensor([[0.0, 0.0], [1.0, 0.5], [3.0, 0.5], [5.0, 0.5]])
    )
    action.apply_actions()
    assert len(composer.set_calls) == 2
    com_call, up_call = composer.set_calls
    assert com_call["env_ids"].tolist() == [1, 2]
    assert com_call["positions"] is None
    assert com_call["torques"] is not None
    assert tuple(com_call["forces"].shape) == (2, 1, 3)
    assert up_call["env_ids"].tolist() == [3]
    assert up_call["positions"] is not None
    assert up_call["torques"] is None
    assert up_call["forces"][0, 0, 2].item() > 0.0


def test_mode_change_to_hold_clears_selected_composer_rows_before_substep():
    action, composer = _fake_action(2)
    action.process_actions(torch.tensor([[1.0, 0.5], [3.0, 0.5]]))
    action.apply_actions()
    previous_set_count = len(composer.set_calls)
    action.process_actions(torch.tensor([[0.0, 1.0], [0.0, 0.2]]))
    assert len(composer.reset_calls) >= 2
    assert composer.reset_calls[-1].tolist() == [0, 1]
    action.apply_actions()
    assert len(composer.set_calls) == previous_set_count


def test_reset_clears_only_selected_tensor_rows():
    action, composer = _fake_action(3)
    action.process_actions(torch.tensor([[1.0, 1.0], [3.0, 1.0], [5.0, 1.0]]))
    before = action.previous_action_features[1].clone()
    action.reset(torch.tensor([0, 2]))
    assert not action.previous_action_features[[0, 2]].any()
    torch.testing.assert_close(action.previous_action_features[1], before)
    assert composer.reset_calls[-1].tolist() == [0, 2]
