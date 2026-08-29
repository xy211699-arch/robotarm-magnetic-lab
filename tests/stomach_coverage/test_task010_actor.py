from __future__ import annotations

import torch
import pytest

from robotarm_magnetic_lab.learning.task010_actor import Task010Actor


def test_actor_forward_shapes_and_deterministic_action():
    actor = Task010Actor()
    obs = torch.zeros((12, 519))
    action = actor(obs, stochastic_output=False)
    assert action.shape == (12, 2)
    params = actor.output_distribution_params()
    assert params["logits"].shape == (12, 6)
    assert params["concentration_raw"].shape == (12, 5, 2)
    assert actor.get_output_log_prob(action).shape == (12, 1)
    assert actor.output_entropy().shape == (12, 1)


def test_actor_carries_and_resets_selected_hidden_rows():
    actor = Task010Actor()
    obs = torch.randn((12, 519))
    actor(obs, stochastic_output=True)
    carried = actor.get_hidden_state().clone()
    actor.reset(torch.tensor([True] + [False] * 11))
    reset_state = actor.get_hidden_state()
    assert torch.equal(reset_state[:, 0], torch.zeros_like(reset_state[:, 0]))
    assert torch.equal(reset_state[:, 1:], carried[:, 1:])


def test_rollout_detach_preserves_values_and_reset_is_not_rollout_boundary():
    actor = Task010Actor()
    actor(torch.randn((4, 519)), stochastic_output=True)
    before = actor.get_hidden_state().clone()
    actor.detach_hidden_state()
    after = actor.get_hidden_state()
    assert torch.equal(before, after)
    assert after.grad_fn is None


def test_actor_rejects_privileged_width():
    with pytest.raises(ValueError, match="expected 519"):
        Task010Actor()(torch.zeros((12, 65)))


def test_sequence_masks_reset_only_requested_time_rows():
    actor = Task010Actor()
    obs = torch.randn((3, 2, 519))
    masks = torch.tensor([[False, False], [True, False], [False, False]])
    logits, raw, hidden = actor.evaluate_parameters(obs, masks=masks)
    assert logits.shape == (3, 2, 6)
    assert raw.shape == (3, 2, 5, 2)
    assert hidden.shape == (1, 2, 256)
