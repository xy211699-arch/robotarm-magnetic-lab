from __future__ import annotations

import torch
from torch.distributions import kl_divergence

from robotarm_magnetic_lab.learning.task010_distribution import Task010ModeBetaDistribution


def _distribution(forced_mode: int | None = None) -> Task010ModeBetaDistribution:
    logits = torch.zeros((8, 6))
    if forced_mode is not None:
        logits[:, forced_mode] = 100.0
    raw = torch.zeros((8, 5, 2))
    return Task010ModeBetaDistribution(logits, raw)


def test_hold_has_zero_strength_and_no_beta_log_prob():
    distribution = _distribution(0)
    action = distribution.mode()
    assert torch.equal(action[:, 1], torch.zeros_like(action[:, 1]))
    expected = torch.log_softmax(distribution.logits, -1)[:, :1]
    assert torch.allclose(distribution.log_prob(action), expected)


def test_entropy_is_categorical_plus_probability_weighted_conditional_entropy():
    distribution = _distribution()
    expected = distribution.categorical.entropy() + (
        distribution.categorical.probs[:, 1:] * distribution.beta.entropy()
    ).sum(-1)
    assert torch.allclose(distribution.entropy().squeeze(-1), expected, atol=1e-6)


def test_sample_support_and_parameter_shapes():
    distribution = _distribution()
    action = distribution.sample()
    assert action.shape == (8, 2)
    assert torch.all((action[:, 0] >= 0) & (action[:, 0] <= 5))
    assert torch.all((action[:, 1] >= 0) & (action[:, 1] <= 1))
    assert distribution.log_prob(action).shape == (8, 1)
    assert distribution.entropy().shape == (8, 1)
    assert torch.all(distribution.concentration > 1.0)


def test_joint_kl_uses_old_mode_probability_weighting():
    old = _distribution()
    new = Task010ModeBetaDistribution(old.logits + torch.randn_like(old.logits) * 0.1, torch.ones_like(old.concentration_raw) * 0.2)
    expected = kl_divergence(old.categorical, new.categorical) + (
        old.categorical.probs[:, 1:] * kl_divergence(old.beta, new.beta)
    ).sum(-1)
    assert torch.allclose(old.kl(new).squeeze(-1), expected, atol=1e-6)


def test_invalid_shapes_or_nonfinite_inputs_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        Task010ModeBetaDistribution(torch.zeros((2, 5)), torch.zeros((2, 5, 2)))
    bad = torch.zeros((2, 6))
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="logits"):
        Task010ModeBetaDistribution(bad, torch.zeros((2, 5, 2)))
