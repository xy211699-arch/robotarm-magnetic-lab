from __future__ import annotations

import torch

from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
from robotarm_magnetic_lab.learning.task010_critic import Task010Critic, Task010SelectiveNormalizer


def test_critic_shape_and_selective_normalization_freeze():
    critic = Task010Critic()
    obs = torch.randn((12, 65))
    values = critic(obs)
    assert values.shape == (12, 1)
    assert torch.isfinite(values).all()
    before = critic.normalizer.running_mean.clone()
    critic.freeze_normalizer()
    critic(obs * 2)
    assert torch.equal(before, critic.normalizer.running_mean)


def test_boolean_and_one_hot_fields_are_not_normalized():
    normalizer = Task010SelectiveNormalizer()
    obs = torch.randn((8, 65))
    obs[:, 22] = 1.0  # contact flag
    obs[:, 27] = 1.0  # wall-normal valid flag
    obs[:, 58:62] = torch.eye(4).repeat(2, 1)
    output = normalizer(obs)
    for index in (22, 27, 58, 59, 60, 61):
        assert torch.equal(output[:, index], obs[:, index])


def test_actor_and_critic_share_no_parameter_storage():
    actor, critic = Task010Actor(), Task010Critic()
    actor_ptrs = {parameter.data_ptr() for parameter in actor.parameters()}
    critic_ptrs = {parameter.data_ptr() for parameter in critic.parameters()}
    assert actor_ptrs.isdisjoint(critic_ptrs)


def test_normalizer_state_roundtrip_includes_statistics_and_freeze():
    first = Task010SelectiveNormalizer()
    first(torch.randn((16, 65)))
    first.freeze()
    second = Task010SelectiveNormalizer()
    second.load_state_dict(first.state_dict())
    assert torch.equal(first.running_mean, second.running_mean)
    assert torch.equal(first.running_var, second.running_var)
    assert second.frozen.item()
