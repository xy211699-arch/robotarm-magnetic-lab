from __future__ import annotations

import torch

from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
from robotarm_magnetic_lab.learning.task010_critic import Task010Critic
from robotarm_magnetic_lab.learning.task010_ppo import (
    Task010PPO,
    Task010RolloutStorage,
    compute_task010_gae,
)


def test_true_task_terminal_has_zero_bootstrap():
    rewards = torch.tensor([[2.0]])
    values = torch.tensor([[1.0]])
    returns, _ = compute_task010_gae(
        rewards, values, torch.tensor([[True]]), torch.tensor([9.0]),
        gamma=0.999, lam=0.95, sampler_interrupted=False,
    )
    assert returns[-1].item() == 2.0


def test_sampler_interruption_bootstraps():
    rewards = torch.tensor([[2.0]])
    values = torch.tensor([[1.0]])
    returns, _ = compute_task010_gae(
        rewards, values, torch.tensor([[False]]), torch.tensor([9.0]),
        gamma=0.999, lam=0.95, sampler_interrupted=True,
    )
    assert returns[-1].item() == pytest.approx(2.0 + 0.999 * 9.0)


def _storage() -> Task010RolloutStorage:
    torch.manual_seed(4)
    storage = Task010RolloutStorage(rollout_steps=4, num_envs=4, device="cpu")
    actor = Task010Actor()
    critic = Task010Critic()
    for step in range(4):
        actor_obs = torch.randn((4, 519))
        critic_obs = torch.randn((4, 65))
        hidden = actor.get_hidden_state().clone()
        action = actor(actor_obs, stochastic_output=True)
        storage.add(
            actor_observation=actor_obs, critic_observation=critic_obs,
            action=action, reward=torch.randn(4) * 0.01,
            terminated=torch.zeros(4, dtype=torch.bool), reset_mask=torch.zeros(4, dtype=torch.bool),
            value=critic(critic_obs).detach().squeeze(1),
            log_prob=actor.get_output_log_prob(action).detach().squeeze(1),
            distribution_parameters=actor.output_distribution_params(), hidden_state=hidden,
        )
    storage.compute_returns(torch.zeros(4), gamma=0.999, lam=0.95, sampler_interrupted=True)
    return storage


def test_recurrent_batches_preserve_environment_time_order():
    storage = _storage()
    seen = []
    for batch in storage.recurrent_batches(num_mini_batches=2, num_epochs=1):
        assert batch["actor_observation"].shape[0] == 4
        seen.extend(batch["env_ids"].tolist())
    assert sorted(seen) == [0, 1, 2, 3]


def test_real_ppo_update_changes_actor_and_critic_parameters():
    storage = _storage()
    actor, critic = Task010Actor(), Task010Critic()
    algorithm = Task010PPO(actor, critic, num_learning_epochs=1, num_mini_batches=2)
    actor_before = torch.cat([parameter.detach().flatten() for parameter in actor.parameters()])
    critic_before = torch.cat([parameter.detach().flatten() for parameter in critic.parameters()])
    diagnostics = algorithm.update(storage)
    actor_after = torch.cat([parameter.detach().flatten() for parameter in actor.parameters()])
    critic_after = torch.cat([parameter.detach().flatten() for parameter in critic.parameters()])
    assert torch.linalg.vector_norm(actor_after - actor_before).item() > 0.0
    assert torch.linalg.vector_norm(critic_after - critic_before).item() > 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values() if isinstance(value, float))


import pytest
