import pytest
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Independent
from robotarm_magnetic_lab.learning.new_stomach_rl_distribution import BoundedNormal
from robotarm_magnetic_lab.learning.new_stomach_rl_actor import ContinuousGRUActor
from robotarm_magnetic_lab.learning.new_stomach_rl_runner import SmokePPO, Rollout


@pytest.mark.parametrize('dimension', [9, 36])
def test_tanh_density_matches_pytorch_and_gradients_finite(dimension):
    mean = torch.zeros((2,dimension), requires_grad=True)
    log_std = torch.full_like(mean, -.7, requires_grad=True)
    distribution = BoundedNormal(mean, log_std)
    action, latent = distribution.sample()
    reference = Independent(TransformedDistribution(Normal(mean, log_std.exp()), TanhTransform()), 1)
    assert (action.abs() < 1).all()
    assert torch.allclose(distribution.log_prob(action, latent), reference.log_prob(action), atol=1e-5)
    (-distribution.log_prob(action.detach(), latent.detach()).mean()).backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(log_std.grad).all()
    assert torch.isfinite(distribution.log_prob(torch.full_like(action, .999999))).all()


def test_recurrent_reset_rows_and_explicit_hidden_no_stale_state():
    actor = ContinuousGRUActor(9)
    observation = torch.randn((3,2,548))
    initial = torch.randn((1,2,256))
    masks = torch.tensor([[False,False],[True,False],[False,False]])
    full, _ = actor.parameters_sequence(observation, initial, masks)
    fresh, _ = actor.parameters_sequence(observation[1:,0:1], torch.zeros((1,1,256)))
    assert torch.allclose(full[1:,0], fresh[:,0], atol=1e-6)
    assert not torch.equal(full[1:,1], fresh[:,0])
    with pytest.raises(ValueError):
        actor.parameters_sequence(torch.zeros((1,549)), None)


@pytest.mark.parametrize('dimension', [9, 36])
def test_minimal_ppo_update_and_checkpoint_roundtrip(tmp_path, dimension):
    torch.manual_seed(8)
    runner = SmokePPO(dimension, 'cpu')
    observations = torch.randn((4,2,548))
    critic = torch.randn((4,2,28))
    hidden = torch.zeros((1,2,256))
    means, _ = runner.actor.parameters_sequence(observations, hidden)
    dist = BoundedNormal(means, runner.actor.log_std.expand_as(means))
    actions, latent = dist.sample()
    data = Rollout(observations, critic, actions.detach(), latent.detach(), dist.log_prob(actions,latent).detach(),
                   torch.randn((4,2)), torch.zeros((4,2),dtype=torch.bool), hidden,
                   means.detach(), torch.randn((2,28)))
    metric = runner.update(data)
    assert all(torch.isfinite(torch.tensor(value)) for value in metric.values())
    assert runner.update_count == 1
    path = tmp_path/'checkpoint.pt'; runner.save(path)
    restored = SmokePPO(dimension, 'cpu'); restored.load(path)
    assert restored.update_count == 1
    for a,b in zip(runner.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(a,b)
    for a,b in zip(runner.critic.parameters(), restored.critic.parameters()):
        assert torch.equal(a,b)
    wrong = SmokePPO(36 if dimension == 9 else 9, 'cpu')
    with pytest.raises(ValueError,match='dimension'):
        wrong.load(path)
