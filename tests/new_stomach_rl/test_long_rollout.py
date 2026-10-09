import pytest
import torch
from robotarm_magnetic_lab.learning.new_stomach_rl_distribution import BoundedNormal
from robotarm_magnetic_lab.learning.new_stomach_rl_actor import ContinuousGRUActor
from robotarm_magnetic_lab.learning.new_stomach_rl_long_rollout import (
    recurrent_step,replay_sequence,sequence_chunks,LongRollout,LongRolloutPPO)


@pytest.mark.parametrize('dim',[9,36])
def test_collect_replay_joint_density_with_asynchronous_warmup_and_chunks(dim):
    torch.manual_seed(4);actor=ContinuousGRUActor(dim)
    obs=torch.randn(6,2,548,requires_grad=True)
    valid=torch.tensor([[1,1],[1,1],[0,1],[1,1],[1,0],[1,1]],dtype=torch.bool)
    starts=torch.zeros_like(valid);starts[2,0]=True;starts[4,1]=True
    initial=torch.randn(1,2,256);hidden=initial.clone();means=[]
    for t in range(6):
        before=hidden.clone();mean,hidden=recurrent_step(actor,obs[t],hidden,starts[t],valid[t]);means.append(mean)
        for row in range(2):
            if not valid[t,row]:
                assert torch.equal(hidden[:,row],torch.zeros_like(hidden[:,row]) if starts[t,row] else before[:,row])
    means=torch.stack(means)
    dist=BoundedNormal(means,actor.log_std.expand_as(means));actions,latent=dist.sample()
    old=dist.log_prob(actions,latent)
    repeated,last,before=replay_sequence(actor,obs,initial,starts,valid)
    assert torch.allclose(means,repeated,atol=1e-7) and torch.allclose(last,hidden,atol=1e-7)
    assert torch.allclose(old[valid],BoundedNormal(repeated,actor.log_std.expand_as(means)).log_prob(actions,latent)[valid],atol=1e-6)
    # Distribution is JOINT, not a mean over action dimensions.
    jac=2*(torch.log(torch.tensor(2.))-latent-torch.nn.functional.softplus(-2*latent))
    assert torch.allclose(old,(dist.base.log_prob(latent)-jac).sum(-1),atol=1e-5)
    for chunk in sequence_chunks(obs,valid,starts,before,4):
        replay,_,_=replay_sequence(actor,chunk['observations'],chunk['initial_hidden'],chunk['episode_start'],chunk['valid'])
        row,start,count=chunk['row'],chunk['start'],chunk['count']
        assert torch.allclose(replay[:count,0],means[start:start+count,row],atol=1e-6)
        assert not chunk['valid'][count:].any()
    means[valid].sum().backward()
    assert torch.count_nonzero(obs.grad[~valid])==0


@pytest.mark.parametrize('dim',[9,36])
def test_masked_ppo_update_keeps_sample_and_executed_commands_separate(dim):
    torch.manual_seed(3);runner=LongRolloutPPO(dim,'cpu')
    obs=torch.randn(4,2,548,requires_grad=True);critic=torch.randn(4,2,28,requires_grad=True)
    valid=torch.tensor([[1,1],[0,1],[1,1],[1,0]],dtype=torch.bool)
    starts=torch.zeros_like(valid);starts[1,0]=True;starts[3,1]=True
    initial=torch.zeros(1,2,256)
    with torch.no_grad():
        means,_,_=replay_sequence(runner.actor,obs,initial,starts,valid)
        dist=BoundedNormal(means,runner.actor.log_std.expand_as(means));actions,latent=dist.sample()
        values=runner.critic(critic).squeeze(-1)
    terminated=torch.zeros_like(valid);truncated=terminated.clone();truncated[0,0]=True
    data=LongRollout(obs,critic,actions.detach(),latent.detach(),dist.log_prob(actions,latent).detach(),
        torch.randn(4,2),values,torch.ones(4,2),valid,starts,terminated,truncated,critic.clone(),
        torch.zeros(4,2,4,9),initial,means,runner.actor.log_std.detach().clone())
    metric=runner.update(data)
    assert metric['effective_samples']==6 and all(torch.isfinite(torch.tensor(x)) for x in metric.values())
    assert obs.grad[~valid].count_nonzero()==0 and critic.grad[~valid].count_nonzero()==0
    data.actions=data.actions*.5
    with pytest.raises(ValueError,match='sampled actions'): data.validate(dim)


def test_user_frozen_eight_env_default_and_formal_training_not_enabled():
    import ast,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    frozen=json.loads((root/'configs/new_stomach_rl/development_environment_v1.json').read_text())
    assert frozen['num_envs']==8 and frozen['formal_training_allowed'] is False
    path=root/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_new_stomach_rl_vector_env_cfg.py'
    tree=ast.parse(path.read_text())
    assigned=[n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and
        any(ast.unparse(t)=='self.scene.num_envs' for t in n.targets) and isinstance(n.value,ast.Constant)]
    assert assigned==[8]
