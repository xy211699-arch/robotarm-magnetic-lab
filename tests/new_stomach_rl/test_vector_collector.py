from types import SimpleNamespace
import pytest
import torch
from robotarm_magnetic_lab.learning.new_stomach_rl_collector import VectorRolloutCollector,join_rollouts
from robotarm_magnetic_lab.learning.new_stomach_rl_long_rollout import LongRolloutPPO


class FakeVector:
    num_envs=2
    def __init__(self,dimension):
        self.dimension=dimension;self.tick=0
        self.lifecycle=SimpleNamespace(active=torch.ones(2,dtype=torch.bool),episode_seconds=torch.zeros(2,dtype=torch.long))
    def observation(self,value=0.):
        return dict(policy=torch.full((2,548),value),critic=torch.full((2,28),value))
    def step(self,action):
        self.tick+=1;valid=self.lifecycle.active.clone()
        self.lifecycle.episode_seconds[valid]+=1;self.lifecycle.active[:]=True
        truncated=torch.tensor([self.tick==2,False]);terminated=torch.tensor([False,self.tick==4])
        final=self.observation(float(self.tick));following=self.observation(float(self.tick))
        for row in (truncated|terminated).nonzero().flatten():
            self.lifecycle.active[row]=False;self.lifecycle.episode_seconds[row]=0
            following['critic'][row]=99.;following['policy'][row]=0.
        return following,torch.ones(2),terminated,truncated,dict(valid_transition=valid,physical_substeps=240,final_obs=final)


@pytest.mark.parametrize('dimension',[9,36])
def test_real_collector_contract_with_timeout_true_terminal_and_warmup(dimension):
    torch.manual_seed(8);runner=LongRolloutPPO(dimension,'cpu');env=FakeVector(dimension)
    collector=VectorRolloutCollector(env,runner,env.observation())
    first=collector.collect(3)
    assert first.valid.tolist()==[[True,True],[True,True],[False,True]]
    assert first.truncated[1,0] and first.episode_start[2,0]
    assert first.final_critic_inputs[1,0,0]==2. and first.final_critic_inputs[1,0,0]!=99.
    assert collector.hidden[:,0].count_nonzero()==0
    # Match collection's per-boundary batch shape; BLAS may round a flattened
    # [T*N] matrix differently from T separate [N] calls.
    with torch.no_grad(): expected=torch.stack([runner.critic(x).squeeze(-1) for x in first.final_critic_inputs])
    assert torch.equal(first.bootstrap_values,expected)
    assert first.executed_history[1,0].eq(2.).all()
    assert collector.samples==5 and collector.timeouts.tolist()==[1,0]
    metric=runner.update(first);assert metric['effective_samples']==5
    second=collector.collect(2)
    assert second.terminated[0,1] and not second.valid[1,1]
    assert collector.hidden[:,1].count_nonzero()==0
    assert all(torch.isfinite(x).all() for x in (second.rewards,second.bootstrap_values))


def test_split_rollouts_join_keeps_order_and_pre_reset_observation():
    torch.manual_seed(5);runner=LongRolloutPPO(9,'cpu');env=FakeVector(9)
    collector=VectorRolloutCollector(env,runner,env.observation())
    first=collector.collect(1);second=collector.collect(2)
    joined=join_rollouts([first,second]);joined.validate(9)
    assert joined.valid.tolist()==[[True,True],[True,True],[False,True]]
    assert torch.equal(joined.initial_hidden,first.initial_hidden)
    assert runner.update(joined)['effective_samples']==5
    second.old_log_std=second.old_log_std+1
    with pytest.raises(ValueError,match='policy changed'):join_rollouts([first,second])
