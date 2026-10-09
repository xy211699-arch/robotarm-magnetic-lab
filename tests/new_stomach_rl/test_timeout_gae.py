import pytest
import torch
from robotarm_magnetic_lab.learning.new_stomach_rl_long_rollout import boundary_gae


def run(r,v,b,valid=None,start=None,terminated=None,truncated=None):
    r=torch.tensor(r,dtype=torch.float64).reshape(-1,1)
    zeros=torch.zeros_like(r,dtype=torch.bool)
    mask=lambda x: zeros if x is None else torch.tensor(x,dtype=torch.bool).reshape(-1,1)
    return boundary_gae(r,torch.tensor(v,dtype=torch.float64).reshape(-1,1),torch.tensor(b,dtype=torch.float64).reshape(-1,1),
        ~zeros if valid is None else mask(valid),mask(start),mask(terminated),mask(truncated),.9,.8)


def test_ordinary_and_rollout_tail_hand_calculation():
    a,ret=run([1.,2.],[.5,.7],[.7,1.])
    # tail delta=2+.9*1-.7=2.2; first=1+.9*.7-.5+.72*2.2
    assert torch.allclose(a[:,0],torch.tensor([2.714,2.2],dtype=torch.float64))
    assert torch.allclose(ret[:,0],torch.tensor([3.214,2.9],dtype=torch.float64))


def test_timeout_bootstraps_final_value_but_stops_advantage_recursion():
    a,_=run([1.,100.],[.5,10.],[3.,11.],start=[False,True],truncated=[True,False])
    assert float(a[0])==pytest.approx(3.2)


def test_true_termination_has_zero_bootstrap():
    a,_=run([1.,100.],[.5,10.],[999.,11.],terminated=[True,False])
    assert float(a[0])==pytest.approx(.5)


def test_reset_padding_does_not_inherit_reward_or_returns():
    a,ret=run([1.,float('nan'),100.],[.5,float('nan'),10.],[3.,float('nan'),11.],
        valid=[True,False,True],start=[False,True,False],truncated=[True,False,False])
    assert float(a[0])==pytest.approx(3.2)
    assert float(a[1])==0 and float(ret[1])==0


def test_episode_start_breaks_recursion_even_without_done():
    a,_=run([1.,100.],[.5,10.],[3.,11.],start=[False,True])
    assert float(a[0])==pytest.approx(3.2)
