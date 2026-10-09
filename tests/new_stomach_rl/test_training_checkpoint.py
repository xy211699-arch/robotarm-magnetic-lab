from types import SimpleNamespace
import random
import numpy as np
import pytest
import torch
from robotarm_magnetic_lab.learning.new_stomach_rl_checkpoint import (
    save_checkpoint,restore_checkpoint,configuration_sha)


def setup():
    actor=torch.nn.Linear(3,2);critic=torch.nn.Linear(2,1)
    runner=SimpleNamespace(actor=actor,critic=critic,dimension=9,gamma=.9,gae_lambda=.8,update_count=7,
        optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters())))
    sum(p.sum() for p in list(actor.parameters())+list(critic.parameters())).backward();runner.optimizer.step()
    libs=[SimpleNamespace(records=[{'pose_id':f'train-{i}'}],fixed=None,rng=np.random.default_rng(i)) for i in range(8)]
    config={'num_envs':8,'purpose':'cpu_fixture'}
    identity=dict(config_sha256=configuration_sha(config),code_sha256='a'*64,assets_sha256='b'*64,weights_sha256='c'*64)
    return runner,libs,config,identity


def test_boundary_restores_parameters_optimizer_counters_rng_and_samplers(tmp_path):
    runner,libs,config,identity=setup();path=tmp_path/'boundary.pt'
    torch.manual_seed(7);np.random.seed(7);random.seed(7)
    info=save_checkpoint(path,runner,libs,config,identity,512,at_episode_boundary=True,statistics={'value':torch.tensor([2.])})
    expected=(random.random(),np.random.random(),torch.rand(2),[lib.rng.integers(10000) for lib in libs])
    saved=[p.detach().clone() for p in runner.actor.parameters()]
    for p in runner.actor.parameters(): p.data.zero_()
    runner.update_count=99;runner.optimizer.state.clear();events=[]
    result=restore_checkpoint(path,info['sha256'],runner,libs,config,identity,
        lambda:events.append('reset'),lambda:events.append('clear'))
    actual=(random.random(),np.random.random(),torch.rand(2),[lib.rng.integers(10000) for lib in libs])
    assert actual[0:2]==expected[0:2] and torch.equal(actual[2],expected[2]) and actual[3]==expected[3]
    assert all(torch.equal(a,b) for a,b in zip(saved,runner.actor.parameters()))
    assert runner.optimizer.state and runner.update_count==7 and result['effective_samples']==512
    assert events==['reset','clear'] and result['trajectory_continuous'] is False
    with pytest.raises(FileExistsError): save_checkpoint(path,runner,libs,config,identity,512,at_episode_boundary=True)


def test_mid_episode_is_explicit_weights_only_not_continuation(tmp_path):
    runner,libs,config,identity=setup();path=tmp_path/'weights.pt'
    with pytest.raises(ValueError,match='mid-episode'): save_checkpoint(path,runner,libs,config,identity,512,at_episode_boundary=False)
    info=save_checkpoint(path,runner,libs,config,identity,512,at_episode_boundary=False,allow_weights_only=True)
    with pytest.raises(ValueError,match='weights-only'): restore_checkpoint(path,info['sha256'],runner,libs,config,identity,lambda:None,lambda:None)
    result=restore_checkpoint(path,info['sha256'],runner,libs,config,identity,lambda:None,lambda:None,allow_weights_only=True)
    assert runner.update_count==0 and not runner.optimizer.state and result['effective_samples']==0


def test_identity_sampler_and_bytes_mismatch_rejected_before_reset(tmp_path):
    runner,libs,config,identity=setup();path=tmp_path/'boundary.pt'
    info=save_checkpoint(path,runner,libs,config,identity,512,at_episode_boundary=True)
    def reset(): pytest.fail('invalid evidence must not reset environment')
    with pytest.raises(ValueError,match='bytes'): restore_checkpoint(path,'0'*64,runner,libs,config,identity,reset,reset)
    wrong=dict(identity,assets_sha256='d'*64)
    with pytest.raises(ValueError,match='identity'): restore_checkpoint(path,info['sha256'],runner,libs,config,wrong,reset,reset)
    libs[0].records=[{'pose_id':'wrong'}]
    with pytest.raises(ValueError,match='sampler'): restore_checkpoint(path,info['sha256'],runner,libs,config,identity,reset,reset)


def test_nonfinite_training_state_is_not_published(tmp_path):
    runner,libs,config,identity=setup();path=tmp_path/'invalid.pt'
    with torch.no_grad(): next(runner.actor.parameters()).fill_(float('nan'))
    with pytest.raises(ValueError,match='nonfinite'):
        save_checkpoint(path,runner,libs,config,identity,512,at_episode_boundary=True)
    assert not path.exists()
