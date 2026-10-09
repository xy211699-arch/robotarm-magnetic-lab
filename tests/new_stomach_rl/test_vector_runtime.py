import pytest
import torch
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_lifecycle import VectorLifecycle, prime_zero_compatible_reward
from robotarm_magnetic_lab.runtime.new_stomach_rl_reward import NewStomachReward


def test_partial_reset_never_advances_other_row_or_generates_warmup_sample():
    life = VectorLifecycle(2,'cpu')
    life.reset_rows([0,1])
    assert life.begin(torch.ones(2,9)).count_nonzero() == 0
    first = life.finish(240)
    assert first['episode_start'].tolist() == [True,True]
    assert first['valid_transition'].tolist() == [False,False]
    for _ in range(3):
        life.begin(torch.ones(2,9)); life.finish(240)
    clock = life.global_tick
    life.reset_rows([0])
    assert life.global_tick == clock and life.episode_seconds.tolist() == [0,3]
    command = life.begin(torch.ones(2,9))
    assert command[0].count_nonzero() == 0 and command[1].count_nonzero() == 9
    result = life.finish(240)
    assert result['valid_transition'].tolist() == [False,True]
    assert result['episode_start'].tolist() == [True,False]
    assert life.episode_seconds.tolist() == [0,4] and life.global_tick == clock+240


def test_lifecycle_checks_steps_and_reset_boundary():
    life = VectorLifecycle(2,'cpu'); life.begin(torch.zeros(2,36))
    with pytest.raises(RuntimeError): life.reset_rows([0])
    with pytest.raises(RuntimeError): life.finish(239)
    life.finish(240)
    with pytest.raises(RuntimeError): life.finish(240)
    with pytest.raises(ValueError): life.begin(torch.zeros(2,10))
    with pytest.raises(ValueError): life.begin(torch.full((2,9),float('nan')))


def test_zero_c0_primes_old_reward_without_rewarding_initialization():
    reward = NewStomachReward(1,'cpu')
    prime_zero_compatible_reward(reward, [0.])
    assert reward._counts.item() == 0 and len(reward.trackers[0]._position_history) == 0
    pose = torch.tensor([[0.,0,0,0,0,0,1.]])
    for tick in range(1,11):
        reward.update_reward_10hz(pose,[tick*.001])
    assert reward.finish_policy_second().item() == pytest.approx(1.,abs=1e-5)
    with pytest.raises(ValueError): NewStomachReward(1,'cpu').reset_reward([0],[0.])
    for bad in (-.1,1.1,float('nan')):
        with pytest.raises(ValueError): prime_zero_compatible_reward(reward,[bad])


def test_batch_rgb_captured_once_and_associated_with_private_row_clocks():
    from types import SimpleNamespace
    from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_runtime import NewStomachRLVectorRuntime,SharedFrameRow
    runtime=NewStomachRLVectorRuntime.__new__(NewStomachRLVectorRuntime)
    runtime.env=SimpleNamespace(num_envs=2,device='cpu',scene={'capsule_camera':None})
    runtime.global_second=-1
    calls=[]
    def sample(second,tick,camera):
        calls.append((second,tick))
        return torch.full((2,2,2,3),second,dtype=torch.uint8),torch.tensor([second+1,second+1])
    runtime.rgb_clock=SimpleNamespace(sample=sample,sync=SimpleNamespace(last_forced_capture=False))
    initialized=[]
    runtime.rows=[SimpleNamespace(term=SimpleNamespace(reset=lambda:None),initialize=lambda:initialized.append(0),finish_second=lambda:torch.tensor([1.])),
        SimpleNamespace(term=SimpleNamespace(reset=lambda:None),initialize=lambda:initialized.append(1),finish_second=lambda:torch.tensor([2.]))]
    assert runtime.finish_batch(torch.tensor([False,False])).tolist()==[0.,0.]
    assert calls==[(0,0)] and initialized==[0,1]
    clocks=[SharedFrameRow(runtime,i) for i in range(2)]
    images=[clock.sample(0,0,None)[0] for clock in clocks]
    images[0].fill_(255)
    assert runtime.rgb.count_nonzero()==0 and images[1].count_nonzero()==0
    assert runtime.finish_batch(torch.tensor([False,True])).tolist()==[0.,2.]
    assert calls==[(0,0),(1,240)]
    clocks[1].sample(1,240,None)
    with pytest.raises(ValueError): clocks[1].sample(1,240,None)


def test_reset_discards_only_selected_visual_reward_and_cover_states():
    from types import SimpleNamespace
    from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_runtime import NewStomachRLVectorRuntime
    runtime=NewStomachRLVectorRuntime.__new__(NewStomachRLVectorRuntime)
    runtime.env=SimpleNamespace(device='cpu')
    sync_resets=[]
    runtime.rgb_clock=SimpleNamespace(sync=SimpleNamespace(reset_rows=lambda ids:sync_resets.append(ids.tolist())))
    reset_records=[]
    runtime.rows=[]
    for i in range(2):
        reward=NewStomachReward(1,'cpu');reward.reset_reward([0],[.1])
        runtime.rows.append(SimpleNamespace(ready=True,coverage=SimpleNamespace(reset=lambda ids,i=i:reset_records.append(i)),
            reward=reward,ten_hz_records=[i],one_hz_records=[i],encoder=None,visual_features=torch.full((1,512),float(i+1))))
    old=runtime.rows[1].visual_features.clone()
    runtime.reset_rows([0])
    assert sync_resets==[[0]] and reset_records==[0]
    assert not runtime.rows[0].ready and runtime.rows[1].ready
    assert runtime.rows[0].visual_features.count_nonzero()==0
    assert torch.equal(runtime.rows[1].visual_features,old)
    assert runtime.rows[1].ten_hz_records==[1]
