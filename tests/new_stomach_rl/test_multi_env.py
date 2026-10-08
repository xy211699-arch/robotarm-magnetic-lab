from pathlib import Path
import torch
import pytest
from robotarm_magnetic_lab.runtime.new_stomach_rl_reward import NewStomachReward
from robotarm_magnetic_lab.learning.new_stomach_rl_actor import ContinuousGRUActor


def test_single_index_guard_not_removed_and_runtime_rejects_unisolated_rows():
    from robotarm_magnetic_lab.runtime.new_stomach_rl_runtime import NewStomachRLRuntime
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='isolation'):
        NewStomachRLRuntime(SimpleNamespace(num_envs=2),None)
    source = Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/actuator_vector_action.py'
    assert 'env.num_envs!=1' in source.read_text()


def test_identical_actor_rows_agree_different_histories_diverge():
    actor = ContinuousGRUActor(9)
    observation = torch.zeros((2,548))
    hidden = torch.zeros((1,2,256))
    mean,_ = actor.parameters_sequence(observation,hidden)
    assert torch.equal(mean[0],mean[1])
    observation[1,512:] = .5
    varied,_ = actor.parameters_sequence(observation,hidden)
    assert not torch.equal(varied[0],varied[1])


def test_reward_row_reset_and_increments_do_not_cross_contaminate():
    runtime = NewStomachReward(2,'cpu')
    runtime.reset_reward([0,1],[.1,.2])
    pose = torch.zeros((2,7));pose[:,6]=1
    for index in range(10):
        runtime.update_reward_10hz(pose,[.1+.001*(index+1),.2])
    total = runtime.finish_policy_second()
    assert total[0].item() == pytest.approx(1,abs=1e-5)
    assert total[1].item() == 0
    runtime.reset_reward([1],[.3])
    assert len(runtime.trackers[0]._coverage_history) == 10
    assert len(runtime.trackers[1]._coverage_history) == 0
