import torch
import pytest

from robotarm_magnetic_lab.runtime.new_stomach_rl_reward import NewStomachReward


def pose(n=1):
    value = torch.zeros((n,7)); value[:, 6] = 1.
    return value


def test_c0_not_rewarded_and_exact_four_terms_ten_tick_sum():
    runtime = NewStomachReward(1, 'cpu')
    runtime.reset_reward([0], torch.tensor([.2]))
    rows = []
    for tick in range(1,11):
        row = runtime.update_reward_10hz(pose(), torch.tensor([.2+.001*tick]))
        rows.append(row)
        assert torch.allclose(row.total_reward, row.coverage_reward+row.escape_reward+
            row.no_progress_reward+row.coverage_resumed_reward)
    assert runtime.finish_policy_second().item() == pytest.approx(1., abs=1e-5)
    with pytest.raises(RuntimeError, match='ten'):
        runtime.finish_policy_second()


def test_partial_second_cannot_return_ppo_reward():
    runtime = NewStomachReward(1, 'cpu'); runtime.reset_reward([0], torch.tensor([.1]))
    runtime.update_reward_10hz(pose(), torch.tensor([.1]))
    with pytest.raises(RuntimeError, match='ten'):
        runtime.finish_policy_second()


def test_five_second_window_and_no_progress_penalty_preserved():
    runtime = NewStomachReward(1, 'cpu'); runtime.reset_reward([0], torch.tensor([.1]))
    for tick in range(1,51):
        row = runtime.update_reward_10hz(pose(), torch.tensor([.1]))
        if tick < 50:
            assert not row.no_progress.any()
        if tick%10 == 0:
            total = runtime.finish_policy_second()
    assert row.no_progress.item()
    assert total.item() == pytest.approx(-.002)


def test_reset_rows_do_not_clear_other_recovery_histories():
    runtime = NewStomachReward(2, 'cpu'); runtime.reset_reward([0,1], torch.tensor([.1,.1]))
    for tick in range(10):
        runtime.update_reward_10hz(pose(2), torch.tensor([.1,.1]))
    runtime.finish_policy_second()
    runtime.reset_reward([1], torch.tensor([.2]))
    assert len(runtime.trackers[0]._position_history) == 10
    assert len(runtime.trackers[1]._position_history) == 0


def test_no_coverage_farming_without_new_area_and_bad_inputs_rejected():
    runtime = NewStomachReward(1, 'cpu'); runtime.reset_reward([0], torch.tensor([.1]))
    for tick in range(10):
        row = runtime.update_reward_10hz(pose(), torch.tensor([.1]))
        assert row.coverage_reward.item() == 0
    assert runtime.finish_policy_second().item() == 0
    with pytest.raises(RuntimeError, match='decreased'):
        runtime.update_reward_10hz(pose(), torch.tensor([.09]))


def test_adapter_matches_old_state_machine_through_escape_resume_and_lock():
    from robotarm_magnetic_lab.runtime.task010_recovery import Task010RecoveryTracker
    runtime = NewStomachReward(1, 'cpu')
    runtime.reset_reward([0], [.1])
    old = Task010RecoveryTracker(1, 'cpu', .1)
    old.previous_coverage.fill_(.1)
    old._initialized.fill_(True)
    totals = []
    phases = set()
    for tick in range(1, 231):
        state = pose()
        state[0, 0] = .0031 if tick >= 51 else 0.
        coverage = torch.tensor([.102 if tick >= 52 else .1])
        actual = runtime.update_reward_10hz(state, coverage)
        expected = old.update(state[:, :3], state[:, 3:], coverage, .1)
        for name in ('coverage_reward', 'escape_reward', 'no_progress_reward',
                     'coverage_resumed_reward', 'phase_one_hot_4', 'total_reward'):
            assert torch.equal(getattr(actual, name), getattr(expected, name))
        totals.append(expected.total_reward)
        phases.add(int(expected.phase_one_hot_4.argmax()))
        if tick % 10 == 0:
            assert torch.allclose(runtime.finish_policy_second(), torch.stack(totals).sum(0))
            totals.clear()
    assert phases == {0, 1, 2, 3}
