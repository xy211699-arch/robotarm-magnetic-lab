import numpy as np
import pytest
import torch
from types import SimpleNamespace as S

from robotarm_magnetic_lab.runtime.new_stomach_rl_capacity_reference import (
    canonical_actions, MagneticInputTape,
)


def test_frozen_raw_sequences_not_legacy_preview_and_bounded():
    single, chunk = canonical_actions('single'), canonical_actions('chunk')
    assert single.shape == (20,9) and chunk.shape == (20,36)
    assert np.isfinite(chunk).all() and np.abs(chunk).max() <= 1
    assert not single[0].any() and not chunk[0].any()
    for axis in range(9):
        assert single[1+2*axis,axis] == .2
        assert single[2+2*axis,axis] == -.2
    # Fixture is a raw action sequence, not four repeats of a 4s preview.
    assert not np.array_equal(chunk[1].reshape(4,9)[0],chunk[1].reshape(4,9)[-1])
    with pytest.raises(ValueError):
        canonical_actions('unknown')


def tape():
    state = dict(raw_wrench=torch.zeros(1,12), wrench=torch.zeros(1,12))
    bridge = S(state=state, elapsed=torch.zeros(1), _filtered_wrench=torch.zeros(1,12),
        magnet_body_index=0,
        robot=S(data=S(body_pos_w=torch.zeros(1,1,3),body_quat_w=torch.zeros(1,1,4))),
        capsule=S(data=S(root_pos_w=torch.zeros(1,3),root_quat_w=torch.zeros(1,4),
            root_lin_vel_w=torch.zeros(1,3),root_ang_vel_w=torch.zeros(1,3))))
    calls=[]
    def original(env, env_ids=None):
        calls.append(len(calls)+1)
        bridge.elapsed += 1/240
        bridge.state['raw_wrench'][:] = len(calls)
        bridge._filtered_wrench[:] = len(calls)*.5
        bridge.state['wrench'][:] = bridge._filtered_wrench
    bridge.physics_step = original
    return MagneticInputTape(bridge,'cpu'), calls


def test_tape_reads_pre_model_input_and_post_model_output_calls_original_once():
    recorder, calls = tape()
    env = S(num_envs=1)
    recorder.begin()
    for i in range(240):
        recorder.physics_step(env)
    before, after = recorder.finish()
    assert len(calls)==240 and before.shape==(240,33) and after.shape==(240,25)
    assert before[0,20:32].count_nonzero()==0
    assert before[1,20:32].tolist()==[.5]*12
    assert after[0,:12].tolist()==[1.]*12
    assert after[-1,12:24].tolist()==[120.]*12
    assert after[0,-1] > before[0,-1]
    recorder.begin()
    assert before[1,20] == .5  # immutable snapshot, not overwritten next episode


def test_tape_ignores_reset_warmup_and_fail_closed_on_missing_ticks():
    recorder, calls = tape()
    env = S(num_envs=1)
    recorder.physics_step(env)
    assert recorder.count==0 and len(calls)==1
    recorder.begin()
    recorder.physics_step(env)
    with pytest.raises(RuntimeError, match='240'):
        recorder.finish()


def test_tape_rejects_nonfinite_and_vector_use_before_calling_bridge():
    recorder, calls = tape()
    recorder.begin()
    with pytest.raises(ValueError, match='single'):
        recorder.physics_step(S(num_envs=2))
    assert not calls
    for _ in range(240):
        recorder.physics_step(S(num_envs=1))
    recorder.inputs[0,0]=float('nan')
    with pytest.raises(RuntimeError, match='nonfinite'):
        recorder.finish()
