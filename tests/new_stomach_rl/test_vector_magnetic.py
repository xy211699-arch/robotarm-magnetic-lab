"""Actual inherited bridge math, fake SI finite-model outputs, CPU only."""
from types import SimpleNamespace
import numpy as np
import torch
from test_vector_action import sdk_stubs,load,PKG,TASK
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_rows import RowEnvironment


def setup_bridge(cls,env,empty):
    b=cls.__new__(cls);b._env=env
    b.robot,b.capsule=env.scene['robot'],env.scene['capsule']
    b.magnet_body_index,b.base_body_index=0,1
    b.elapsed=torch.zeros(1);b.frame_count=0;b.state=empty(1,'cpu')
    b._filtered_wrench=torch.full((1,12),.003)
    b._field_last_position=np.full(3,np.nan)
    b.config={'simulation':{'max_force_n':.02,'max_torque_nm':.0009,'wrench_filter_time_constant_s':.15,'apply_forces':True},
        'magnets':{'target_cylinder':{'center_offset_axis_m':-.004}},
        'external_magnet':{'capsule':{'linear_drag_n_per_m_s':.1,'angular_drag_nm_per_rad_s':.002},
            'coupling_ramp_s':.2,'release_delay_s':.1}}
    calls=[]
    def force(*args,reverse=False):
        calls.append(tuple(np.array(v).copy() for v in args))
        return np.array([.04,.03,-.05])*(1 if reverse else -1),np.array([.01,.02,.03])
    b.model=SimpleNamespace(force_torque_on_cube_si=lambda *args:force(*args,reverse=True),force_torque_si=force)
    b.collision_model=SimpleNamespace(clearance_by_frame=lambda robot,idx:{'l4':.01})
    b._update_field_visualization=lambda *args:None;b._log=lambda msg:None
    return b,calls


def make_parent():
    writes=[]
    def wrap(value): return SimpleNamespace(torch=value)
    composer=SimpleNamespace(set_forces_and_torques_index=lambda **kw:writes.append(kw),reset=lambda **kw:writes.append(kw))
    body=torch.tensor([[[0.,0,.2],[0,0,0]],[[4.,0,.2],[4,0,0]]])
    quats=torch.tensor([0.,0,0,1.]).expand(2,2,4)
    robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=wrap(body),body_quat_w=wrap(quats),joint_pos=wrap(torch.zeros(2,6))),permanent_wrench_composer=composer)
    capsule=SimpleNamespace(data=SimpleNamespace(root_pos_w=wrap(torch.tensor([[0.,0,0],[4.,0,0]])),
        root_quat_w=wrap(torch.tensor([0.,0,0,1.]).expand(2,4)),
        root_lin_vel_w=wrap(torch.full((2,3),.1)),root_ang_vel_w=wrap(torch.full((2,3),.2))),permanent_wrench_composer=composer)
    parent=SimpleNamespace(num_envs=2,device='cpu',cfg=None,step_dt=1.,physics_dt=1/240,sim=None,event_manager=None,
        scene={'robot':robot,'capsule':capsule})
    return parent,writes


def test_vector_inherits_identical_original_raw_clip_drag_ramp_and_filter(monkeypatch):
    sdk_stubs(monkeypatch)
    old=load(monkeypatch,PKG+'.mdp.legacy_bridge',TASK/'mdp/legacy_bridge.py')
    new=load(monkeypatch,PKG+'.mdp.new_stomach_rl_vector_magnetic',TASK/'mdp/new_stomach_rl_vector_magnetic.py')
    assert new.ScopedMagneticRow.physics_step is old.LegacyMagneticCollisionBridge.physics_step
    parent,writes=make_parent();local=RowEnvironment(parent,1)
    original,_=setup_bridge(old.LegacyMagneticCollisionBridge,local,old._empty_state)
    adapted,calls=setup_bridge(new.ScopedMagneticRow,local,old._empty_state)
    for _ in range(240):
        original.physics_step(local);adapted.physics_step(local)
        assert torch.equal(original.state['raw_wrench'],adapted.state['raw_wrench'])
        assert torch.equal(original.state['wrench'],adapted.state['wrench'])
        assert torch.equal(original.elapsed,adapted.elapsed)
    assert len(calls)==480
    assert all(v['env_ids'].tolist()==[1] for v in writes)
    assert adapted.state['raw_wrench'][0,:3].tolist()==original.state['raw_wrench'][0,:3].tolist()
    assert torch.linalg.vector_norm(adapted.state['wrench'][0,6:9])<=.02


def test_partial_magnetic_reset_preserves_other_private_filter_and_elapsed(monkeypatch):
    sdk_stubs(monkeypatch)
    old=load(monkeypatch,PKG+'.mdp.legacy_bridge',TASK/'mdp/legacy_bridge.py')
    new=load(monkeypatch,PKG+'.mdp.new_stomach_rl_vector_magnetic',TASK/'mdp/new_stomach_rl_vector_magnetic.py')
    parent,writes=make_parent()
    rows=[setup_bridge(new.ScopedMagneticRow,RowEnvironment(parent,i),old._empty_state)[0] for i in range(2)]
    group=new.NewStomachRLVectorMagnetic.__new__(new.NewStomachRLVectorMagnetic);group.rows=rows
    for _ in range(24): group.physics_step(parent)
    snapshot=rows[1]._filtered_wrench.clone();elapsed=rows[1].elapsed.clone();count=rows[1].frame_count
    group.reset([0])
    assert rows[0]._filtered_wrench.count_nonzero()==0 and rows[0].elapsed.item()==0
    assert torch.equal(rows[1]._filtered_wrench,snapshot) and torch.equal(rows[1].elapsed,elapsed)
    assert rows[1].frame_count==count
    assert writes[-1]['env_ids'].tolist()==[0]
