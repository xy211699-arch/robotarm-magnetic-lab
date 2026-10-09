"""Reuse the actual old schedule/servo code with fake assets and safe geometry."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import pytest
import torch
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_rows import RowAsset

PKG = 'robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab'
TASK = Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab'


def load(monkeypatch,name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module);spec.loader.exec_module(module)
    return module


def sdk_stubs(monkeypatch):
    package = ModuleType(PKG+'.mdp');package.__path__=[str(TASK/'mdp')]
    monkeypatch.setitem(sys.modules,PKG+'.mdp',package)
    managers = ModuleType('isaaclab.managers')
    class Base:
        def __init__(self,cfg,env): self.cfg,self._env=cfg,env
    managers.ManagerTermBase = Base
    monkeypatch.setitem(sys.modules,'isaaclab.managers',managers)
    actions = ModuleType('isaaclab.managers.action_manager');actions.ActionTerm=Base
    monkeypatch.setitem(sys.modules,actions.__name__,actions)
    config = ModuleType('isaaclab.managers.manager_term_cfg');config.ActionTermCfg=object
    monkeypatch.setitem(sys.modules,config.__name__,config)
    utils = ModuleType('isaaclab.utils.configclass');utils.configclass=lambda cls:cls
    monkeypatch.setitem(sys.modules,utils.__name__,utils)


def test_real_single_and_chunk_row_schedules_write_only_their_own_simulator_indices(monkeypatch):
    sdk_stubs(monkeypatch)
    old = load(monkeypatch,PKG+'.mdp.actuator_vector_action',TASK/'mdp/actuator_vector_action.py')
    load(monkeypatch,PKG+'.mdp.new_stomach_actuator_action',TASK/'mdp/new_stomach_actuator_action.py')
    rl = load(monkeypatch,PKG+'.mdp.new_stomach_rl_action',TASK/'mdp/new_stomach_rl_action.py')
    module = load(monkeypatch,PKG+'.mdp.new_stomach_rl_vector_action',TASK/'mdp/new_stomach_rl_vector_action.py')
    assert module.ScopedRLRowAction.process_actions is rl.NewStomachRLAction.process_actions
    assert module.ScopedRLRowAction.apply_actions is rl.NewStomachRLAction.apply_actions
    monkeypatch.setattr(old,'UrdfXrdfSafetyModel',lambda *args:SimpleNamespace())
    monkeypatch.setattr(module.ScopedRLRowAction,'_geometry',lambda self:None)
    monkeypatch.setattr(module.ScopedRLRowAction,'clearance',lambda self,path:1.)
    names=['j1','j2','j3','j4','j5','j6','ballxj','ballyj','ballzj']
    def wrap(value): return SimpleNamespace(torch=value)
    data = SimpleNamespace(joint_names=names,body_names=['l6','magl','base_link'],
        joint_pos=wrap(torch.zeros(2,9)),joint_vel=wrap(torch.zeros(2,9)),
        body_pos_w=wrap(torch.zeros(2,3,3)),body_quat_w=wrap(torch.tensor([0.,0,0,1.]).expand(2,3,4)),
        body_link_jacobian_w=wrap(torch.eye(6).repeat(2,3,1,1)),
        soft_joint_pos_limits=wrap(torch.tensor([-10.,10.]).expand(2,9,2)))
    writes=[]
    def position(**kwargs):
        ids=kwargs['env_ids'];writes.append(ids.tolist());data.joint_pos.torch[ids]=kwargs['target']
    robot=SimpleNamespace(data=data,is_fixed_base=False,num_base_dofs=0,
        permanent_wrench_composer=SimpleNamespace(reset=lambda **kw:None),
        set_joint_position_target_index=position,set_joint_velocity_target_index=lambda **kw:None)
    env=SimpleNamespace(num_envs=2,device='cpu',step_dt=1.,physics_dt=1/240,
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=1/240)),sim=None,event_manager=None,
        scene={'robot':robot,'capsule':SimpleNamespace(data=SimpleNamespace(root_link_pos_w=wrap(torch.zeros(2,3))),
            permanent_wrench_composer=SimpleNamespace(reset=lambda **kw:None))})
    for mode,dim in (('single',9),('chunk',36)):
        cfg=SimpleNamespace(mode=mode,required_clearance_m=.005,urdf_path='unused',xrdf_path='unused',
            collision_mesh_suffix='/Collider',gravity_compensation=False)
        cfg.copy=lambda:copy.copy(cfg)
        term=module.NewStomachRLVectorAction(cfg,env)
        raw=torch.zeros(2,dim);raw[1,-1]=.2
        term.process_actions(raw)
        for _ in range(240): term.apply_actions()
        assert writes[-480:] == [[0],[1]]*240
        assert term.rows[0].tick == term.rows[1].tick == 240
        assert term.executed_history[0].count_nonzero() == 0
        assert term.executed_history[1].count_nonzero() > 0
        history=term.executed_history[1].clone()
        target=term.rows[1]._target.clone()
        term.reset([0])
        assert torch.equal(term.executed_history[1],history)
        assert torch.equal(term.rows[1]._target,target)
        assert term.rows[0].tick == 0 and term.rows[1].tick == 240


def test_vector_wrapper_rejects_nonfinite_and_shape_before_planning(monkeypatch):
    sdk_stubs(monkeypatch)
    old = ModuleType(PKG+'.mdp.new_stomach_rl_action')
    old.NewStomachRLAction=object;old.NewStomachRLActionCfg=object
    monkeypatch.setitem(sys.modules,old.__name__,old)
    module=load(monkeypatch,PKG+'.mdp.new_stomach_rl_vector_action',TASK/'mdp/new_stomach_rl_vector_action.py')
    obj=module.NewStomachRLVectorAction.__new__(module.NewStomachRLVectorAction)
    obj.cfg=SimpleNamespace(mode='single');obj.rows=[None,None]
    for value in (torch.zeros(1,9),torch.full((2,9),float('inf'))):
        with pytest.raises(ValueError): obj.process_actions(value)
