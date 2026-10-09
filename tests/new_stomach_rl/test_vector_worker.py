"""Run the complete validator with fake Kit/physics, never initialize GPU."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import numpy as np
import pytest
import torch
from test_magnetic_equivalence import make_reference
from robotarm_magnetic_lab.runtime.new_stomach_rl_reference_audit import audit_reference_pair
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_lifecycle import VectorLifecycle


@pytest.mark.parametrize('substeps,origin',[(240,0.),(239,0.),(240,1.)])
def test_complete_fake_validator_saves_evidence_or_stops_without_retry(tmp_path,monkeypatch,substeps,origin):
    scripts=Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('_p2_fake_worker',scripts/'validate_vector_isolation.py')
    script=importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
    pair=[make_reference(tmp_path/'single','single'),make_reference(tmp_path/'chunk','chunk')]
    manifest=tmp_path/'registry.json';manifest.write_text(json.dumps(audit_reference_pair(*pair)))
    monkeypatch.setattr(script,'audit_assets',lambda *args:{'fake_cpu_fixture':True})
    for name in ('poses','mask'): (tmp_path/name).write_text('fixture')
    class Coverage:
        def __init__(self):
            self.c10,self.c1=SimpleNamespace(mask=torch.tensor([[True]+[False]*7])),SimpleNamespace(mask=torch.tensor([[True]+[False]*7]))
            self.c10_records=[]
            self._boundary_visible=self.c10.mask.clone()
        def _update(self,accumulator,visible,sample,tick,records):
            records.append(dict(physics_tick=tick,sample_id=sample,sim_time_s=tick/240,coverage=[.1]))
            return SimpleNamespace(visible_mask=visible)
    class Row:
        def __init__(self):
            self.coverage=Coverage();self.policy_second=0;self.ready=False;self.ten_hz_records=[];self.one_hz_records=[]
            self.term=SimpleNamespace(executed_history=torch.zeros(1,4,9))
        def boundary(self,active):
            coverage=self.coverage
            if not active:
                self.policy_second=0;coverage.c10_records=[]
                coverage._update(coverage.c10,coverage._boundary_visible,0,0,coverage.c10_records)
                coverage._update(coverage.c1,coverage._boundary_visible,0,0,[])
            else:
                for tick in range(1,11):
                    absolute=self.policy_second*240+tick*24
                    coverage._update(coverage.c10,coverage._boundary_visible,absolute//24,absolute,coverage.c10_records)
                self.policy_second+=1
                coverage._update(coverage.c1,coverage._boundary_visible,self.policy_second,self.policy_second*240,[])
            self.ten_hz_records=coverage.c10_records
            self.one_hz_records.append(dict(policy_second=self.policy_second,rgb_frame=self.policy_second+1,
                sim_time_s=float(self.policy_second),coverage=[.1],rgb_sha256='0'*64))
    class Scene(dict):
        def update(self,dt): pass
    env=SimpleNamespace(num_envs=2,device='cuda:0',physics_dt=1/240,sim=SimpleNamespace(device='cuda:0'),
        runtime=SimpleNamespace(rows=[Row(),Row()],rgb=SimpleNamespace(device='cuda:0')),scene=Scene(),_sim_step_counter=0)
    original_tensor=torch.tensor
    def cpu_tensor(*args,**kwargs):
        if str(kwargs.get('device','')).startswith('cuda'): kwargs['device']='cpu'
        return original_tensor(*args,**kwargs)
    monkeypatch.setattr(torch,'tensor',cpu_tensor)
    life=VectorLifecycle(2,'cpu');steps=[];closes=[]
    terms=[SimpleNamespace(capsule=None,env_origin=np.zeros(3)) for _ in range(2)]
    terms[1].env_origin[0]=origin
    term=SimpleNamespace(rows=terms,executed_history=torch.zeros(2,4,9))
    bridges=[SimpleNamespace(physics_step=lambda *args:None) for _ in range(2)]
    env.action_manager=SimpleNamespace(get_term=lambda name:term)
    env.event_manager=SimpleNamespace(get_term_cfg=lambda name:SimpleNamespace(func=SimpleNamespace(rows=bridges)))
    env.close=lambda:closes.append(1)
    def finish(valid):
        for i,row in enumerate(env.runtime.rows): row.boundary(bool(valid[i]))
        return torch.zeros(2)
    env.runtime.finish_batch=finish
    def reset_rows(ids):
        life.reset_rows(ids)
        for i in ids: env.runtime.rows[int(i)].ready=False
    env.reset_rows=reset_rows
    def step(action):
        life.begin(action);valid=life._valid.clone();steps.append(1)
        for _ in range(substeps):
            env._sim_step_counter+=1
            for b in bridges: b.physics_step(None)
            env.scene.update(1/240)
        reward=env.runtime.finish_batch(valid)
        extra=life.finish(240)
        extra['physical_substeps']=substeps
        truncated=(life.episode_seconds>=120)&valid
        extra['final_mask']=truncated.clone()
        extra['final_obs']={'critic':life.episode_seconds[:,None].expand(2,28).clone()}
        ids=truncated.nonzero().flatten().tolist()
        if ids: reset_rows(ids)
        return {},reward,torch.zeros(2,dtype=torch.bool),truncated,extra
    env.step=step
    def reset(**kw):
        reset_rows([0,1]);step(torch.zeros(2,9))
    env.reset=reset
    class Trace:
        def __init__(self,*args): self.count=0
        def begin(self): self.count=0
        def append(self): self.count+=1
        def finish(self):
            if self.count!=240: raise RuntimeError('trace incomplete')
            return torch.zeros(240,50)
    class Tape:
        def __init__(self,*args): self.count=0;self.active=False
        def begin(self): self.count=0;self.active=True
        def physics_step(self,*args):
            if self.active: self.count+=1
        def finish(self):
            self.active=False
            if self.count!=240: raise RuntimeError('magnetic incomplete')
            return torch.zeros(240,33),torch.zeros(240,25)
    import robotarm_magnetic_lab.runtime.new_stomach_rl_trace as trace_module
    import robotarm_magnetic_lab.runtime.new_stomach_rl_capacity_reference as tape_module
    monkeypatch.setattr(trace_module,'PhysicalStepRecorder',Trace)
    monkeypatch.setattr(tape_module,'MagneticInputTape',Tape)
    app=ModuleType('isaaclab.app')
    class Launcher:
        def __init__(self,args): self.app=SimpleNamespace(close=lambda:None)
        @staticmethod
        def add_app_launcher_args(parser): parser.add_argument('--device',default='cuda:0')
    @contextmanager
    def launch(*args): yield
    app.AppLauncher=Launcher;app.launch_simulation=launch
    monkeypatch.setitem(sys.modules,'isaaclab.app',app)
    gym=ModuleType('gymnasium');gym.make=lambda *args,**kw:SimpleNamespace(unwrapped=env)
    monkeypatch.setitem(sys.modules,'gymnasium',gym)
    import omni
    usd=ModuleType('omni.usd')
    usd.get_context=lambda:SimpleNamespace(get_stage=lambda:SimpleNamespace(
        GetPrimAtPath=lambda path:SimpleNamespace(GetAttribute=lambda name:SimpleNamespace(Get=lambda:True))))
    monkeypatch.setitem(sys.modules,'omni.usd',usd)
    monkeypatch.setattr(omni,'usd',usd,raising=False)
    task='robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.'
    env_module=ModuleType(task+'new_stomach_rl_vector_env');env_module.register_vector=lambda:'fake'
    cfg_module=ModuleType(task+'robotarm_magnetic_new_stomach_rl_vector_env_cfg')
    cfg_module.RobotarmMagneticNewStomachRLVectorCfg=lambda **kw:SimpleNamespace(scene=SimpleNamespace(),sim=SimpleNamespace())
    monkeypatch.setitem(sys.modules,env_module.__name__,env_module);monkeypatch.setitem(sys.modules,cfg_module.__name__,cfg_module)
    monkeypatch.setattr(sys,'argv',[str(scripts/'validate_vector_isolation.py'),'--pose_manifest',str(tmp_path/'poses'),
        '--mask',str(tmp_path/'mask'),'--tolerance_manifest',str(manifest),'--output_root',str(tmp_path/'output')])
    if substeps==239 or origin:
        with pytest.raises(SystemExit): script.main()
    else: script.main()
    summary=json.loads(next((tmp_path/'output').glob('*/summary.json')).read_text())
    assert closes==[1]
    if substeps==240 and not origin:
        assert summary['status']=='pass' and len(summary['runs'])==2
        assert len(steps)==244  # two initial HOLDs and 2 x 121 batches
        assert all(v['row_1_valid_transitions']==120 for v in summary['runs'])
        assert len(summary['comparisons'])==19
    elif substeps==239:
        assert summary['status']=='fail' and len(summary['runs'])==0 and len(steps)==2
    else:
        assert summary['status']=='fail' and len(steps)==21 and not summary['runs']
        assert any(not item['passed'] for item in summary['paired_prefix_20s'])
        assert list((tmp_path/'output').glob('*/failed_partial_tape.npz'))
