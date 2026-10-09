"""Exercise the real new step/reset methods with fake physics, never Kit."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import pytest
import torch
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_lifecycle import VectorLifecycle


def load_env(monkeypatch):
    sdk = ModuleType('isaaclab.envs')
    class Base:
        def _reset_idx(self,ids):
            self.episode_length_buf[ids] = 0
            self.base_resets.append(ids.tolist())
    sdk.ManagerBasedRLEnv = Base
    monkeypatch.setitem(sys.modules,'isaaclab.envs',sdk)
    path = Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/new_stomach_rl_vector_env.py'
    spec = importlib.util.spec_from_file_location('_p2_fake_env',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.NewStomachRLVectorEnv


def fake_env(monkeypatch):
    cls = load_env(monkeypatch)
    env = cls.__new__(cls)
    env.num_envs,env.device,env.physics_dt = 2,'cpu',1/240
    env.lifecycle = VectorLifecycle(2,'cpu')
    env._sim_step_counter,env.common_step_counter = 0,0
    env.cfg = SimpleNamespace(sim=SimpleNamespace(render_interval=240))
    env.render_enabled = False
    env.episode_length_buf = torch.zeros(2,dtype=torch.int64)
    env.base_resets,env.force_calls,env.commands,env.updates,env.sim_calls = [],[],[],[],[]
    env.rows = [SimpleNamespace(env_origin=[-2.,0,0]),SimpleNamespace(env_origin=[2.,0,0])]
    term = SimpleNamespace(rows=env.rows,reset=lambda ids:None)
    env.action_manager = SimpleNamespace(total_action_dim=9,
        get_term=lambda name:term,process_action=lambda action:env.commands.append(action.clone()),
        apply_action=lambda:env.force_calls.append(1))
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name:SimpleNamespace(func=SimpleNamespace(reset=lambda ids:None)))
    class Scene(dict):
        def write_data_to_sim(self): pass
        def update(self,dt): env.updates.append(dt)
    env.scene = Scene(capsule=SimpleNamespace(write_root_pose_to_sim_index=lambda **kw:None,
        write_root_velocity_to_sim_index=lambda **kw:None))
    env.sim = SimpleNamespace(is_rendering=False,step=lambda **kw:env.sim_calls.append(1),forward=lambda:None)
    env.recorder_manager = SimpleNamespace(record_pre_step=lambda:None,record_post_physics_decimation_step=lambda:None)
    env.reset_pose_ids = ['train-1','train-2']
    pose = [1.,.1,.05,0,0,0,1.]
    env.libraries = [SimpleNamespace(sample=lambda:('train-1',pose)),SimpleNamespace(sample=lambda:('train-2',pose))]
    env.runtime = SimpleNamespace(reset_rows=lambda ids:None,
        finish_batch=lambda valid:valid.float(),
        observations=lambda:{'policy':env.lifecycle.episode_seconds[:,None].float().expand(2,548).clone(),
                             'critic':env.lifecycle.episode_seconds[:,None].float().expand(2,28).clone()})
    env.termination_manager = SimpleNamespace(terminated=torch.zeros(2,dtype=torch.bool),time_outs=torch.zeros(2,dtype=torch.bool))
    def compute():
        env.termination_manager.time_outs = env.episode_length_buf >= 120
        return env.termination_manager.time_outs
    env.termination_manager.compute = compute
    return env


def test_actual_step_partial_reset_uses_one_batch_and_keeps_active_row_budget(monkeypatch):
    env = fake_env(monkeypatch)
    env.step(torch.ones(2,9))
    env.step(torch.ones(2,9))
    before = len(env.sim_calls)
    env.reset_rows([0])
    assert len(env.sim_calls) == before
    observation,reward,terminated,truncated,extra = env.step(torch.ones(2,9))
    assert len(env.sim_calls)-before == 240
    assert env.lifecycle.episode_seconds.tolist() == [0,2]
    assert extra['valid_transition'].tolist() == [False,True]
    assert extra['episode_start'].tolist() == [True,False]
    assert reward.tolist() == [0,1]
    assert env.commands[-1][0].count_nonzero() == 0
    assert env.commands[-1][1].count_nonzero() == 9
    assert len(env.force_calls) == len(env.sim_calls)


def test_timeout_final_obs_not_replaced_by_reset_or_new_c0(monkeypatch):
    env = fake_env(monkeypatch);env.step(torch.zeros(2,9))
    env.lifecycle.episode_seconds[:] = torch.tensor([119,3])
    observation,_,_,truncated,extra = env.step(torch.zeros(2,9))
    assert truncated.tolist() == [True,False]
    assert extra['final_mask'].tolist() == [True,False]
    assert extra['final_obs']['critic'][0,0] == 120
    assert observation['critic'][0,0] == 0 and observation['critic'][1,0] == 4
    assert env.lifecycle.global_tick == 480 and env.base_resets == [[0]]


def test_safety_failure_not_autoreset_or_counted_success(monkeypatch):
    env = fake_env(monkeypatch);env.step(torch.zeros(2,9))
    env.termination_manager.terminated[1] = True
    with pytest.raises(RuntimeError,match='safety termination'):
        env.step(torch.zeros(2,9))
    assert env.base_resets == []


def test_reset_in_shared_batch_cannot_change_pose(monkeypatch):
    env = fake_env(monkeypatch);env.lifecycle.begin(torch.zeros(2,9))
    with pytest.raises(RuntimeError,match='inside physics'):
        env.reset_rows([0])
    assert not env.base_resets
