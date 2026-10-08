"""Fake capture/environment verifies hook lifecycle, never initializes Kit/GPU."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_magnetic_equivalence import make_reference
from robotarm_magnetic_lab.runtime.new_stomach_rl_reference_audit import audit_reference_pair
from robotarm_magnetic_lab.runtime.new_stomach_rl_rgb import PolicyRGBBoundary


@pytest.mark.parametrize('physical_calls,trace_error', [(240,False), (239,False), (240,True)])
def test_fake_capture_runs_two_repeats_or_stops_on_count_fault_and_restores_hooks(tmp_path, monkeypatch, physical_calls, trace_error):
    scripts = Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('capacity_profile_fake_test', scripts/'profile_capacity.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    pair = [make_reference(tmp_path/'single','single'), make_reference(tmp_path/'chunk','chunk')]
    registration = tmp_path/'registration.json'
    registration.write_text(json.dumps(audit_reference_pair(*pair)))
    noop = lambda *a, **kw: None
    monkeypatch.setattr(PolicyRGBBoundary,'sample',noop)
    synchronizations = []
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device: synchronizations.append(device))
    if trace_error:
        class FakeTrace:
            def __enter__(self):
                return self
            def __exit__(self,*args):
                return False
            def export_chrome_trace(self,path):
                raise OSError('fake diagnostic export failure')
        monkeypatch.setattr(torch.profiler,'profile',lambda **kwargs:FakeTrace())
    bridge = SimpleNamespace(model=SimpleNamespace(force_torque_on_cube_si=noop,force_torque_si=noop),
                             _update_field_visualization=noop,_log=noop)
    def magnetic(*a,**kw):
        bridge.model.force_torque_on_cube_si()
        bridge.model.force_torque_si()
    bridge.physics_step = magnetic
    term = SimpleNamespace(process_actions=noop,clearance=noop,apply_actions=noop)
    runtime = SimpleNamespace(coverage=SimpleNamespace(update_geometry_at_physics_tick=noop),
                              reward=SimpleNamespace(update_reward_10hz=noop),_encode_visual=noop,
                              optics=SimpleNamespace(forward_physics=noop),
                              visibility=SimpleNamespace(raycaster=SimpleNamespace(query=noop)),
                              policy_second=0,rgb=torch.zeros((1,2,2,3),dtype=torch.uint8),
                              rgb_frames=torch.tensor([1]))
    class Environment:
        device = 'cuda:0'
        num_envs = 1
        def reset(self, **kwargs):
            runtime.policy_second = 0
            runtime.rgb_frames[0] = 1
            return None
        def step(self, action):
            term.process_actions(action)
            for _ in range(physical_calls):
                term.apply_actions()
                bridge.physics_step()
                self.sim.step()
                self.scene.update()
            for _ in range(10):
                runtime.coverage.update_geometry_at_physics_tick()
                runtime.reward.update_reward_10hz()
            PolicyRGBBoundary.sample(None)
            runtime._encode_visual()
            runtime.policy_second += 1
            runtime.rgb_frames[0] += 1
            return None
    env = Environment()
    env.runtime = runtime
    env.sim = SimpleNamespace(step=noop,render=noop,physics_manager=SimpleNamespace(forward=noop))
    env.scene = SimpleNamespace(update=noop)
    env.action_manager = SimpleNamespace(get_term=lambda name:term)
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name:SimpleNamespace(func=bridge))
    original_step, original_reset = env.step, env.reset
    gym = ModuleType('gymnasium')
    gym.make = lambda *a, **kw: SimpleNamespace(unwrapped=env)
    capture = ModuleType('capture_capacity_reference')
    capture.__file__ = str(scripts/'capture_capacity_reference.py')
    def persist(report, output, shutdown):
        (output/'summary.json').write_text(json.dumps(report))
    capture.persist_before_shutdown = persist
    def capture_main():
        output = tmp_path/'output'/'fake-worker'
        output.mkdir(parents=True)
        for source in pair[0].iterdir():
            if source.name == 'summary.json':
                continue
            if source.is_dir():
                shutil.copytree(source,output/source.name)
            else:
                shutil.copyfile(source,output/source.name)
        made = gym.make('fake').unwrapped
        report = dict(status='pass')
        try:
            for _ in range(2):
                made.reset(seed=1008)
                for _ in range(20):
                    made.step(None)
        except Exception as error:
            report.update(status='fail',error=str(error))
        capture.persist_before_shutdown(report,output,noop)
    capture.main = capture_main
    monkeypatch.setitem(sys.modules,'gymnasium',gym)
    monkeypatch.setitem(sys.modules,'capture_capacity_reference',capture)
    app = ModuleType('isaaclab.app')
    class FakeLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--device',default='cuda:0')
    app.AppLauncher = FakeLauncher
    monkeypatch.setitem(sys.modules,'isaaclab.app',app)
    arguments = ['--mode','single','--pose_manifest','unused','--mask','unused',
                '--tolerance_manifest',str(registration),'--output_root',str(tmp_path/'output'),
                '--timing','--save_rgb_pixels']
    script.run(arguments + (['--trace'] if trace_error else []))
    assert env.step == original_step and env.reset == original_reset
    assert bridge.physics_step == magnetic
    assert capture.persist_before_shutdown is persist
    assert term.apply_actions is noop
    report = json.loads((tmp_path/'output/fake-worker/summary.json').read_text())
    if trace_error:
        assert report['status'] == 'fail'
        assert report['error']['message'] == 'fake diagnostic export failure'
        assert report['profile']['valid_transitions'] == 40
    elif physical_calls == 240:
        assert report['status'] == 'pass'
        assert report['profile']['valid_transitions'] == 40
        assert len(synchronizations) == 84  # 42 outer boundaries, two syncs each
        assert len(report['paired_reference_comparison']) == 2
        assert (tmp_path/'output/fake-worker/repeat_1/rgb_pixels.npz').is_file()
    else:
        assert report['status'] == 'fail'
        assert report['profile']['failed_boundaries'] == 1
        assert report['profile']['valid_transitions'] == 0
        assert runtime.policy_second == 1  # no silent next action/reset after fault


def test_forwarding_removes_only_diagnostic_flags(monkeypatch):
    scripts = Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('capacity_forward_test',scripts/'profile_capacity.py')
    script = importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
    assert script.forwarded_arguments(['--timing','--mode','single','--tolerance_manifest=x',
        '--save_rgb_pixels','--kit_args=--/UJITSO/enabled=false','--trace']) == [
            '--mode','single','--kit_args=--/UJITSO/enabled=false']
