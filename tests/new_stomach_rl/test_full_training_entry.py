import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from test_r2_entry import config,supervisor


def full_config(tmp_path):
    cfg=config(tmp_path)
    cfg.update(purpose='full_training',max_updates=1000,save_interval=50,max_wall_seconds=None,
        formal_training_allowed=True,partial_reset_boundary=None,restore_before_update=None,
        tensorboard=True,time_limited_probe=False,approval='user_confirmed_2026-10-10_1000x64_N8_save50')
    return cfg


@pytest.mark.parametrize('group',list('ABCD'))
def test_explicit_approved_full_config_is_1000_not_three(tmp_path,group):
    cfg=full_config(tmp_path);cfg['group']=group;p=tmp_path/'full.json';p.write_text(json.dumps(cfg))
    result=supervisor().validate_config(p)
    assert result['max_updates']==1000 and result['num_envs']==8 and result['rollout_steps']==64
    assert result['save_interval']==50 and result['max_wall_seconds'] is None
    assert result['partial_reset_boundary'] is None and result['restore_before_update'] is None


@pytest.mark.parametrize('field,value',[('max_updates',999),('save_interval',1),('num_envs',12),
    ('rollout_steps',32),('max_wall_seconds',7200),('epochs',4),('approval','not_approved')])
def test_full_cannot_change_frozen_user_parameters(tmp_path,field,value):
    cfg=full_config(tmp_path);cfg[field]=value;p=tmp_path/'full.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):supervisor().validate_config(p)


def test_1000_update_fake_worker_completes_without_gpu(tmp_path):
    m=supervisor();script=tmp_path/'fake.py'
    script.write_text('''import hashlib,json,pathlib,sys,time
sys.path.insert(0,sys.argv[2])
from supervise_training import checkpoint_due
p=pathlib.Path(sys.argv[1]);saved=[]
for update in range(1,1001):
 if checkpoint_due(update,50):
  ck=p/f'update_{update:04d}.pt';ck.write_bytes(b'CPU fixture, not a model')
  saved.append(update)
checkpoint=dict(path=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),kind='training_boundary')
(p/'schedule.json').write_text(json.dumps(saved))
(p/'summary.json').write_text(json.dumps(dict(run_id=p.name,status='completed',update_count=1000,
 effective_samples=512000,global_boundaries=64000,checkpoint_update=1000,checkpoint=checkpoint)))
''')
    cfg=full_config(tmp_path);cfg.update(startup_timeout_seconds=5,heartbeat_timeout_seconds=5,stop_grace_seconds=.1)
    result=m.monitor([sys.executable,str(script),str(tmp_path),str(Path(__file__).resolve().parents[2]/'scripts/new_stomach_rl')],tmp_path,cfg,poll=.01)
    assert result['status']=='completed' and result['update_count']==1000
    assert json.loads((tmp_path/'schedule.json').read_text())==list(range(50,1001,50))
    assert len(list(tmp_path.glob('update_*.pt')))==20


def test_training_loop_uses_config_and_resume_update_not_hardcoded_three():
    root=Path(__file__).resolve().parents[2]
    tree=ast.parse((root/'scripts/new_stomach_rl/train.py').read_text())
    loops=[ast.unparse(x.iter) for x in ast.walk(tree) if isinstance(x,ast.For)]
    assert "range(runner.update_count, config['max_updates'])" in loops
    assert 'range(3)' not in loops
    assert any("config['save_interval']" in ast.unparse(x) for x in ast.walk(tree) if isinstance(x,ast.If))
    compile((root/'scripts/new_stomach_rl/train.py').read_text(),str(root/'scripts/new_stomach_rl/train.py'),'exec')


def test_resume_launch_copies_boundary_checkpoint_and_keeps_original_budget(tmp_path,monkeypatch,capsys):
    from types import SimpleNamespace
    m=supervisor();m.ROOT=tmp_path;m.BASE=tmp_path/'jobs'
    target=tmp_path/'scripts/new_stomach_rl/train.py';target.parent.mkdir(parents=True);target.write_text('# CPU fixture only')
    parent=tmp_path/'parent';parent.mkdir()
    cfg=full_config(tmp_path);cfg['implementation_head']='a'*40
    frozen=parent/'frozen_config.json';frozen.write_text(json.dumps(cfg));original=frozen.read_bytes()
    ck=parent/'update_0150.pt';ck.write_bytes(b'not a real model; no worker launched')
    (parent/'status.json').write_text(json.dumps(dict(status='interrupted',update_count=175,
        latest_checkpoint=dict(path=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),kind='training_boundary'))))
    commands=[]
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:'a'*40+'\n')
    monkeypatch.setattr(m.subprocess,'Popen',lambda command,**kw:commands.append(command) or SimpleNamespace(pid=42))
    monkeypatch.setattr(sys,'argv',['supervise_training.py','resume','--run_dir',str(parent)])
    m.main();launch_result=json.loads(capsys.readouterr().out)
    folder=Path(launch_result['run_dir']);launch=json.loads((folder/'launch.json').read_text())
    assert launch['config']['max_updates']==1000 and launch['config']['save_interval']==50
    assert frozen.read_bytes()==original
    copied=Path(launch['resume']['path'])
    assert copied.is_relative_to(folder) and copied.read_bytes()==ck.read_bytes()
    assert launch['resume']['parent_run_dir']==str(parent)
    assert len(commands)==1 and 'worker' in commands[0]


@pytest.mark.parametrize('status,kind',[('running','training_boundary'),('completed','training_boundary'),
    ('interrupted','weights_only')])
def test_resume_rejects_active_completed_or_weights_only_before_launch(tmp_path,monkeypatch,status,kind):
    m=supervisor();m.ROOT=tmp_path;m.BASE=tmp_path/'jobs';parent=tmp_path/'parent';parent.mkdir()
    (parent/'frozen_config.json').write_text(json.dumps(full_config(tmp_path)))
    (parent/'status.json').write_text(json.dumps(dict(status=status,latest_checkpoint=dict(kind=kind))))
    monkeypatch.setattr(sys,'argv',['supervise_training.py','resume','--run_dir',str(parent)])
    with pytest.raises(SystemExit):m.main()
    assert not m.BASE.exists()
