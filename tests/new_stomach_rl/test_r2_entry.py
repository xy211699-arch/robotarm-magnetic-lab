import importlib.util
import json
from pathlib import Path
import pytest


ROOT=Path(__file__).resolve().parents[2]


def supervisor():
    spec=importlib.util.spec_from_file_location('_r2_supervisor',ROOT/'scripts/new_stomach_rl/supervise_training.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def config(tmp_path):
    m=supervisor();evidence=tmp_path/'evidence';evidence.write_text('fixture')
    import hashlib
    return dict(purpose='r2_smoke',group='D',num_envs=8,max_updates=3,rollout_steps=64,
        max_wall_seconds=7200,startup_timeout_seconds=1909,heartbeat_timeout_seconds=181,stop_grace_seconds=20,
        evidence_files=[dict(path=str(evidence),sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())],
        fixed_replay_seconds=120,partial_reset_boundary=5,restore_before_update=3,
        gamma=.999**10,gae_lambda=.95,epochs=2,sequence_length=64,
        minibatch='whole_valid_sequence',formal_training_allowed=False,device='cuda:0',visualizer='none')


@pytest.mark.parametrize('field,value',[('num_envs',12),('max_updates',1000),('fixed_replay_seconds',20),
    ('partial_reset_boundary',17),('epochs',4),('formal_training_allowed',True),('device','cpu'),('max_wall_seconds',14400)])
def test_entry_cannot_silently_expand_r2_or_reduce_acceptance(tmp_path,field,value):
    m=supervisor();cfg=config(tmp_path);p=tmp_path/'config.json';p.write_text(json.dumps(cfg))
    assert m.validate_config(p)['num_envs']==8
    cfg[field]=value;p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):m.validate_config(p)


def test_gpu_entry_modules_have_no_import_time_launch():
    import ast
    for name in ('train.py','prepare_r2.py'):
        path=ROOT/'scripts/new_stomach_rl'/name;tree=ast.parse(path.read_text())
        # GPU packages are imported only in main, not while CPU tests load code.
        assert not any(isinstance(x,ast.ImportFrom) and x.module and x.module.startswith('isaaclab') for x in tree.body)
        compile(path.read_text(),str(path),'exec')
