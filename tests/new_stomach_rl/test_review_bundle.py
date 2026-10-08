import hashlib
import json
from pathlib import Path

import pytest

from robotarm_magnetic_lab.runtime.new_stomach_rl_review_bundle import (
    AUDITED_HEAD, export_bundle, verify_bundle, audit_capacity_bundle, check_reference_sources,
)


def fixture(tmp_path):
    root = tmp_path/'old/artifacts/new_stomach_rl'
    paths = ['preflight/run/summary.json', 'benchmark/run/envs_1/summary.json',
             'benchmark/run/envs_2/summary.json', 'benchmark/run/envs_1/step_timings.csv',
             'evidence/preflight_regression_final.log', 'evidence/final_executed_commands.json']
    files = []
    for name in paths:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original evidence '+name)
        if name.endswith('envs_1/summary.json'):
            path.write_text(json.dumps(dict(status='pass',num_envs=1,benchmark_steps=208,
                env_steps_per_s=.5,transitions_per_s=.5,timer_means={'step_wall_s':2.},
                committed_fraction=1.,projected_fraction=0.,hold_fraction=0.)))
        if name.endswith('step_timings.csv'):
            blocks=[f'warmup_{r}' for r in range(2) for _ in range(8)]
            blocks += [f'repeat_{r}' for r in range(2) for _ in range(32)]
            blocks += [f'rollout_{r}' for r in range(2) for _ in range(64)]
            path.write_text('block,step_wall_s,projection_scale\n'+
                ''.join(f'{block},2.0,1.0\n' for block in blocks))
        files.append(dict(path=str(path), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    large = root/'physics_240hz.jsonl'
    large.write_text('must not copy physics')
    files.append(dict(path=str(large), bytes=large.stat().st_size,
                      sha256=hashlib.sha256(large.read_bytes()).hexdigest()))
    inv = tmp_path/'inventory.json'
    inv.write_text(json.dumps(dict(audited_head=AUDITED_HEAD, files=files,
        external_dependencies=[dict(path='/external/config', bytes=12, sha256='identity')],
        executed_source_snapshots=[])))
    return inv, files


def test_identity_roundtrip_portable_bundle_excludes_large_data(tmp_path):
    inventory, files = fixture(tmp_path)
    output = tmp_path/'bundle'
    result = export_bundle(inventory, output)
    assert result['status'] == 'complete'
    assert len(result['artifacts']) == 6
    for item in result['artifacts']:
        assert (output/item['bundle_path']).read_bytes() == Path(item['source_path']).read_bytes()
    assert not list(output.rglob('physics_240hz.jsonl'))
    # Windows verification must not need the Linux original absolute files.
    for item in files:
        Path(item['path']).unlink()
    assert verify_bundle(output)['audited_head'] == AUDITED_HEAD
    audit=audit_capacity_bundle(output)
    assert audit['measured_steps']==64 and audit['all_logged_steps']==208
    assert audit['steady_transitions_per_s']==.5
    assert audit['q_valid_including_reset']=='not_measured_in_previous_contract'


@pytest.mark.parametrize('fault', ['missing', 'bytes', 'sha', 'head'])
def test_reject_corrupt_source_before_creating_bundle(tmp_path, fault):
    inventory, files = fixture(tmp_path)
    data = json.loads(inventory.read_text())
    if fault == 'missing':
        Path(files[0]['path']).unlink()
    elif fault in ('bytes', 'sha'):
        data['files'][0]['bytes' if fault == 'bytes' else 'sha256'] = 9 if fault == 'bytes' else 'bad'
    else:
        data['audited_head'] = '0'*40
    inventory.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        export_bundle(inventory, tmp_path/'bundle')
    assert not (tmp_path/'bundle').exists()


def test_wrong_inventory_hash_and_existing_output_rejected(tmp_path):
    inventory, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='inventory SHA'):
        export_bundle(inventory, tmp_path/'bundle', expected_inventory_sha256='bad')
    export_bundle(inventory, tmp_path/'bundle')
    with pytest.raises(FileExistsError):
        export_bundle(inventory, tmp_path/'bundle')


def test_portable_verifier_rejects_tamper_and_manifest_traversal(tmp_path):
    inventory, _ = fixture(tmp_path)
    export_bundle(inventory, tmp_path/'bundle')
    path = tmp_path/'bundle/bundle_manifest.json'
    data = json.loads(path.read_text())
    item = data['artifacts'][0]
    (tmp_path/'bundle'/item['bundle_path']).write_text('tampered')
    with pytest.raises(ValueError, match='identity'):
        verify_bundle(tmp_path/'bundle')
    item['bundle_path'] = '../inventory.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='bundle path'):
        verify_bundle(tmp_path/'bundle')


def test_reference_checks_live_repo_sources_not_old_absolute_worktree(tmp_path):
    inventory,_=fixture(tmp_path)
    data=json.loads(inventory.read_text())
    live=tmp_path/'new_repo/source/robotarm/core.py'
    live.parent.mkdir(parents=True);live.write_text('unchanged original code')
    data['executed_source_snapshots']=[dict(path='/old/worktree/source/robotarm/core.py',
        bytes=live.stat().st_size,sha256=hashlib.sha256(live.read_bytes()).hexdigest())]
    inventory.write_text(json.dumps(data))
    export_bundle(inventory,tmp_path/'bundle')
    assert check_reference_sources(tmp_path/'new_repo',tmp_path/'bundle')==['source/robotarm/core.py']
    live.write_text('modified')
    with pytest.raises(ValueError,match='identity'):
        check_reference_sources(tmp_path/'new_repo',tmp_path/'bundle')
