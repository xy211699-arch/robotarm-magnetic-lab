"""P0 reference verification and preregistration; no GPU or model optimization."""
import hashlib
import json

import numpy as np
import pytest

from robotarm_magnetic_lab.runtime.new_stomach_rl_reference_audit import audit_reference_pair


def make_reference(root, mode):
    root.mkdir()
    dim = 9 if mode == 'single' else 36
    raw = np.zeros((20, dim)).tolist()
    (root/'raw_actions.json').write_text(json.dumps(dict(mode=mode, pose_id='train-0003', seed=1008, actions=raw)))
    ticks = [0, 0]
    branches = [10, 1]
    for tick in range(24, 4801, 24):
        ticks.append(tick); branches.append(10)
        if tick % 240 == 0:
            ticks.append(tick); branches.append(1)
    arrays = dict(physics=np.zeros((20, 240, 50), np.float32),
                  magnetic_inputs=np.zeros((20, 240, 33), np.float32),
                  magnetic_outputs=np.zeros((20, 240, 25), np.float32),
                  branch_hz=np.array(branches), physics_tick=np.array(ticks),
                  visible_packed=np.ones((222, 1), np.uint8), cumulative_packed=np.ones((222, 1), np.uint8),
                  num_vertices=np.array(8), bitorder=np.array('little'))
    for repeat in range(2):
        folder = root/f'repeat_{repeat}'
        folder.mkdir()
        np.savez_compressed(folder/'reference_tape.npz', **arrays)
        rows = [dict(physics_tick=tick, sample_id=i, sim_time_s=i/10, coverage=[.1]) for i, tick in enumerate(range(0, 4801, 24))]
        (folder/'coverage_reward_10hz.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        rows = [dict(policy_second=i, rgb_frame=i+1, sim_time_s=float(i), coverage=[.1], rgb_sha256=str(repeat)*64) for i in range(21)]
        (folder/'rgb_1hz.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        rows = [dict(policy_second=i+1, raw_action=raw[i], actual_history=np.zeros((4, 9)).tolist(), reward=0.) for i in range(20)]
        (folder/'policy_1hz.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    refresh_manifest(root, mode)
    return root


def refresh_manifest(root, mode):
    files = [p for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'summary.json']
    data = dict(status='pass', stage='P0_reference', mode=mode, pose_id='train-0003',
                head='c1ef70a9c1593c182df9353a4086bc16ff6f1de3', assets={'frozen_fixture': 'unchanged'},
                devices=dict(environment='cuda:0', simulation='cuda:0', camera='cuda:0', gpu_dynamics=True),
                model_updates=0, optimization_started=False, physics_hz=240, policy_hz=1, coverage_hz=10,
                repeat_runs=2, repeats=[dict(repeat=i, policy_seconds=20, physics_steps=4800, magnetic_calls=4800,
                                          c10_points=201, c1_points=21, coverage_masks=222) for i in range(2)],
                artifacts=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (root/'summary.json').write_text(json.dumps(data))


@pytest.fixture
def pair(tmp_path):
    return make_reference(tmp_path/'single', 'single'), make_reference(tmp_path/'chunk', 'chunk')


def test_preregister_before_optimization_from_original_repeatability(pair):
    result = audit_reference_pair(*pair)
    assert result['status'] == 'pass'
    assert result['optimization_started'] is False
    assert result['tolerances']['magnetic_outputs'] == dict(atol=0., rtol=0., source='original repeatability')
    assert result['tolerances']['visible_and_cumulative_masks'] == 'exact'
    assert result['rgb_pixels']['status'] == 'not_measurable_from_hashes'
    assert result['references']['single']['rgb_equal_hashes'] == 0
    assert result['references']['chunk']['physics_steps_per_repeat'] == 4800


def test_tampered_or_missing_artifact_cannot_be_a_baseline(pair):
    file = pair[0]/'raw_actions.json'
    file.write_text(file.read_text()+' ')
    with pytest.raises(ValueError, match='identity'):
        audit_reference_pair(*pair)
    file.unlink()
    with pytest.raises(ValueError):
        audit_reference_pair(*pair)


def test_registry_rejects_nonfinite_data_even_with_updated_manifest(pair):
    file = pair[0]/'repeat_1/reference_tape.npz'
    arrays = dict(np.load(file, allow_pickle=False))
    arrays['magnetic_outputs'][0, 0, 0] = np.nan
    np.savez_compressed(file, **arrays)
    refresh_manifest(pair[0], 'single')
    with pytest.raises(ValueError, match='nonfinite'):
        audit_reference_pair(*pair)


def test_changed_mask_or_magnetic_repetition_is_recorded_not_silently_relaxed(pair):
    file = pair[0]/'repeat_1/reference_tape.npz'
    arrays = dict(np.load(file, allow_pickle=False))
    arrays['magnetic_outputs'][0, 0, 0] = .125
    arrays['magnetic_outputs'][0, 0, 12] = .125
    arrays['physics'][0, 0, 13] = .125
    np.savez_compressed(file, **arrays)
    refresh_manifest(pair[0], 'single')
    result = audit_reference_pair(*pair)
    assert result['tolerances']['magnetic_outputs']['atol'] == .125
    assert result['references']['single']['max_abs_differences']['magnetic_outputs'] == .125
    assert result['tolerances']['magnetic_outputs']['rtol'] == 0


def test_incomplete_frames_or_nonmonotonic_coverage_is_rejected(pair):
    file = pair[0]/'repeat_0/rgb_1hz.jsonl'
    rows = file.read_text().splitlines()
    file.write_text('\n'.join(rows[:-1])+'\n')
    refresh_manifest(pair[0], 'single')
    with pytest.raises(ValueError, match='21'):
        audit_reference_pair(*pair)


def test_gpu_device_and_mode_identity_are_actual_metadata_requirements(pair):
    file = pair[0]/'summary.json'
    data = json.loads(file.read_text())
    data['devices']['gpu_dynamics'] = False
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='GPU'):
        audit_reference_pair(*pair)
