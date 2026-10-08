"""Offline audit of user-captured originals; register tolerances before candidates.

RGB hashes identify frames, but cannot establish pixel-error tolerances. This
module never reads an optimized run and never changes original artifacts.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def identity(file):
    file = Path(file)
    return dict(path=str(file.resolve()), bytes=file.stat().st_size,
                sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def _nonfinite_json(value):
    raise ValueError(f'nonfinite JSON: {value}')


def read_json(file):
    return json.loads(Path(file).read_text(), parse_constant=_nonfinite_json)


def read_rows(file):
    return [json.loads(row, parse_constant=_nonfinite_json)
            for row in Path(file).read_text().splitlines() if row.strip()]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_reference(directory, mode):
    root = Path(directory).resolve()
    summary = read_json(root/'summary.json')
    _require(summary.get('status') == 'pass' and summary.get('stage') == 'P0_reference', 'P0 reference status must be pass')
    _require(summary.get('mode') == mode and summary.get('pose_id') == 'train-0003', 'fixed mode/pose identity mismatch')
    _require(summary.get('model_updates') == 0 and summary.get('optimization_started') is False, 'original reference only, no optimization/training')
    _require([summary.get(k) for k in ('physics_hz','policy_hz','coverage_hz','repeat_runs')] == [240,1,10,2], 'reference clocks/repeats mismatch')
    devices = summary['devices']
    _require(devices.get('gpu_dynamics') is True and all(str(devices.get(k)).startswith('cuda') for k in ('environment','simulation','camera')), 'actual GPU metadata required')
    _require(len(summary.get('repeats', [])) == 2, 'two completed repeats required')
    for i, row in enumerate(summary['repeats']):
        _require([row.get(k) for k in ('repeat','policy_seconds','physics_steps','magnetic_calls','c10_points','c1_points','coverage_masks')] == [i,20,4800,4800,201,21,222], 'reference repeat count mismatch')
    expected = {'raw_actions.json'} | {f'repeat_{i}/{name}' for i in range(2) for name in (
        'reference_tape.npz','policy_1hz.jsonl','coverage_reward_10hz.jsonl','rgb_1hz.jsonl')}
    registered = set()
    for row in summary['artifacts']:
        file = Path(row['path']).resolve()
        _require(file.is_relative_to(root), 'artifact identity escapes supplied original directory')
        relative = file.relative_to(root).as_posix()
        _require(relative not in registered and file.is_file(), 'artifact identity duplicate/missing')
        actual = identity(file)
        _require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], f'artifact identity mismatch: {relative}')
        registered.add(relative)
    _require(registered == expected, 'reference artifact set incomplete/unexpected')
    actions = read_json(root/'raw_actions.json')
    dim = 9 if mode == 'single' else 36
    raw = np.asarray(actions['actions'], float)
    _require(actions['mode'] == mode and actions['pose_id'] == 'train-0003' and actions['seed'] == 1008,
             'fixed action source mismatch')
    _require(raw.shape == (20,dim) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all(), 'raw action contract mismatch')
    tapes, policies, coverage, rgb = [], [], [], []
    for repeat in range(2):
        folder = root/f'repeat_{repeat}'
        with np.load(folder/'reference_tape.npz', allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
        for key, shape in [('physics',(20,240,50)),('magnetic_inputs',(20,240,33)),('magnetic_outputs',(20,240,25))]:
            _require(arrays[key].shape == shape and arrays[key].dtype == np.float32, f'{key} shape/dtype mismatch')
            _require(np.isfinite(arrays[key]).all(), f'nonfinite {key}')
        _require(np.array_equal(arrays['physics'][:,:,13:25], arrays['magnetic_outputs'][:,:,12:24]), 'physical/applied wrench mismatch')
        vertices = int(arrays['num_vertices'])
        _require(vertices > 0 and str(arrays['bitorder']) == 'little', 'coverage mask metadata invalid')
        for key in ('visible_packed','cumulative_packed'):
            _require(arrays[key].shape == (222,(vertices+7)//8) and arrays[key].dtype == np.uint8, 'coverage mask shape/dtype mismatch')
        _require(arrays['branch_hz'].shape == arrays['physics_tick'].shape == (222,), '222 masks required')
        for branch, ticks in ((10,np.arange(0,4801,24)),(1,np.arange(0,4801,240))):
            rows = arrays['branch_hz'] == branch
            _require(np.array_equal(arrays['physics_tick'][rows], ticks), 'physical mask ticks mismatch')
            visible, cumulative = arrays['visible_packed'][rows], arrays['cumulative_packed'][rows]
            previous = np.zeros((vertices+7)//8, np.uint8)
            for now, accumulated in zip(visible, cumulative):
                _require(np.array_equal(previous | now, accumulated), 'cumulative coverage must be exact visible union')
                previous = accumulated
        _require(np.isin(arrays['branch_hz'],[1,10]).all(), 'unknown coverage branch')
        c10 = read_rows(folder/'coverage_reward_10hz.jsonl')
        c1 = read_rows(folder/'rgb_1hz.jsonl')
        policy = read_rows(folder/'policy_1hz.jsonl')
        _require(len(c10) == 201 and len(c1) == 21 and len(policy) == 20, '201 C10 / 21 RGB / 20 actions required')
        _require([r['physics_tick'] for r in c10] == list(range(0,4801,24)), 'C10 clock mismatch')
        _require(np.allclose([r['sim_time_s'] for r in c10], np.arange(201)/10, rtol=0, atol=1e-12), 'C10 time mismatch')
        _require([r['policy_second'] for r in c1] == list(range(21)) and [r['rgb_frame'] for r in c1] == list(range(1,22)), 'new RGB frame required each boundary')
        _require([r['sim_time_s'] for r in c1] == list(range(21)), 'RGB time mismatch')
        _require([r['policy_second'] for r in policy] == list(range(1,21)), 'policy clock mismatch')
        _require(np.array_equal([r['raw_action'] for r in policy], raw), 'recorded raw actions changed')
        _require(np.asarray([r['actual_history'] for r in policy]).shape == (20,4,9), 'issued history shape mismatch')
        for rows in (c10,c1):
            values = np.asarray([r['coverage'][0] for r in rows])
            _require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all() and (np.diff(values) >= 0).all(), 'nonmonotonic/nonfinite coverage')
        tapes.append(arrays); policies.append(policy); coverage.append(c10); rgb.append(c1)
    differences = {k:float(np.max(np.abs(tapes[0][k].astype(float)-tapes[1][k].astype(float))))
                   for k in ('physics','magnetic_inputs','magnetic_outputs')}
    for key in ('visible_packed','cumulative_packed','branch_hz','physics_tick','num_vertices','bitorder'):
        _require(np.array_equal(tapes[0][key],tapes[1][key]), 'original coverage masks/clock differ; exact gate cannot be frozen')
    _require(coverage[0] == coverage[1], 'original C10/reward repeatability mismatch')
    for key in ('actual_history','raw_action','reward'):
        _require([r[key] for r in policies[0]] == [r[key] for r in policies[1]], f'original {key} repeatability mismatch')
    for key in ('coverage','policy_second','rgb_frame','sim_time_s'):
        _require([r[key] for r in rgb[0]] == [r[key] for r in rgb[1]], f'original C1 {key} repeatability mismatch')
    return dict(mode=mode, head=summary['head'], summary=identity(root/'summary.json'),
                source_artifacts=summary['artifacts'], physics_steps_per_repeat=4800,
                max_abs_differences=differences, coverage_masks_exact=True,
                c10_initial=coverage[0][0]['coverage'][0], c10_final=coverage[0][-1]['coverage'][0],
                c1_final=rgb[0][-1]['coverage'][0],
                rgb_equal_hashes=sum(a['rgb_sha256'] == b['rgb_sha256'] for a,b in zip(*rgb)),
                assets=summary['assets'])


def audit_reference_pair(single_directory, chunk_directory):
    refs = {mode:audit_reference(directory,mode) for mode,directory in (
        ('single',single_directory),('chunk',chunk_directory))}
    _require(refs['single']['assets'] == refs['chunk']['assets'] and refs['single']['head'] == refs['chunk']['head'], 'frozen assets/head differ across modes')
    tolerances = {key:dict(atol=max(r['max_abs_differences'][key] for r in refs.values()),
                           rtol=0., source='original repeatability')
                  for key in ('physics','magnetic_inputs','magnetic_outputs')}
    tolerances.update(visible_and_cumulative_masks='exact', issued_history='exact', c10_reward='exact',
                      rgb_frame_sequence='exact', rgb_pixels=None)
    return dict(status='pass', stage='P1_original_preregistration', optimization_started=False,
                references=refs, tolerances=tolerances,
                rgb_pixels=dict(status='not_measurable_from_hashes',
                                reason='Originals store image hashes, not pixels; do not infer a pixel tolerance or RGB equivalence'))
