"""P0 portable, bounded evidence export. Standard-library only; no simulation."""
import hashlib
import json
import csv
import math
from pathlib import Path, PurePosixPath
import shutil

AUDITED_HEAD = '69f393cb95268eb17986f5ee0a2049386d6745aa'
INVENTORY_SHA = '6ce46685d8b92950ae0db7a54b741823bdb168d0534d15fb31efefd6e4f98c2f'
MAX_BYTES = 10*1024*1024


def identity(path):
    path = Path(path)
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def check(path, item):
    if identity(path) != {key:item[key] for key in ('bytes','sha256')}:
        raise ValueError(f'artifact identity mismatch: {path}')


def selected(item):
    path = PurePosixPath(item['path'])
    return (path.name == 'summary.json' or path.name == 'step_timings.csv'
        or path.name in ('preflight_regression_final.log','final_executed_commands.json'))


def destination(item):
    # Original metadata is Linux POSIX, even when verifier runs on Windows.
    parts = PurePosixPath(item['path']).parts
    if not PurePosixPath(item['path']).is_absolute():
        raise ValueError('source artifact requires original absolute path')
    try:
        start = next(i for i in range(len(parts)-1)
            if parts[i:i+2] == ('artifacts','new_stomach_rl'))
    except StopIteration as error:
        raise ValueError('source artifact outside preflight evidence tree') from error
    result = PurePosixPath('artifacts', *parts[start+2:])
    if '..' in result.parts:
        raise ValueError('invalid bundle path')
    return str(result)


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative:
        raise ValueError('invalid bundle path')
    result = root.joinpath(*name.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('invalid bundle path: symlink escapes root')
    return result


def export_bundle(source_inventory, output_dir, expected_inventory_sha256=None):
    source_inventory, output = Path(source_inventory), Path(output_dir)
    original = identity(source_inventory)
    if expected_inventory_sha256 is not None and original['sha256'] != expected_inventory_sha256:
        raise ValueError('inventory SHA differs from frozen report')
    inventory = json.loads(source_inventory.read_text())
    if inventory['audited_head'] != AUDITED_HEAD:
        raise ValueError('wrong audited implementation HEAD')
    items = [item for item in inventory['files'] if selected(item)]
    if not items or sum(item['bytes'] for item in items)+original['bytes'] > MAX_BYTES:
        raise ValueError('empty or oversized review bundle')
    destinations = [destination(item) for item in items]
    if len(set(destinations)) != len(destinations):
        raise ValueError('duplicate bundle destinations')
    for item in items:
        check(item['path'], item)
    # Never overwrite an existing export or mutate source evidence.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_inventory, output/'source_inventory.json')
    manifest = dict(status='complete', audited_head=AUDITED_HEAD,
        source_inventory=dict(source_path=str(source_inventory.resolve()),
            bundle_path='source_inventory.json', **original), artifacts=[],
        external_dependencies=inventory.get('external_dependencies',[]),
        source_code_snapshots=inventory.get('executed_source_snapshots',[]),
        omitted='Large physical trajectories/checkpoints remain at original paths in source_inventory.json')
    for item, relative in zip(items, destinations):
        target = safe_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item['path'], target)
        check(target, item)
        manifest['artifacts'].append(dict(source_path=item['path'], bundle_path=relative,
            bytes=item['bytes'], sha256=item['sha256']))
    (output/'bundle_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    verify_bundle(output, expected_inventory_sha256)
    return manifest


def verify_bundle(output_dir, expected_inventory_sha256=None):
    root = Path(output_dir)
    manifest = json.loads((root/'bundle_manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest['audited_head'] != AUDITED_HEAD:
        raise ValueError('incomplete bundle or wrong audited implementation HEAD')
    source = manifest['source_inventory']
    inventory_path = safe_path(root, source['bundle_path'])
    check(inventory_path, source)
    if expected_inventory_sha256 is not None and identity(inventory_path)['sha256'] != expected_inventory_sha256:
        raise ValueError('inventory SHA differs from frozen report')
    inventory = json.loads(inventory_path.read_text())
    if inventory['audited_head'] != AUDITED_HEAD:
        raise ValueError('wrong audited implementation HEAD')
    expected = {item['path']:item for item in inventory['files'] if selected(item)}
    if len(expected) != len(manifest['artifacts']):
        raise ValueError('incomplete bundle inventory')
    for item in manifest['artifacts']:
        target = safe_path(root, item['bundle_path'])
        original = expected.pop(item['source_path'], None)
        if original is None or destination(original) != item['bundle_path']:
            raise ValueError('unexpected bundle artifact')
        check(target, original)
        check(target, item)
    if (manifest['external_dependencies'] != inventory.get('external_dependencies', [])
        or manifest['source_code_snapshots'] != inventory.get('executed_source_snapshots', [])):
        raise ValueError('external dependency/source identity mismatch')
    return manifest


def audit_capacity_bundle(output_dir):
    """Recompute the OLD D single-env figures directly from delivered CSV."""
    root=Path(output_dir)
    manifest=verify_bundle(root)
    candidates=[item for item in manifest['artifacts']
        if item['bundle_path'].endswith('/envs_1/step_timings.csv')]
    if len(candidates)!=1:
        raise ValueError('one completed single-environment timing CSV required')
    path=safe_path(root,candidates[0]['bundle_path'])
    summary=json.loads((path.parent/'summary.json').read_text())
    with path.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    measured=[row for row in rows if row['block'].startswith('repeat')]
    if len(rows)!=208 or len(measured)!=64 or summary['status']!='pass' or summary['num_envs']!=1:
        raise ValueError('old capacity measurement shape/status differs')
    elapsed=sum(float(row['step_wall_s']) for row in measured)
    if not math.isfinite(elapsed) or elapsed<=0:
        raise ValueError('invalid measured wall time')
    rate=len(measured)/elapsed
    for key in ('env_steps_per_s','transitions_per_s'):
        if not math.isclose(rate,summary[key],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f'CSV/summary disagreement: {key}')
    for key,value in summary['timer_means'].items():
        recomputed=sum(float(row.get(key,0.)) for row in measured)/len(measured)
        if not math.isclose(recomputed,value,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f'CSV/summary timer disagreement: {key}')
    scales=[float(row['projection_scale']) for row in rows]
    fractions=dict(committed_fraction=sum(x>0 for x in scales)/len(scales),
        projected_fraction=sum(0<x<1 for x in scales)/len(scales),
        hold_fraction=sum(x==0 for x in scales)/len(scales))
    if any(not math.isclose(value,summary[key],rel_tol=1e-12) for key,value in fractions.items()):
        raise ValueError('CSV/summary projection disagreement')
    return dict(status='pass',source_csv=str(path),all_logged_steps=len(rows),measured_steps=len(measured),
        measured_wall_s=elapsed,mean_step_wall_s=elapsed/len(measured),steady_transitions_per_s=rate,
        equivalent_768000_sampling_hours=768000/rate/3600,**fractions,
        q_valid_including_reset='not_measured_in_previous_contract',
        note='Original D steady-state measurement only; initialization/reset excluded; nested timers not additive')


def check_reference_sources(repo_root, bundle_dir):
    """Reject a baseline collected with modified original preflight sources."""
    manifest=verify_bundle(bundle_dir)
    snapshots=manifest['source_code_snapshots']
    if not snapshots:
        raise ValueError('original executed source snapshots required')
    checked=[]
    for item in snapshots:
        parts=PurePosixPath(item['path']).parts
        anchors=[i for i,name in enumerate(parts) if name in ('scripts','source','tests')]
        if len(anchors)!=1:
            raise ValueError('ambiguous original source relative path')
        relative=PurePosixPath(*parts[anchors[0]:])
        path=safe_path(Path(repo_root),str(relative))
        check(path,item)
        checked.append(str(relative))
    return checked
