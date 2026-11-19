#!/usr/bin/env python3
"""Offline evidence audit; never starts Isaac, changes source files, or approves labels.
Requires Python 3.10+, numpy and Pillow. See delivery README for interpretation.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

COMPONENTS = ('image.png', 'sample.json', 'depth_m.npy', 'instance_ids.npy',
              'scene_snapshot.usdc', 'capture_script.py', 'config.json',
              'replay_config.json', 'SHA256SUMS.json')
MAPPING = {'normal': 0, 'raised': 1, 'ulcerated': 2}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ['status'])
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dhash(image):
    values = np.asarray(image.convert('L').resize((17, 16)))
    value = 0
    for bit in (values[:, 1:] > values[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return '%064x' % value


def pose_difference(a, b):
    p = np.array(a['camera_position_world_m'], dtype=float)
    q = np.array(a['camera_orientation_world'], dtype=float)
    r = np.array(b['camera_position_world_m'], dtype=float)
    s = np.array(b['camera_orientation_world'], dtype=float)
    dot = min(1., abs(float(np.dot(q, s))) / (np.linalg.norm(q) * np.linalg.norm(s)))
    return float(np.linalg.norm(p-r)*1000), math.degrees(2*math.acos(dot))


def valid_pose(sample):
    p, q = sample.get('camera_position_world_m'), sample.get('camera_orientation_world')
    return (isinstance(p, list) and len(p) == 3 and all(math.isfinite(x) for x in p)
            and isinstance(q, list) and len(q) == 4 and all(math.isfinite(x) for x in q)
            and abs(math.sqrt(sum(x*x for x in q))-1) < 1e-6
            and sample.get('quaternion_order') == 'wxyz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if source == out or source in out.parents:
        parser.error('output must be outside source to preserve evidence')
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    files = sorted(p for p in source.rglob('*') if p.is_file())
    stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    hashes, inventory, image_checks, image_data = {}, [], [], {}
    source_parse_issues = []
    print('Hashing', len(files), 'source files', flush=True)
    for index, path in enumerate(files):
        rel = path.relative_to(source).as_posix()
        sha = digest(path)
        hashes[str(path)] = sha
        inventory.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha})
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            check = {'path': rel, 'bytes': path.stat().st_size, 'sha256': sha, 'issues': []}
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    rgb = im.convert('RGB')
                    check.update(width=im.width, height=im.height, mode=im.mode,
                                 dhash256=dhash(rgb),
                                 pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
            except Exception as exc:
                check['issues'].append(str(exc))
            image_checks.append(check)
            image_data[str(path)] = check
        if path.suffix == '.py':
            try:
                ast.parse(path.read_bytes(), filename=rel)
            except Exception as exc:
                source_parse_issues.append({'path': rel, 'error': str(exc)})
        if (index+1) % 500 == 0:
            print('Hashed', index+1, '/', len(files), flush=True)
    write_csv(out/'all_file_index.csv', inventory)
    with (out/'SHA256SUMS.source').open('w', encoding='utf-8') as stream:
        for item in inventory:
            stream.write(item['sha256'] + '  ' + item['path'] + '\n')
    write_json(out/'image_decode_checks.json', image_checks)

    orbit = source/'orbit_v6'
    classification, batches, config_by_batch = [], [], {}
    for config_path in sorted(orbit.glob('collect*/collection_config.json')):
        batch, config = config_path.parent, read_json(config_path)
        config_by_batch[batch.name] = config
        manifest = batch/'classification_manifest.jsonl'
        rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()] if manifest.exists() else []
        complete = len(rows) == config.get('target_count')
        role = ('replacement_candidate' if '20261118T17' in batch.name and config['class_name'] != 'normal'
                else 'complete_candidate' if complete else 'incomplete')
        if not complete:
            role = 'incomplete'
        for index, row in enumerate(rows, 1):
            row = dict(row, batch_id=batch.name, batch_complete=complete, batch_role=role,
                       source_manifest=manifest.relative_to(source).as_posix(), frame_index=index)
            classification.append(row)
        batches.append({'batch_id': batch.name, 'class_name': config.get('class_name'),
                        'target_count': config.get('target_count'), 'manifest_rows': len(rows),
                        'complete': complete, 'role': role,
                        'source_scout': config.get('region', {}).get('source_sample_path'),
                        'source_image_sha256': config.get('region', {}).get('source_image_sha256'),
                        'region_id': config.get('region', {}).get('region_id'),
                        'script_sha256_recorded': config.get('script_sha256'),
                        'script_sha256_actual': hashes.get(str(batch/'collection_script.py')),
                        'summary_exists': (batch/'collection_summary.json').exists(),
                        'split_status': 'unassigned_source_groups_need_review'})
    write_csv(out/'batch_index.csv', batches)
    write_json(out/'batch_configs.json', config_by_batch)
    by_id = {r['sample_id']: r for r in classification}
    checks, samples, asset_refs, missing = [], {}, {}, []
    frame_dirs = sorted({p.parent for p in orbit.rglob('scene_snapshot.usdc')} | {p.parent for p in orbit.rglob('sample.json')})
    for frame in frame_dirs:
        absent = [name for name in COMPONENTS if not (frame/name).is_file()]
        if absent:
            missing.append({'capture_dir': frame.relative_to(source).as_posix(), 'missing': absent})
        if not (frame/'sample.json').is_file():
            continue
        sample = read_json(frame/'sample.json')
        sid = sample['sample_id']
        samples[sid] = sample
        row = by_id.get(sid)
        image_path = frame/'image.png'
        image = image_data.get(str(image_path), {})
        item = {'sample_id': sid, 'capture_dir': frame.relative_to(source).as_posix(),
                'source_kind': 'classification' if row else 'scout' if frame.parent.name == 'scouts' else 'pilot',
                'batch_id': row['batch_id'] if row else frame.relative_to(orbit).parts[0],
                'capture_group_id': row.get('capture_group_id') if row else frame.relative_to(orbit).parts[0],
                'trajectory_id': sample.get('trajectory_id'), 'scene_id': sample.get('scene_id'),
                'class_name': row.get('proposed_class') if row else None,
                'distance_mm': sample.get('distance_mm'),
                'image_sha256': hashes.get(str(image_path)),
                'pixel_sha256': image.get('pixel_sha256'), 'dhash256': image.get('dhash256'),
                'issues': [], 'file_hash_mismatches': [], 'within_30_60': False,
                'visual_status': 'pending_view_review', 'split': None}
        item['issues'].extend('missing:'+x for x in absent)
        sums = read_json(frame/'SHA256SUMS.json') if (frame/'SHA256SUMS.json').is_file() else {}
        for name, expected in sums.items():
            path = frame/name
            if hashes.get(str(path)) != expected.get('sha256') or (path.is_file() and path.stat().st_size != expected.get('bytes')):
                item['file_hash_mismatches'].append(name)
        for name in COMPONENTS[:-1]:
            if name not in sums:
                item['issues'].append('missing_checksum:'+name)
        if item['file_hash_mismatches']:
            item['issues'].append('capture_checksum_mismatch')
        if sample.get('image_sha256') != hashes.get(str(image_path)):
            item['issues'].append('sample_image_hash_mismatch')
        if image.get('issues') or (image.get('width'), image.get('height')) != (801, 801):
            item['issues'].append('image_decode_or_dimensions')
        if sample.get('image_resolution') != [801, 801]:
            item['issues'].append('metadata_dimensions')
        if not valid_pose(sample):
            item['issues'].append('pose')
        if 'optical forward local -Z' not in sample.get('camera_coordinate_convention', ''):
            item['issues'].append('axis_convention')
        distance = sample.get('distance_mm')
        item['within_30_60'] = bool(isinstance(distance, (float, int)) and math.isfinite(distance) and 30 <= distance <= 60)
        if sample.get('valid') is not True:
            item['issues'].append('capture_invalid')
        try:
            depth = np.load(frame/'depth_m.npy', mmap_mode='r', allow_pickle=False)
            ids = np.load(frame/'instance_ids.npy', mmap_mode='r', allow_pickle=False)
            if depth.shape[:2] != (801, 801) or ids.shape[:2] != (801, 801):
                item['issues'].append('buffer_dimensions')
            center = float(np.asarray(depth[400, 400]).reshape(-1)[0])*1000
            instance = int(np.asarray(ids[400, 400]).reshape(-1)[0])
            render = sample.get('renderer_occlusion_check', {})
            item['depth_center_mm'] = center if math.isfinite(center) else None
            item['depth_error_mm'] = abs(center-distance) if math.isfinite(center) else None
            item['depth_tolerance_mm'] = render.get('tolerance_m', 0)*1000
            item['instance_center_id'] = instance
            if not math.isfinite(center) or item['depth_error_mm'] > item['depth_tolerance_mm']:
                item['issues'].append('center_depth_disagrees_with_geometric_label')
            if instance != render.get('instance_id') or render.get('instance_path') != sample.get('stomach_mesh_path'):
                item['issues'].append('center_instance_disagrees')
        except Exception as exc:
            item['issues'].append('buffer_read:'+str(exc))
        sync = sample.get('synchronization', {})
        for key in ('render_completed', 'timeline_paused_before', 'timeline_paused_after',
                    'camera_world_matrix_unchanged', 'simulation_time_unchanged'):
            if sync.get(key) is not True:
                item['issues'].append('synchronization:'+key)
        if row:
            batch = orbit/row['batch_id']
            alias = batch/row['image']
            if hashes.get(str(alias)) != row.get('image_sha256') or row.get('image_sha256') != item['image_sha256']:
                item['issues'].append('alias_or_manifest_image_hash')
            if row.get('photo_dhash256') != item['dhash256']:
                item['issues'].append('manifest_dhash')
            for key in ('camera_position_world_m', 'camera_orientation_world', 'quaternion_order', 'distance_mm'):
                if row.get(key) != sample.get(key):
                    item['issues'].append('manifest_sample_disagreement:'+key)
            if not 5-1e-6 <= row.get('camera_view_change_deg', -1) <= 10+1e-6:
                item['issues'].append('view_change')
            if row.get('proposed_class_id') != MAPPING.get(row.get('proposed_class')):
                item['issues'].append('class_mapping')
            if row.get('source_script_sha256') != hashes.get(str(batch/'collection_script.py')):
                item['issues'].append('collection_script_hash')
        for dependency in sample.get('file_dependencies', []):
            key = (dependency.get('path'), dependency.get('sha256'))
            asset_refs[key] = dependency
        checks.append(item)
    write_json(out/'capture_checks.json', checks)
    write_json(out/'missing_files.json', missing)
    assets = []
    for index, ((name, expected), dependency) in enumerate(sorted(asset_refs.items())):
        path = Path(name)
        actual = digest(path) if path.is_file() else None
        record = {'source_path': name, 'expected_sha256': expected, 'actual_sha256': actual,
                  'matches': actual == expected, 'bytes': path.stat().st_size if path.is_file() else None}
        if actual:
            target = out/'external_assets'/actual/path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            record['bundle_path'] = target.relative_to(out).as_posix()
            if digest(target) != actual:
                raise RuntimeError('asset copy mismatch: '+name)
        assets.append(record)
    write_json(out/'external_asset_index.json', assets)
    # Record every raw sample and source label separately from any review proposal.
    check_by_id = {x['sample_id']: x for x in checks}
    flat = []
    for row in classification:
        check = check_by_id[row['sample_id']]
        flat.append({'sample_id': row['sample_id'], 'batch_id': row['batch_id'],
                     'frame_index': row['frame_index'], 'batch_role': row['batch_role'],
                     'batch_complete': row['batch_complete'], 'original_class': row['proposed_class'],
                     'original_class_id': row['proposed_class_id'], 'original_label_status': row['label_status'],
                     'image': str((orbit/row['batch_id']/row['image']).relative_to(source)),
                     'sample_json': check['capture_dir']+'/sample.json',
                     'sha256': check['image_sha256'], 'distance_mm': row['distance_mm'],
                     'capture_group_id': row['capture_group_id'], 'trajectory_id': row.get('trajectory_id'),
                     'scene_id': row['scene_id'], 'split': '',
                     'technical_check': 'pass' if not check['issues'] else 'fail',
                     'classification_status': 'pending_view_review', 'suggested_class': '',
                     'review_basis': 'whole-image versus central-target policy unresolved; source label preserved'})
    write_csv(out/'classification_index.csv', flat)
    write_csv(out/'classification_visual_review.csv', [dict(r, visual_observation='', reviewed_at='', reviewer='') for r in flat])
    # Pair tests use newly computed image hashes/fingerprints, not stored fingerprints.
    pairs, parents = [], list(range(len(checks)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for i, a in enumerate(checks):
        for j in range(i+1, len(checks)):
            b = checks[j]
            reasons, translation, angle, hamming = [], None, None, None
            if a.get('image_sha256') and a['image_sha256'] == b.get('image_sha256'):
                reasons.append('exact_file')
            if a.get('pixel_sha256') and a['pixel_sha256'] == b.get('pixel_sha256'):
                reasons.append('exact_decoded_pixels')
            try:
                translation, angle = pose_difference(samples[a['sample_id']], samples[b['sample_id']])
                if translation < 1e-5 and angle < 1e-4:
                    reasons.append('exact_pose')
                elif translation < 2 and angle < 5:
                    reasons.append('near_pose')
            except (ValueError, TypeError, KeyError):
                pass
            if a.get('dhash256') and b.get('dhash256'):
                hamming = (int(a['dhash256'], 16)^int(b['dhash256'], 16)).bit_count()
                if hamming < 4:
                    reasons.append('near_image_dhash')
            if reasons:
                pairs.append({'sample_a': a['sample_id'], 'sample_b': b['sample_id'],
                              'batch_a': a['batch_id'], 'batch_b': b['batch_id'],
                              'reasons': '|'.join(reasons), 'translation_mm': translation,
                              'angle_deg': angle, 'dhash_hamming': hamming})
                parents[find(j)] = find(i)
    write_csv(out/'duplicate_pairs.csv', pairs,
              ['sample_a','sample_b','batch_a','batch_b','reasons','translation_mm','angle_deg','dhash_hamming'])
    clusters = defaultdict(list)
    for i, item in enumerate(checks):
        clusters[find(i)].append(item)
    # Conservative candidates only; scout/pilot are checked under the same rules.
    def rank(item):
        row = by_id.get(item['sample_id'], {})
        return (0 if row.get('batch_role') == 'replacement_candidate' else
                1 if row.get('batch_complete') else 2 if row else 3,
                item['sample_id'])
    selected, distance_rows = [], []
    for members in clusters.values():
        qualified = sorted((x for x in members if not x['issues'] and x['within_30_60']), key=rank)
        winner = qualified[0]['sample_id'] if qualified else None
        cluster_id = 'identity_'+hashlib.sha256('|'.join(sorted(x['sample_id'] for x in members)).encode()).hexdigest()[:16]
        for item in members:
            if item['sample_id'] == winner:
                selected.append(item)
            distance_rows.append({'sample_id': item['sample_id'], 'image': item['capture_dir']+'/image.png',
                                  'sha256': item['image_sha256'], 'distance_mm': item['distance_mm'],
                                  'source_kind': item['source_kind'], 'batch_id': item['batch_id'],
                                  'capture_group_id': item['capture_group_id'],
                                  'identity_cluster': cluster_id, 'representative_id': winner,
                                  'selected_technical_candidate': item['sample_id'] == winner,
                                  'technical_check': 'pass' if not item['issues'] else 'fail',
                                  'within_30_60': item['within_30_60'], 'visual_status': 'pending_view_review',
                                  'classification_shared_identity': item['sample_id'] in by_id,
                                  'split': '', 'acceptance_status': 'candidate_not_final'})
    write_csv(out/'distance_index.csv', distance_rows)
    # Link groups by actual scout identity and duplicate pairs; never manufacture independence.
    relations = []
    for a in batches:
        for b in batches:
            if a['batch_id'] < b['batch_id'] and a['source_image_sha256'] and a['source_image_sha256'] == b['source_image_sha256']:
                relations.append({'batch_a': a['batch_id'], 'batch_b': b['batch_id'], 'basis': 'same_source_scout_sha256'})
    for pair in pairs:
        if pair['batch_a'] != pair['batch_b']:
            relations.append({'batch_a': pair['batch_a'], 'batch_b': pair['batch_b'], 'basis': pair['reasons']})
    write_csv(out/'source_relations.csv', relations, ['batch_a','batch_b','basis'])
    # All classification records get an overview. Pages are not automatically marked reviewed.
    sheet_dir = out/'contact_sheets'
    sheet_dir.mkdir()
    pages = []
    categories = defaultdict(list)
    for row in flat:
        categories[row['batch_id']].append((row['sample_id'], source/row['image'], row['original_class'], row['distance_mm']))
    for item in checks:
        if item['sample_id'] not in by_id:
            categories[item['batch_id']].append((item['sample_id'], source/item['capture_dir']/'image.png', item['source_kind'], item['distance_mm']))
    for group, members in sorted(categories.items()):
        for offset in range(0, len(members), 20):
            canvas = Image.new('RGB', (1400, 1232), '#202020')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), 'REVIEW ONLY / '+group, fill='white')
            for tile, (sid, path, label, distance) in enumerate(members[offset:offset+20]):
                x, y = (tile%5)*280, 28+(tile//5)*300
                try:
                    with Image.open(path) as im:
                        thumb = im.convert('RGB')
                        thumb.thumbnail((276,276))
                        canvas.paste(thumb, (x+(280-thumb.width)//2,y))
                except Exception:
                    draw.text((x+4,y+30), 'READ ERROR', fill='red')
                draw.text((x+3,y+278), '%03d %s %.2fmm' % (offset+tile+1,label,distance), fill='white')
            name = '%s_page%02d.jpg' % (group,offset//20+1)
            canvas.save(sheet_dir/name, quality=92)
            pages.append({'sheet': 'contact_sheets/'+name, 'samples': len(members[offset:offset+20]),
                          'status': 'pending_view_review', 'observations': ''})
    write_csv(out/'contact_sheet_review.csv', pages)
    modified = [str(p.relative_to(source)) for p in files if not p.exists() or (p.stat().st_size,p.stat().st_mtime_ns) != stats[str(p)]]
    counts = Counter(r['proposed_class'] for r in classification)
    full_counts = Counter(r['proposed_class'] for r in classification if r['batch_complete'])
    source_groups = {c: sorted({r['capture_group_id'] for r in classification if r['proposed_class']==c and r['batch_complete']}) for c in MAPPING}
    previous = source/'image_reviews/review_20261118T095704Z_cb480d/image_integrity_checks.json'
    summary = {'schema': 'gastric_offline_audit_v1', 'started_utc': started,
               'finished_utc': datetime.now(timezone.utc).isoformat(), 'source': str(source),
               'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': pillow_version},
               'source_file_count':len(files), 'source_bytes':sum(x['bytes'] for x in inventory),
               'source_modified_during_audit':modified, 'image_file_count':len(image_checks),
               'image_decode_failures':sum(bool(x['issues']) for x in image_checks),
               'orbit_sample_count':len(checks), 'capture_technical_failures':sum(bool(x['issues']) for x in checks),
               'classification_rows':len(classification), 'classification_rows_by_class':dict(counts),
               'complete_batch_candidate_counts':dict(full_counts), 'incomplete_rows':sum(not r['batch_complete'] for r in classification),
               'classification_approved_counts':{c:0 for c in MAPPING}, 'classification_pending_counts':dict(counts),
               'classification_label_policy':'unresolved_whole_image_vs_central_target',
               'complete_candidate_capture_group_ids':source_groups,
               'distance_in_range_before_dedup':sum(x['within_30_60'] for x in checks),
               'distance_technical_candidates_after_conservative_dedup':len(selected),
               'distance_selected_by_source_kind':dict(Counter(x['source_kind'] for x in selected)),
               'distance_final_accepted':0, 'distance_target':500,
               'distance_minimum_candidate_shortfall':max(0,500-len(selected)),
               'duplicate_pair_counts':dict(Counter(reason for pair in pairs for reason in pair['reasons'].split('|'))),
               'orphan_or_partial_capture_dirs':len(missing), 'external_asset_versions':len(assets),
               'external_asset_mismatches':sum(not a['matches'] for a in assets),
               'source_python_parse_issues':source_parse_issues,
               'historical_review':{'path':str(previous), 'sha256_actual_now':hashes.get(str(previous)),
                                    'historical_result':{'images':200,'file_issues':0,'asset_issues':0},
                                    'scope':'historical result only; new checks are capture_checks.json'},
               'classification_visual_review_200_csv_found':[str(p) for p in source.rglob('classification_visual_review_200.csv')],
               'unverified':['full-resolution semantic review','whole-image versus central-target classification policy',
                             'GPU replay and first-hit raycast recomputation','independent train/validation/test allocation'],
               'notes':['Saved center Euclidean depth is compared only at optical-axis center; no off-axis depth substitution.',
                        'All canonical images are decoded, hashed, fingerprinted and paired with source metadata.',
                        'Source scout/pose relations take precedence over arbitrary capture group IDs.',
                        'Source directory contents are archived, including outside-orbit debug data; only orbit samples count toward targets.']}
    write_json(out/'audit_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if modified:
        raise SystemExit('Source changed during audit; do not publish this audit')


if __name__ == '__main__':
    main()
