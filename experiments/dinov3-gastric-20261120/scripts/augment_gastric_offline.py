"""Materialize reproducible lighting and camera-roll variants on the developer machine."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


SEED = 42
CLASS_COUNTS = {'normal': 100, 'raised': 50, 'ulcerated': 50}
EDGES = tuple(range(30, 62, 2))
MEAN_MM = 45.0
SD_MM = 7.5
ROTATE = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def stable_rng(*parts):
    key = ':'.join(str(part) for part in (SEED, *parts))
    return random.Random(hashlib.sha256(key.encode()).digest())


def local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'File outside dataset: {relative}')
    return path


def gaussian_cdf(distance):
    return 0.5 * (1 + math.erf((distance - MEAN_MM) / (SD_MM * math.sqrt(2))))


def target_counts(total):
    mass = gaussian_cdf(EDGES[-1]) - gaussian_cdf(EDGES[0])
    probabilities = [(gaussian_cdf(hi) - gaussian_cdf(lo)) / mass for lo, hi in itertools.pairwise(EDGES)]
    expected = [total * p for p in probabilities]
    counts = [math.floor(n) for n in expected]
    by_remainder = sorted(range(len(counts)), key=lambda i: (-(expected[i] - counts[i]), i))
    for i in by_remainder[:total - sum(counts)]:
        counts[i] += 1
    return counts, probabilities


def distance_bin(label):
    if not EDGES[0] <= label <= EDGES[-1]:
        raise ValueError(f'Distance {label} mm is outside the input interval')
    return min(int((label - EDGES[0]) // 2), len(EDGES) - 2)


def regression_allocation(rows, target):
    buckets = defaultdict(list)
    counts, probabilities = target_counts(target)
    for row in rows:
        if row['split'] == 'train':
            buckets[distance_bin(float(row['distance']))].append(row)
    copies = {r['sample_id']: 1 for r in rows}
    for idx, goal in enumerate(counts):
        parents = sorted(buckets[idx], key=lambda r: r['sample_id'])
        if not parents or goal < len(parents):
            raise ValueError(f'Target {goal} is incompatible with {len(parents)} original images in bin {idx}')
        stable_rng('regression-bin', idx).shuffle(parents)
        each, remainder = divmod(goal, len(parents))
        for order, row in enumerate(parents):
            copies[row['sample_id']] = each + (order < remainder)
    report = [dict(lower_mm=lo, upper_mm=hi, source_train=len(buckets[i]), target_probability=probabilities[i],
                   target_train=counts[i]) for i, (lo, hi) in enumerate(itertools.pairwise(EDGES))]
    return copies, report


def variants(root, stage, row, count, task):
    sid = row['sample_id']
    image_path = local_path(root, row['image_path'])
    expected = row['sha256']
    if sha256(image_path) != expected:
        raise ValueError(f'Unexpected source SHA-256: {sid}')
    rng = stable_rng(task, sid)
    angles = [0, 90, 180, 270]
    rng.shuffle(angles)
    with Image.open(image_path) as source:
        if source.size != (801, 801):
            raise ValueError(f'Source image size changed: {sid}')
        original = source.convert('RGB')
    base_folder = row['label'] if task == 'classification' else row['split']
    output = []
    sibling_hashes = set()
    for i in range(count):
        variant = sid if i == 0 else f'{sid}__aug{i:03d}'
        relative = f'{task}/images/{base_folder}/{variant}.png'
        path = stage / relative
        if i == 0:
            shutil.copyfile(image_path, path)
            angle, exposure, gamma = 0, 0.0, 1.0
        else:
            angle = angles[(i - 1) % len(angles)]
            # Sample a mild exposure and contrast response, spanning dark and bright cases.
            exposure = rng.uniform(0.10, 0.35) * (1 if i % 2 else -1)
            gamma = rng.uniform(0.9, 1.1)
            table = [min(255, max(0, round(255 * (val / 255) ** gamma * 2 ** exposure))) for val in range(256)]
            transformed = original.transpose(ROTATE[angle]) if angle else original
            transformed.point(table * 3).save(path, format='PNG', compress_level=2)
        variant_hash = sha256(path)
        if variant_hash in sibling_hashes:
            raise ValueError(f'Duplicate output within {sid}: {variant}')
        sibling_hashes.add(variant_hash)
        entry = dict(sample_id=variant, parent_sample_id=sid, image_path=relative, sha256=variant_hash,
                     parent_sha256=expected, parent_image_path=row['image_path'], parent_metadata_path=row['metadata_path'],
                     group_id=row.get('group_id') or row['capture_group_id'], capture_group_id=row['capture_group_id'],
                     split=row['split'], scene_id=row.get('scene_id', ''), is_augmented=i > 0,
                     rotation_cw_deg=angle, exposure_ev=exposure, gamma=gamma)
        if task == 'classification':
            entry.update(label=row['label'], label_id=row['label_id'], label_review_status=row['label_review_status'])
        else:
            entry.update(distance=row['distance'], distance_unit=row['distance_unit'],
                         acceptance_status=row['acceptance_status'], bin_lower_mm=row['bin_lower_mm'])
        output.append(entry)
    return output


def generate(task, root, stage, rows, copies, workers):
    for row in rows:
        folder = row['label'] if task == 'classification' else row['split']
        (stage / task / 'images' / folder).mkdir(parents=True, exist_ok=True)
    result = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = (pool.submit(variants, root, stage, row, copies[row['sample_id']], task) for row in rows)
        futures = list(jobs)
        for i, job in enumerate(futures, 1):
            result.extend(job.result())
            if i % 25 == 0 or i == len(futures):
                print(f'{task}: {i}/{len(futures)} source frames processed', flush=True)
    return result


def verify(rows, originals, task):
    parents = {r['sample_id']: r for r in originals}
    if len({r['sample_id'] for r in rows}) != len(rows) or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError(f'{task} has duplicate output IDs or hashes')
    groups = defaultdict(set)
    for r in rows:
        source = parents[r['parent_sample_id']]
        assert r['split'] == source['split'] and r['parent_sha256'] == source['sha256']
        groups[r['group_id']].add(r['split'])
        if task == 'classification':
            assert r['label'] == source['label'] and r['label_review_status'] == source['label_review_status']
        else:
            assert r['distance'] == source['distance']
            assert r['split'] == 'train' or not r['is_augmented']
        if not r['is_augmented']:
            assert r['sha256'] == source['sha256']
    assert all(len(splits) == 1 for splits in groups.values())
    assert Counter(r['parent_sample_id'] for r in rows if not r['is_augmented']) == dict.fromkeys(parents, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification-root', type=Path, required=True)
    parser.add_argument('--regression-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regression-train-size', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    classification_root, regression_root, output = (p.resolve() for p in (args.classification_root, args.regression_root, args.output))
    if output.exists():
        raise FileExistsError(output)
    class_csv, distance_csv = classification_root / 'classification.csv', regression_root / 'regression.csv'
    classes, distances = read_csv(class_csv), read_csv(distance_csv)
    assert Counter(r['label'] for r in classes) == CLASS_COUNTS
    assert all(not r['split'] for r in classes)
    assert Counter(r['split'] for r in distances) == {'train': 208, 'val': 40, 'test': 64}
    source_hashes = {'classification': sha256(class_csv), 'regression': sha256(distance_csv)}
    copies, bins = regression_allocation(distances, args.regression_train_size)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f'.{output.name}-'))
    print(f'Writing staged images in {stage}', flush=True)
    labels = generate('classification', classification_root, stage, classes, {r['sample_id']: 10 for r in classes}, args.workers)
    values = generate('regression', regression_root, stage, distances, copies, args.workers)
    verify(labels, classes, 'classification')
    verify(values, distances, 'regression')
    assert Counter(r['label'] for r in labels) == {k: v * 10 for k, v in CLASS_COUNTS.items()}
    train = [r for r in values if r['split'] == 'train']
    actual = Counter(distance_bin(float(r['distance'])) for r in train)
    for i, row in enumerate(bins):
        row['actual_train'] = actual[i]
        row['augmented_added'] = actual[i] - row['source_train']
        assert row['target_train'] == actual[i]
    assert len(train) == args.regression_train_size

    write_csv(stage / 'classification.csv', labels)
    write_csv(stage / 'regression.csv', values)
    for split in ('train', 'val', 'test'):
        write_csv(stage / f'regression-{split}.csv', [r for r in values if r['split'] == split])
    write_csv(stage / 'distance-bins.csv', bins)
    report = {
        'classification_source': str(classification_root), 'regression_source': str(regression_root),
        'source_manifest_sha256': source_hashes, 'seed': SEED, 'classification_split': 'unassigned',
        'classification_counts': dict(Counter(r['label'] for r in labels)),
        'classification_originals': len(classes), 'classification_added': len(labels) - len(classes),
        'regression_counts': dict(Counter(r['split'] for r in values)),
        'regression_train_unique_source_labels': len({float(r['distance']) for r in train}),
        'regression_train_parent_count': len({r['parent_sample_id'] for r in train}),
        'regression_train_max_images_per_parent_including_original': max(copies.values()),
        'regression_train_mean_mm': sum(float(r['distance']) for r in train) / len(train),
        'regression_train_added': sum(r['is_augmented'] for r in train),
        'distance_target': {'family': 'truncated normal', 'mean_mm': MEAN_MM, 'sd_mm': SD_MM,
                            'bounds_mm': [EDGES[0], EDGES[-1]], 'bin_width_mm': 2,
                            'matching': 'exact integer quotas at bin level; source distances remain unchanged'},
        'augmentation': {'rotation_clockwise_degrees': [0, 90, 180, 270], 'exposure_absolute_ev_range': [0.1, 0.35],
                         'gamma_range': [0.9, 1.1], 'image_size': [801, 801]},
        'output_manifest_sha256': {name: sha256(stage / name) for name in ('classification.csv', 'regression.csv')},
        'ready': True,
    }
    (stage / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    stage.rename(output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
