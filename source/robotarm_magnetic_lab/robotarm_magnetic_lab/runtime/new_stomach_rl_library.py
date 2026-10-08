"""Frozen external pose library, private sampler, no regeneration/fallback."""
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from robotarm_magnetic_lab.coverage.new_stomach_rl_coverage import GEOMETRY_SHA as RUNTIME_MESH_SHA


class FrozenPoseLibrary:
    def __init__(self, path, split, seed, fixed_pose_id=None):
        if split not in ('train','validation'):
            raise ValueError('preflight must not sample the test split')
        manifest = json.loads(Path(path).read_text())
        digest = hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k != 'config_sha256'},
            sort_keys=True,separators=(',',':')).encode()).hexdigest()
        raw = Path(manifest['data_path']).read_bytes()
        if (digest != manifest['config_sha256'] or manifest['stomach_geometry_sha256'] != RUNTIME_MESH_SHA
            or hashlib.sha256(raw).hexdigest() != manifest['data_sha256']
            or len(raw) != manifest['data_bytes']):
            raise ValueError('frozen new-stomach library identity mismatch')
        records = [json.loads(line) for line in raw.splitlines()]
        if Counter(r['split'] for r in records) != {'train':1000,'validation':100,'test':100}:
            raise ValueError('library split mismatch')
        self.records = [r for r in records if r['split'] == split]
        for record in self.records:
            if record['stomach_geometry_sha256'] != RUNTIME_MESH_SHA:
                raise ValueError('old stomach pose rejected')
        self.fixed = next((r for r in self.records if r['pose_id'] == fixed_pose_id),None)
        if fixed_pose_id is not None and self.fixed is None:
            raise ValueError('fixed pose must belong to requested split')
        self.rng = np.random.default_rng(seed)

    def sample(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        record = self.fixed if self.fixed is not None else self.records[int(self.rng.integers(len(self.records)))]
        pose = np.asarray(record['pose_world_xyzw'])
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-5:
            raise ValueError('invalid frozen pose')
        return record['pose_id'], pose
