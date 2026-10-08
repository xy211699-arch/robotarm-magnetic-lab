"""Exact shared visibility with independent physical 10Hz and RGB 1Hz clocks.

No RGB render occurs in the geometry path. Both use the archived incidence,
distance, circular-FOV, inward-normal and CUDA first-hit implementations.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from .area_weights import target_vertex_area_weights
from .batched_accumulator import BatchedCoverageAccumulator
from .batched_visibility import (BatchedWarpFirstHitRaycaster, batched_candidate_mask,
    build_incident_face_table, visible_from_batched_first_hits)

GEOMETRY_SHA = 'c4028caa5da750d769a6c5e39709bd7caac975339f35eac5d36fc0e48036c72e'
MASK_SHA = '7261eeec9deeefcd960bb245aae0f495a459f9e3a221abd460017783fdd4d58c'


def validate_frozen_target(geometry_sha, labels):
    if geometry_sha != GEOMETRY_SHA:
        raise ValueError('new stomach runtime geometry hash required')
    labels = np.asarray(labels)
    if (labels.shape != (429029,) or np.count_nonzero(labels) != 176109
            or np.count_nonzero(labels == 0) != 252920):
        raise ValueError('frozen target face counts differ')


def optical_from_capsule(pose, offset=(0., 0., -.0127), camera_quat_xyzw=(0., 1., 0., 0.)):
    pose = torch.as_tensor(pose)
    if pose.ndim != 2 or pose.shape[1] != 7 or not torch.isfinite(pose).all():
        raise ValueError('finite [N,7] capsule pose xyzw required')
    q = pose[:, 3:]
    norm = torch.linalg.vector_norm(q, dim=1, keepdim=True)
    if (norm <= 0).any():
        raise ValueError('zero capsule quaternion')
    q = q / norm

    def rotate(vector):
        v = pose.new_tensor(vector).expand(len(pose), -1)
        t = 2 * torch.linalg.cross(q[:, :3], v)
        return v + q[:, 3:] * t + torch.linalg.cross(q[:, :3], t)

    from scipy.spatial.transform import Rotation
    axis_local = Rotation.from_quat(camera_quat_xyzw).as_matrix()[:, 2]
    return pose[:, :3] + rotate(offset), rotate(axis_local)


class GeometryVisibility:
    def __init__(self, reference, weights, raycaster, device):
        self.reference, self.device, self.raycaster = reference, str(device), raycaster
        self.weights = torch.tensor(np.asarray(weights), dtype=torch.float64, device=device)
        self.vertices = torch.tensor(np.asarray(reference.vertices_world), dtype=torch.float64, device=device)
        self.incident = build_incident_face_table(reference.incident_triangles, device)

    def __call__(self, optical):
        centers, axes = optical
        centers = centers.to(device=self.device, dtype=torch.float64)
        axes = axes.to(device=self.device, dtype=torch.float64)
        candidate, distances = batched_candidate_mask(self.vertices, centers, axes, .070, 60.)
        candidate &= self.weights[None] > 0
        hits, faces = self.raycaster.query(centers, candidate, self.vertices)
        return visible_from_batched_first_hits(self.reference, centers, self.vertices, candidate,
            distances, hits, faces, self.incident, normal_sign=1)


def frozen_visibility(reference, mask_path, device):
    mask_path = Path(mask_path)
    if hashlib.sha256(mask_path.read_bytes()).hexdigest() != MASK_SHA:
        raise ValueError('frozen target mask SHA mismatch')
    with np.load(mask_path, allow_pickle=False) as data:
        labels = data['face_labels']
        validate_frozen_target(reference.geometry_sha256, labels)
        weights = target_vertex_area_weights(reference, np.flatnonzero(labels == 0))
    if abs(weights.sum() - .04020530122973115) > 1e-9:
        raise ValueError('frozen reachable area differs')
    return GeometryVisibility(reference, weights, BatchedWarpFirstHitRaycaster(reference, device), device)


@dataclass(frozen=True)
class CoverageDelta:
    physics_tick: int
    sim_time_s: float
    sample_id: int
    coverage: torch.Tensor
    delta: torch.Tensor
    visible_mask: torch.Tensor


class DualRateCoverage:
    def __init__(self, visibility, num_envs, geometry_optics=optical_from_capsule):
        self.visibility, self.num_envs = visibility, int(num_envs)
        self.geometry_optics = geometry_optics
        # Persistent states must be regular tensors even when inference calls
        # construct/use the runtime. Do not change the shared accumulator.
        with torch.inference_mode(False):
            self.c10 = BatchedCoverageAccumulator(visibility.weights, num_envs, visibility.device)
            self.c1 = BatchedCoverageAccumulator(visibility.weights, num_envs, visibility.device)
        self.reset(list(range(num_envs)))

    def reset(self, env_ids):
        if sorted(int(i) for i in env_ids) != list(range(self.num_envs)):
            raise ValueError('partial reset requires isolated per-environment clocks; not implemented')
        with torch.inference_mode(False), torch.no_grad():
            rows = torch.arange(self.num_envs, device=self.visibility.device)
            self.c10.reset_rows(rows); self.c1.reset_rows(rows)
        self.c10_records, self.c1_records = [], []
        self._last_observed_tick = self._geometry_tick = self._frame = 0
        self._boundary_visible = None
        self._initialized = False

    def _update(self, accumulator, visible, sample, tick, records):
        with torch.inference_mode(False), torch.no_grad():
            result = accumulator.update(torch.full((self.num_envs,), sample,
                dtype=torch.int64, device=self.visibility.device), visible)
            coverage = result.coverage_fraction.clone()
            delta = result.newly_covered_area_m2 / accumulator.total_area_m2
            records.append({'physics_tick': tick, 'sim_time_s': tick / 240., 'sample_id': sample,
                'coverage': coverage.cpu().tolist(), 'delta': delta.cpu().tolist()})
            return CoverageDelta(tick, tick/240., sample, coverage, delta, visible.clone())

    def initialize(self, optical):
        if self._initialized:
            raise RuntimeError('coverage C0 already initialized')
        visible = self.visibility(optical)
        self._update(self.c10, visible, 0, 0, self.c10_records)
        self._update(self.c1, visible, 0, 0, self.c1_records)
        self._boundary_visible = visible
        self._initialized = True

    def update_geometry_at_physics_tick(self, physics_tick, capsule_pose):
        tick = int(physics_tick)
        if not self._initialized or tick < self._last_observed_tick:
            raise ValueError('uninitialized/decreasing physical clock')
        self._last_observed_tick = tick
        if tick % 24 or tick == self._geometry_tick:
            return None
        if tick != self._geometry_tick + 24:
            raise ValueError('skipped actual 10Hz physical boundary')
        visible = self.visibility(self.geometry_optics(capsule_pose))
        self._geometry_tick = tick
        self._boundary_visible = visible
        return self._update(self.c10, visible, tick//24, tick, self.c10_records)

    def observe_camera_frame(self, frame_id, sim_time_s, optical_pose):
        frame = int(frame_id)
        if abs(float(sim_time_s) - frame) > 1e-8 or frame != self._frame + 1:
            raise ValueError('1Hz RGB clock must increment exactly one real frame')
        if self._geometry_tick != frame * 240:
            raise RuntimeError('RGB and geometry must share the same boundary')
        visible = self.visibility(optical_pose)
        if not torch.equal(visible, self._boundary_visible):
            raise RuntimeError('RGB and geometry visibility differ at the same boundary')
        result = self._update(self.c1, visible, frame, frame*240, self.c1_records)
        self._frame = frame
        if (self.c1.mask & ~self.c10.mask).any():
            raise RuntimeError('C1 cumulative mask is not a subset of C10')
        return result
