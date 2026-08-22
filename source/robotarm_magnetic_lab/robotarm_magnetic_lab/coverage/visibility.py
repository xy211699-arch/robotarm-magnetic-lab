"""Circular-FOV candidate filtering and first-hit visibility queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .reference_mesh import ReferenceMesh


MAX_OBSERVATION_DISTANCE_M = 0.05
FOV_HALF_ANGLE_DEG = 60.0
HIT_DISTANCE_TOLERANCE_M = 1.0e-4
NORMAL_DOT_TOLERANCE = 1.0e-10


def triangle_normals(reference: ReferenceMesh, normal_sign: int = 1) -> np.ndarray:
    """Return signed unit world-space normals after USD winding correction.

    USD ``orientation`` defines winding handedness, but it does not declare
    which side of a thin anatomical shell is the lumen.  ``normal_sign`` is
    therefore an explicit, asset-calibrated lumen-side convention.
    """
    sign = int(normal_sign)
    if sign not in (-1, 1):
        raise ValueError("normal_sign must be -1 or +1")
    vertices = np.asarray(reference.vertices_world, dtype=np.float64)
    triangles = np.asarray(reference.triangles, dtype=np.int64)
    edges_a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edges_b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    normals = np.cross(edges_a, edges_b)
    norms = np.linalg.norm(normals, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("reference mesh contains a non-finite or degenerate face normal")
    return sign * normals / norms[:, None]


def camera_facing_first_hits(
    origin_world: np.ndarray,
    targets_world: np.ndarray,
    hit_face_ids: np.ndarray,
    reference: ReferenceMesh,
    tolerance: float = NORMAL_DOT_TOLERANCE,
    normal_sign: int = 1,
) -> np.ndarray:
    """Accept hits whose oriented face normal points strictly toward the camera."""
    origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
    targets = np.asarray(targets_world, dtype=np.float64).reshape(-1, 3)
    face_ids = np.asarray(hit_face_ids, dtype=np.int64).reshape(-1)
    if len(targets) != len(face_ids):
        raise ValueError("target and hit-face arrays must have equal length")
    directions = targets - origin
    norms = np.linalg.norm(directions, axis=1)
    valid = (
        np.isfinite(directions).all(axis=1)
        & np.isfinite(norms)
        & (norms > np.finfo(np.float64).eps)
        & (face_ids >= 0)
        & (face_ids < len(reference.triangles))
    )
    result = np.zeros(len(targets), dtype=np.bool_)
    if not np.any(valid):
        return result
    rays = directions[valid] / norms[valid, None]
    normals = triangle_normals(reference, normal_sign=normal_sign)[face_ids[valid]]
    result[valid] = np.einsum("ij,ij->i", normals, rays) < -abs(float(tolerance))
    return result


def candidate_vertices(
    vertices_world: np.ndarray,
    optical_center_world: np.ndarray,
    optical_axis_world: np.ndarray,
    max_distance_m: float = MAX_OBSERVATION_DISTANCE_M,
    half_angle_deg: float = FOV_HALF_ANGLE_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices/distances within the inclusive sphere and circular cone."""
    vertices = np.asarray(vertices_world, dtype=np.float64)
    center = np.asarray(optical_center_world, dtype=np.float64).reshape(3)
    axis = np.asarray(optical_axis_world, dtype=np.float64).reshape(3)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_world must have shape (N, 3)")
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 0.0:
        raise ValueError("optical axis must be finite and nonzero")
    axis /= axis_norm
    offsets = vertices - center
    distances = np.linalg.norm(offsets, axis=1)
    finite = np.isfinite(offsets).all(axis=1) & np.isfinite(distances)
    nonzero = distances > np.finfo(np.float64).eps
    directions = np.zeros_like(offsets)
    directions[nonzero] = offsets[nonzero] / distances[nonzero, None]
    cosine = directions @ axis
    threshold = float(np.cos(np.deg2rad(half_angle_deg)))
    epsilon = 16.0 * np.finfo(np.float64).eps
    mask = finite & nonzero & (distances <= max_distance_m + epsilon) & (cosine >= threshold - epsilon)
    indices = np.flatnonzero(mask).astype(np.int64)
    return indices, distances[indices]


def visible_from_first_hits(
    candidate_indices: np.ndarray,
    vertex_distances_m: np.ndarray,
    hit_distances_m: np.ndarray,
    hit_face_ids: np.ndarray,
    incident_triangles: tuple[tuple[int, ...], ...],
    tolerance_m: float = HIT_DISTANCE_TOLERANCE_M,
) -> np.ndarray:
    """Apply the incident-triangle and hit-distance visibility rule."""
    candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    vertex_distances = np.asarray(vertex_distances_m, dtype=np.float64).reshape(-1)
    hit_distances = np.asarray(hit_distances_m, dtype=np.float64).reshape(-1)
    face_ids = np.asarray(hit_face_ids, dtype=np.int64).reshape(-1)
    if not (len(candidates) == len(vertex_distances) == len(hit_distances) == len(face_ids)):
        raise ValueError("candidate and hit arrays must have equal length")
    visible = np.zeros(len(candidates), dtype=np.bool_)
    epsilon = 16.0 * np.finfo(np.float64).eps
    for ray_index, vertex_index in enumerate(candidates):
        if vertex_index < 0 or vertex_index >= len(incident_triangles):
            raise ValueError("candidate vertex index is out of range")
        face_id = int(face_ids[ray_index])
        visible[ray_index] = (
            face_id in incident_triangles[int(vertex_index)]
            and np.isfinite(hit_distances[ray_index])
            and abs(float(hit_distances[ray_index] - vertex_distances[ray_index]))
            <= tolerance_m + epsilon
        )
    return visible


class FirstHitRaycaster(Protocol):
    def query(self, origin_world: np.ndarray, targets_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class ScalarFirstHitRaycaster:
    """Deterministic small-fixture oracle; never use for production stomach rays."""

    reference: ReferenceMesh

    def query(self, origin_world: np.ndarray, targets_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        origin = np.asarray(origin_world, dtype=np.float64).reshape(3)
        targets = np.asarray(targets_world, dtype=np.float64).reshape(-1, 3)
        distances = np.full(len(targets), np.inf, dtype=np.float64)
        face_ids = np.full(len(targets), -1, dtype=np.int64)
        vertices = self.reference.vertices_world
        triangles = self.reference.triangles
        for ray_index, target in enumerate(targets):
            delta = target - origin
            target_distance = float(np.linalg.norm(delta))
            if target_distance <= np.finfo(np.float64).eps:
                continue
            direction = delta / target_distance
            for face_index, triangle in enumerate(triangles):
                v0, v1, v2 = vertices[triangle]
                edge1 = v1 - v0
                edge2 = v2 - v0
                pvec = np.cross(direction, edge2)
                determinant = float(np.dot(edge1, pvec))
                if abs(determinant) < 1.0e-12:
                    continue
                inverse = 1.0 / determinant
                tvec = origin - v0
                u = float(np.dot(tvec, pvec) * inverse)
                if u < -1.0e-12 or u > 1.0 + 1.0e-12:
                    continue
                qvec = np.cross(tvec, edge1)
                v = float(np.dot(direction, qvec) * inverse)
                if v < -1.0e-12 or u + v > 1.0 + 1.0e-12:
                    continue
                distance = float(np.dot(edge2, qvec) * inverse)
                if distance >= 0.0 and distance < distances[ray_index]:
                    distances[ray_index] = distance
                    face_ids[ray_index] = face_index
        return distances, face_ids


class WarpFirstHitRaycaster:
    """GPU-batched first-hit adapter over Isaac Lab's Warp mesh operation."""

    def __init__(self, reference: ReferenceMesh, device: str = "cuda:0") -> None:
        if not str(device).startswith("cuda"):
            raise ValueError("production visibility requires a CUDA device")
        from isaaclab.utils.warp.ops import convert_to_warp_mesh

        self.reference = reference
        self.device = str(device)
        self._mesh = convert_to_warp_mesh(
            np.asarray(reference.vertices_world, dtype=np.float32),
            np.asarray(reference.triangles.reshape(-1), dtype=np.int32),
            device=self.device,
        )

    def query(self, origin_world: np.ndarray, targets_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        from isaaclab.utils.warp.ops import raycast_mesh

        targets = np.asarray(targets_world, dtype=np.float32).reshape(-1, 3)
        if len(targets) == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
        origin = np.asarray(origin_world, dtype=np.float32).reshape(3)
        delta = targets - origin
        norms = np.linalg.norm(delta, axis=1)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("ray target must differ from origin")
        directions = delta / norms[:, None]
        starts_t = torch.as_tensor(origin, dtype=torch.float32, device=self.device).view(1, 1, 3)
        starts_t = starts_t.expand(1, len(targets), 3).contiguous()
        directions_t = torch.as_tensor(directions, dtype=torch.float32, device=self.device).view(1, -1, 3)
        _, distances, _, face_ids = raycast_mesh(
            starts_t,
            directions_t,
            self._mesh,
            max_dist=float(max(norms.max() + 1.0e-3, MAX_OBSERVATION_DISTANCE_M + 1.0e-3)),
            return_distance=True,
            return_face_id=True,
        )
        return (
            distances.detach().cpu().numpy().reshape(-1).astype(np.float64),
            face_ids.detach().cpu().numpy().reshape(-1).astype(np.int64),
        )
