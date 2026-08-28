"""Exact batched candidate and first-hit visibility gates for TASK-009D0."""

from __future__ import annotations

import numpy as np
import torch

from .reference_mesh import ReferenceMesh
from .visibility import (
    FOV_HALF_ANGLE_DEG,
    HIT_DISTANCE_TOLERANCE_M,
    MAX_OBSERVATION_DISTANCE_M,
    NORMAL_DOT_TOLERANCE,
    triangle_normals,
)


def build_incident_face_table(
    incident_triangles: tuple[tuple[int, ...], ...], device: str | torch.device
) -> torch.Tensor:
    """Build a padded ``[V,K]`` table whose unused entries are ``-1``."""
    width = max((len(faces) for faces in incident_triangles), default=0)
    width = max(width, 1)
    result = torch.full(
        (len(incident_triangles), width),
        -1,
        dtype=torch.int64,
        device=device,
    )
    for vertex_id, faces in enumerate(incident_triangles):
        if faces:
            result[vertex_id, : len(faces)] = torch.as_tensor(
                faces, dtype=torch.int64, device=device
            )
    return result


def batched_candidate_mask(
    vertices_local: torch.Tensor,
    optical_centers_local: torch.Tensor,
    optical_axes_world: torch.Tensor,
    max_distance_m: float = MAX_OBSERVATION_DISTANCE_M,
    half_angle_deg: float = FOV_HALF_ANGLE_DEG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inclusive circular-FOV candidates and all target distances."""
    vertices = vertices_local.to(dtype=torch.float64)
    centers = optical_centers_local.to(device=vertices.device, dtype=torch.float64)
    axes = optical_axes_world.to(device=vertices.device, dtype=torch.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_local must have shape [V,3]")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("optical_centers_local must have shape [E,3]")
    if axes.shape != centers.shape:
        raise ValueError("optical axes must match optical center shape")
    axis_norms = torch.linalg.vector_norm(axes, dim=1)
    if torch.any(~torch.isfinite(axes)).item() or torch.any(axis_norms <= 0).item():
        raise ValueError("optical axes must be finite and nonzero")
    axes = axes / axis_norms[:, None]
    offsets = vertices[None, :, :] - centers[:, None, :]
    distances = torch.linalg.vector_norm(offsets, dim=2)
    finite = torch.isfinite(offsets).all(dim=2) & torch.isfinite(distances)
    epsilon = 16.0 * torch.finfo(torch.float64).eps
    nonzero = distances > torch.finfo(torch.float64).eps
    safe_distances = torch.where(nonzero, distances, torch.ones_like(distances))
    directions = offsets / safe_distances[:, :, None]
    cosine = (directions * axes[:, None, :]).sum(dim=2)
    threshold = float(np.cos(np.deg2rad(half_angle_deg)))
    mask = (
        finite
        & nonzero
        & (distances <= float(max_distance_m) + epsilon)
        & (cosine >= threshold - epsilon)
    )
    return mask, distances


def visible_from_batched_first_hits(
    reference: ReferenceMesh,
    optical_centers_local: torch.Tensor,
    target_vertices_local: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_distances_m: torch.Tensor,
    hit_distances_m: torch.Tensor,
    hit_face_ids: torch.Tensor,
    incident_face_table: torch.Tensor,
    tolerance_m: float = HIT_DISTANCE_TOLERANCE_M,
    normal_tolerance: float = NORMAL_DOT_TOLERANCE,
    normal_sign: int = -1,
) -> torch.Tensor:
    """Apply candidate, incidence, distance and lumen-normal gates on device."""
    device = candidate_mask.device
    candidate = candidate_mask.to(dtype=torch.bool)
    target_distances = target_distances_m.to(device=device, dtype=torch.float64)
    hit_distances = hit_distances_m.to(device=device, dtype=torch.float64)
    face_ids = hit_face_ids.to(device=device, dtype=torch.int64)
    centers = optical_centers_local.to(device=device, dtype=torch.float64)
    targets = target_vertices_local.to(device=device, dtype=torch.float64)
    incident = incident_face_table.to(device=device, dtype=torch.int64)
    expected = candidate.shape
    if not (
        target_distances.shape == expected
        and hit_distances.shape == expected
        and face_ids.shape == expected
        and expected == (centers.shape[0], targets.shape[0])
        and incident.shape[0] == targets.shape[0]
    ):
        raise ValueError("batched visibility tensor shapes do not agree")
    if int(normal_sign) not in (-1, 1):
        raise ValueError("normal_sign must be -1 or +1")
    valid_face = (face_ids >= 0) & (face_ids < len(reference.triangles))
    incidence = (
        incident[None, :, :] == face_ids[:, :, None]
    ).any(dim=2) & valid_face
    epsilon = 16.0 * torch.finfo(torch.float64).eps
    distance_ok = torch.isfinite(hit_distances) & (
        torch.abs(hit_distances - target_distances)
        <= float(tolerance_m) + epsilon
    )
    rays = targets[None, :, :] - centers[:, None, :]
    ray_norms = torch.linalg.vector_norm(rays, dim=2)
    nonzero = ray_norms > torch.finfo(torch.float64).eps
    rays = rays / torch.where(nonzero, ray_norms, torch.ones_like(ray_norms))[:, :, None]
    normals = torch.as_tensor(
        triangle_normals(reference, normal_sign=normal_sign),
        device=device,
        dtype=torch.float64,
    )
    safe_faces = torch.clamp(face_ids, min=0, max=max(len(normals) - 1, 0))
    selected_normals = normals[safe_faces]
    normal_ok = (
        (selected_normals * rays).sum(dim=2) < -abs(float(normal_tolerance))
    ) & nonzero & valid_face
    return candidate & incidence & distance_ok & normal_ok


class BatchedWarpFirstHitRaycaster:
    """One CUDA Warp raycast over all ``E*V`` environment/vertex pairs."""

    def __init__(self, reference: ReferenceMesh, device: str = "cuda:0") -> None:
        if not str(device).startswith("cuda"):
            raise ValueError("production visibility requires a CUDA device")
        from isaaclab.utils.warp.ops import convert_to_warp_mesh

        self.reference = reference
        self.device = str(device)
        self._vertices = torch.as_tensor(
            np.asarray(reference.vertices_world, dtype=np.float64),
            device=self.device,
            dtype=torch.float64,
        )
        self._mesh = convert_to_warp_mesh(
            np.asarray(reference.vertices_world, dtype=np.float32),
            np.asarray(reference.triangles.reshape(-1), dtype=np.int32),
            device=self.device,
        )
        self.last_ray_count = 0
        self.last_candidate_counts = torch.empty(0, dtype=torch.int64, device=self.device)

    @property
    def vertices(self) -> torch.Tensor:
        return self._vertices

    def query(
        self,
        optical_centers_local: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        target_vertices_local: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab.utils.warp.ops import raycast_mesh

        centers = optical_centers_local.to(device=self.device, dtype=torch.float64)
        targets = self._vertices if target_vertices_local is None else target_vertices_local.to(
            device=self.device, dtype=torch.float64
        )
        offsets = targets[None, :, :] - centers[:, None, :]
        norms = torch.linalg.vector_norm(offsets, dim=2)
        valid = norms > torch.finfo(torch.float32).eps
        safe_norms = torch.where(valid, norms, torch.ones_like(norms))
        directions = (offsets / safe_norms[:, :, None]).to(dtype=torch.float32)
        safe_direction = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
        )
        directions = torch.where(valid[:, :, None], directions, safe_direction)
        starts = centers[:, None, :].expand(-1, len(targets), -1).to(dtype=torch.float32).contiguous()
        directions = directions.contiguous()
        _, distances, _, face_ids = raycast_mesh(
            starts,
            directions,
            self._mesh,
            max_dist=float(MAX_OBSERVATION_DISTANCE_M + 1.0e-3),
            return_distance=True,
            return_face_id=True,
        )
        distances = distances.to(device=self.device, dtype=torch.float64)
        face_ids = face_ids.to(device=self.device, dtype=torch.int64)
        if candidate_mask is None:
            candidate = valid
        else:
            candidate = candidate_mask.to(device=self.device, dtype=torch.bool) & valid
            if candidate.shape != norms.shape:
                raise ValueError("candidate mask must have shape [E,V]")
        distances = torch.where(
            candidate, distances, torch.full_like(distances, float("inf"))
        )
        face_ids = torch.where(candidate, face_ids, torch.full_like(face_ids, -1))
        self.last_ray_count = int(centers.shape[0] * targets.shape[0])
        self.last_candidate_counts = candidate.sum(dim=1)
        return distances, face_ids
