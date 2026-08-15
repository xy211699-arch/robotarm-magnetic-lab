"""Topology-preserving navigation on an approved triangular stomach surface."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .geometry import closest_point_on_triangle, normalized


class SurfaceLostError(RuntimeError):
    """No valid same-component surface continuation exists."""


@dataclass(frozen=True)
class SurfaceHit:
    point_world: np.ndarray
    normal_world: np.ndarray
    triangle_id: int
    component_id: int
    barycentric: np.ndarray
    boundary_limited: bool = False


def _oriented_triangle_normals(vertices: np.ndarray, triangles: np.ndarray, sign: int) -> np.ndarray:
    values = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths <= 1.0e-14):
        raise ValueError("surface contains a degenerate triangle")
    return int(sign) * values / lengths[:, None]


def _topology(triangles: np.ndarray):
    edge_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_id, triangle in enumerate(triangles):
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge_map[tuple(sorted((int(first), int(second))))].append(int(triangle_id))
    if any(len(values) > 2 for values in edge_map.values()):
        raise ValueError("surface contains a non-manifold edge")
    adjacency = [set() for _ in range(len(triangles))]
    boundary = set()
    for edge, owners in edge_map.items():
        if len(owners) == 1:
            boundary.add(edge)
        elif len(owners) == 2:
            adjacency[owners[0]].add(owners[1])
            adjacency[owners[1]].add(owners[0])
    return tuple(tuple(sorted(values)) for values in adjacency), frozenset(boundary)


def _components(adjacency: tuple[tuple[int, ...], ...]) -> np.ndarray:
    result = np.full(len(adjacency), -1, dtype=np.int64)
    component = 0
    for start in range(len(adjacency)):
        if result[start] >= 0:
            continue
        queue = deque([start])
        result[start] = component
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if result[neighbor] < 0:
                    result[neighbor] = component
                    queue.append(neighbor)
        component += 1
    return result


class SurfaceNavigationMesh:
    def __init__(
        self,
        *,
        vertices: np.ndarray,
        triangles: np.ndarray,
        normals: np.ndarray,
        adjacency: tuple[tuple[int, ...], ...],
        boundary_edges: frozenset[tuple[int, int]],
        component_ids: np.ndarray,
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64).copy()
        self.triangles = np.asarray(triangles, dtype=np.int64).copy()
        self.normals = np.asarray(normals, dtype=np.float64).copy()
        self.adjacency = adjacency
        self.boundary_edges = boundary_edges
        self.component_ids = np.asarray(component_ids, dtype=np.int64).copy()
        self.centroids = self.vertices[self.triangles].mean(axis=1)

    @classmethod
    def from_reference(cls, reference, inward_sign: int) -> "SurfaceNavigationMesh":
        if int(inward_sign) not in (-1, 1):
            raise ValueError("inward_sign must be -1 or +1")
        vertices = np.asarray(reference.vertices_world, dtype=np.float64)
        triangles = np.asarray(reference.triangles, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("surface arrays must have shapes (N,3) and (M,3)")
        adjacency, boundary = _topology(triangles)
        return cls(
            vertices=vertices,
            triangles=triangles,
            normals=_oriented_triangle_normals(vertices, triangles, int(inward_sign)),
            adjacency=adjacency,
            boundary_edges=boundary,
            component_ids=_components(adjacency),
        )

    def local_candidate_triangles(
        self, triangle_id: int, target_world: np.ndarray, recovery_radius_m: float
    ) -> tuple[int, ...]:
        triangle_id = int(triangle_id)
        if not 0 <= triangle_id < len(self.triangles):
            raise SurfaceLostError(f"invalid active triangle {triangle_id}")
        component = int(self.component_ids[triangle_id])
        candidates = {triangle_id, *self.adjacency[triangle_id]}
        distance = np.linalg.norm(self.centroids - np.asarray(target_world), axis=1)
        nearby = np.flatnonzero(
            (self.component_ids == component) & (distance <= float(recovery_radius_m))
        )
        candidates.update(int(value) for value in nearby)
        return tuple(sorted(candidates))

    def _is_boundary_projection(self, triangle_id: int, barycentric: np.ndarray) -> bool:
        triangle = self.triangles[int(triangle_id)]
        for zero_index in np.flatnonzero(np.asarray(barycentric) <= 1.0e-10):
            edge_vertices = [int(triangle[index]) for index in range(3) if index != int(zero_index)]
            if tuple(sorted(edge_vertices)) in self.boundary_edges:
                return True
        return False

    def rank_projected_candidates(self, target: np.ndarray, candidates) -> list[tuple]:
        ranked = []
        for triangle_id in candidates:
            point, barycentric = closest_point_on_triangle(
                target, self.vertices[self.triangles[int(triangle_id)]]
            )
            distance_sq = float((point - target) @ (point - target))
            ranked.append(
                (
                    distance_sq,
                    int(triangle_id),
                    tuple(float(value) for value in barycentric),
                    point,
                    barycentric,
                )
            )
        ranked.sort(key=lambda item: item[:3])
        return ranked

    def closest_hit(self, point_world: np.ndarray, component_id: int | None = None) -> SurfaceHit:
        """Return the exact deterministic closest point, optionally on one component."""
        if component_id is None:
            candidates = range(len(self.triangles))
        else:
            candidates = np.flatnonzero(self.component_ids == int(component_id))
        ranked = self.rank_projected_candidates(
            np.asarray(point_world, dtype=np.float64).reshape(3), candidates
        )
        if not ranked:
            raise SurfaceLostError("surface has no eligible triangle")
        _, triangle_id, _, point, barycentric = ranked[0]
        return SurfaceHit(
            point_world=np.asarray(point, dtype=np.float64),
            normal_world=normalized(self.normals[triangle_id], name="surface normal"),
            triangle_id=int(triangle_id),
            component_id=int(self.component_ids[triangle_id]),
            barycentric=np.asarray(barycentric, dtype=np.float64),
            boundary_limited=False,
        )

    def advance(
        self,
        triangle_id: int,
        point_world: np.ndarray,
        tangent_delta_world: np.ndarray,
        recovery_radius_m: float,
    ) -> SurfaceHit:
        if recovery_radius_m <= 0.0:
            raise ValueError("recovery_radius_m must be positive")
        target = np.asarray(point_world, dtype=np.float64).reshape(3) + np.asarray(
            tangent_delta_world, dtype=np.float64
        ).reshape(3)
        candidates = self.local_candidate_triangles(triangle_id, target, recovery_radius_m)
        ranked = self.rank_projected_candidates(target, candidates)
        if not ranked:
            raise SurfaceLostError("no same-component surface candidate")
        distance_sq, selected, _, point, barycentric = ranked[0]
        boundary = self._is_boundary_projection(selected, barycentric) and distance_sq > 1.0e-20
        if math_sqrt(distance_sq) > float(recovery_radius_m) and not boundary:
            raise SurfaceLostError("same-component surface is outside recovery radius")
        return SurfaceHit(
            point_world=np.asarray(point, dtype=np.float64),
            normal_world=normalized(self.normals[selected], name="surface normal"),
            triangle_id=selected,
            component_id=int(self.component_ids[selected]),
            barycentric=np.asarray(barycentric, dtype=np.float64),
            boundary_limited=bool(boundary),
        )


def math_sqrt(value: float) -> float:
    return float(np.sqrt(max(0.0, float(value))))
