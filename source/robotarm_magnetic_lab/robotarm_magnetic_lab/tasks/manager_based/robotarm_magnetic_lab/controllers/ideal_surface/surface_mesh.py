"""Topology-preserving navigation on an approved triangular stomach surface."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .geometry import normalized


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


def _closest_points_on_triangles(
    point: np.ndarray, triangle_vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Ericson closest-point regions, equivalent to the scalar helper."""
    point = np.asarray(point, dtype=np.float64).reshape(3)
    triangles = np.asarray(triangle_vertices, dtype=np.float64).reshape(-1, 3, 3)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    points = np.empty((len(triangles), 3), dtype=np.float64)
    barycentric = np.empty_like(points)
    assigned = np.zeros(len(triangles), dtype=np.bool_)

    def select(mask: np.ndarray) -> np.ndarray:
        selected = np.asarray(mask, dtype=np.bool_) & ~assigned
        assigned[selected] = True
        return selected

    mask = select((d1 <= 0.0) & (d2 <= 0.0))
    points[mask], barycentric[mask] = a[mask], (1.0, 0.0, 0.0)
    mask = select((d3 >= 0.0) & (d4 <= d3))
    points[mask], barycentric[mask] = b[mask], (0.0, 1.0, 0.0)
    mask = select((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))
    values = d1[mask] / (d1[mask] - d3[mask])
    points[mask] = a[mask] + values[:, None] * ab[mask]
    barycentric[mask] = np.column_stack((1.0 - values, values, np.zeros(len(values))))
    mask = select((d6 >= 0.0) & (d5 <= d6))
    points[mask], barycentric[mask] = c[mask], (0.0, 0.0, 1.0)
    mask = select((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))
    values = d2[mask] / (d2[mask] - d6[mask])
    points[mask] = a[mask] + values[:, None] * ac[mask]
    barycentric[mask] = np.column_stack((1.0 - values, np.zeros(len(values)), values))
    mask = select((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0))
    values = (d4[mask] - d3[mask]) / (
        (d4[mask] - d3[mask]) + (d5[mask] - d6[mask])
    )
    points[mask] = b[mask] + values[:, None] * (c[mask] - b[mask])
    barycentric[mask] = np.column_stack((np.zeros(len(values)), 1.0 - values, values))
    mask = ~assigned
    denominator = 1.0 / (va[mask] + vb[mask] + vc[mask])
    v_values = vb[mask] * denominator
    w_values = vc[mask] * denominator
    points[mask] = a[mask] + v_values[:, None] * ab[mask] + w_values[:, None] * ac[mask]
    barycentric[mask] = np.column_stack(
        (1.0 - v_values - w_values, v_values, w_values)
    )
    return points, barycentric


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
        triangle_vertices = self.vertices[self.triangles]
        edge_lengths = np.linalg.norm(
            triangle_vertices[:, [1, 2, 0]] - triangle_vertices[:, [0, 1, 2]],
            axis=2,
        ).reshape(-1)
        positive_edges = edge_lengths[edge_lengths > 1.0e-12]
        median_edge = float(np.median(positive_edges)) if len(positive_edges) else 1.0e-3
        self._spatial_cell_size_m = max(4.0 * median_edge, 1.0e-4)
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for triangle_id, centroid in enumerate(self.centroids):
            buckets[self._cell_key(centroid)].append(int(triangle_id))
        self._centroid_cells = {
            key: tuple(values) for key, values in sorted(buckets.items())
        }

    def _cell_key(self, point_world: np.ndarray) -> tuple[int, int, int]:
        values = np.floor(
            np.asarray(point_world, dtype=np.float64).reshape(3)
            / self._spatial_cell_size_m
        ).astype(np.int64)
        return int(values[0]), int(values[1]), int(values[2])

    def _nearby_centroid_triangles(
        self, target_world: np.ndarray, recovery_radius_m: float
    ) -> tuple[int, ...]:
        target = np.asarray(target_world, dtype=np.float64).reshape(3)
        center = self._cell_key(target)
        cells = int(np.ceil(float(recovery_radius_m) / self._spatial_cell_size_m))
        candidates: list[int] = []
        for x_value in range(center[0] - cells, center[0] + cells + 1):
            for y_value in range(center[1] - cells, center[1] + cells + 1):
                for z_value in range(center[2] - cells, center[2] + cells + 1):
                    candidates.extend(
                        self._centroid_cells.get((x_value, y_value, z_value), ())
                    )
        if not candidates:
            return ()
        unique = np.asarray(sorted(set(candidates)), dtype=np.int64)
        distances = np.linalg.norm(self.centroids[unique] - target, axis=1)
        return tuple(int(value) for value in unique[distances <= float(recovery_radius_m)])

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
        nearby = self._nearby_centroid_triangles(target_world, recovery_radius_m)
        candidates.update(
            int(value)
            for value in nearby
            if int(self.component_ids[int(value)]) == component
        )
        return tuple(sorted(candidates))

    def _is_boundary_projection(self, triangle_id: int, barycentric: np.ndarray) -> bool:
        triangle = self.triangles[int(triangle_id)]
        for zero_index in np.flatnonzero(np.asarray(barycentric) <= 1.0e-10):
            edge_vertices = [int(triangle[index]) for index in range(3) if index != int(zero_index)]
            if tuple(sorted(edge_vertices)) in self.boundary_edges:
                return True
        return False

    def rank_projected_candidates(self, target: np.ndarray, candidates) -> list[tuple]:
        candidate_ids = np.asarray(tuple(candidates), dtype=np.int64)
        if not len(candidate_ids):
            return []
        points, barycentrics = _closest_points_on_triangles(
            target, self.vertices[self.triangles[candidate_ids]]
        )
        ranked = []
        for index, triangle_id in enumerate(candidate_ids):
            point = points[index]
            barycentric = barycentrics[index]
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
