"""Unified deterministic local surface queries for flat and stomach scenes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .geometry import normalized
from ..ideal_surface.geometry import closest_point_on_triangle


@dataclass(frozen=True)
class LocalSurfaceHit:
    point_world: np.ndarray
    triangle_id: int
    one_ring_triangle_ids: tuple[int, ...]
    normal_world: np.ndarray
    geometry_digest: str

    def __post_init__(self) -> None:
        point = np.asarray(self.point_world, dtype=np.float64).reshape(3).copy()
        normal = normalized(self.normal_world, name="local surface normal")
        point.setflags(write=False)
        normal.setflags(write=False)
        object.__setattr__(self, "point_world", point)
        object.__setattr__(self, "normal_world", normal)


class TriangleMeshSurfaceQuery:
    """Exact closest-triangle query with shared-vertex one-ring normal smoothing."""

    def __init__(self, vertices, triangles, *, inward_sign: int) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3).copy()
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3).copy()
        if int(inward_sign) not in (-1, 1):
            raise ValueError("inward_sign must be -1 or +1")
        if len(self.vertices) < 3 or len(self.triangles) < 1:
            raise ValueError("surface mesh must be nonempty")
        if np.min(self.triangles) < 0 or np.max(self.triangles) >= len(self.vertices):
            raise ValueError("triangle index is outside the vertex array")
        raw_area_vectors = np.cross(
            self.vertices[self.triangles[:, 1]] - self.vertices[self.triangles[:, 0]],
            self.vertices[self.triangles[:, 2]] - self.vertices[self.triangles[:, 0]],
        )
        if np.any(np.linalg.norm(raw_area_vectors, axis=1) <= 1.0e-14):
            raise ValueError("surface contains a degenerate triangle")
        self._inward_area_vectors = int(inward_sign) * raw_area_vectors
        vertex_faces: list[list[int]] = [[] for _ in range(len(self.vertices))]
        for face_id, triangle in enumerate(self.triangles):
            for vertex_id in triangle:
                vertex_faces[int(vertex_id)].append(int(face_id))
        self._vertex_faces = tuple(tuple(sorted(set(items))) for items in vertex_faces)
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.vertices, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(self.triangles, dtype="<i8").tobytes())
        digest.update(bytes([int(inward_sign) & 0xFF]))
        self.geometry_digest = digest.hexdigest()

    def one_ring(self, triangle_id: int) -> tuple[int, ...]:
        triangle = self.triangles[int(triangle_id)]
        return tuple(sorted({face for vertex in triangle for face in self._vertex_faces[int(vertex)]}))

    def query(self, point_world) -> LocalSurfaceHit:
        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        ranked = []
        for triangle_id, indices in enumerate(self.triangles):
            nearest, barycentric = closest_point_on_triangle(point, self.vertices[indices])
            delta = nearest - point
            ranked.append(
                (
                    float(delta @ delta),
                    int(triangle_id),
                    tuple(float(value) for value in barycentric),
                    nearest,
                )
            )
        _, triangle_id, _, nearest = min(ranked, key=lambda item: item[:3])
        ring = self.one_ring(triangle_id)
        normal = normalized(self._inward_area_vectors[np.asarray(ring, dtype=np.int64)].sum(axis=0))
        return LocalSurfaceHit(nearest, triangle_id, ring, normal, self.geometry_digest)


class FlatSurfaceQuery(TriangleMeshSurfaceQuery):
    def __init__(self, *, vertices, triangles) -> None:
        super().__init__(vertices, triangles, inward_sign=1)

    @classmethod
    def regular_plane(cls, *, half_extent_m: float = 1.0, cells_per_side: int = 8):
        if half_extent_m <= 0.0 or cells_per_side < 1:
            raise ValueError("flat plane dimensions must be positive")
        coordinates = np.linspace(-float(half_extent_m), float(half_extent_m), int(cells_per_side) + 1)
        vertices = np.asarray([[x, y, 0.0] for y in coordinates for x in coordinates])
        stride = len(coordinates)
        triangles = []
        for row in range(int(cells_per_side)):
            for column in range(int(cells_per_side)):
                lower_left = row * stride + column
                triangles.extend(
                    [
                        [lower_left, lower_left + 1, lower_left + stride + 1],
                        [lower_left, lower_left + stride + 1, lower_left + stride],
                    ]
                )
        return cls(vertices=vertices, triangles=np.asarray(triangles, dtype=np.int64))


class StomachSurfaceQuery(TriangleMeshSurfaceQuery):
    """Adapter for world-space arrays produced by the verified stage mesh reader."""

    def __init__(self, *, vertices, triangles, inward_sign: int) -> None:
        super().__init__(vertices, triangles, inward_sign=inward_sign)

    @classmethod
    def from_reference(cls, reference, *, inward_sign: int):
        return cls(
            vertices=np.asarray(reference.vertices_world, dtype=np.float64),
            triangles=np.asarray(reference.triangles, dtype=np.int64),
            inward_sign=inward_sign,
        )

