"""Fit stomach tube openings and derive a deterministic horizontal pose."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


_EPSILON = 1.0e-12
_MAX_OPENING_PLANARITY_M = 5.0e-5


@dataclass(frozen=True)
class OpeningFit:
    """Planar circular approximation of one terminal mesh opening."""

    center: np.ndarray
    axis: np.ndarray
    radius_m: float
    planarity_m: float
    vertex_count: int

    def to_json(self) -> dict:
        record = asdict(self)
        record["center"] = self.center.tolist()
        record["axis"] = self.axis.tolist()
        return record


def _finite_array(value: np.ndarray, *, name: str, columns: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite Nx{columns} array")
    return array


def extract_boundary_components(vertices: np.ndarray, triangles: np.ndarray) -> list[np.ndarray]:
    """Return connected vertex components formed by single-use triangle edges."""

    vertices = _finite_array(vertices, name="vertices", columns=3).astype(np.float64, copy=False)
    triangles = _finite_array(triangles, name="triangles", columns=3).astype(np.int64, copy=False)
    if triangles.size == 0:
        return []
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangle index is outside the vertex array")

    edges = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0
    )
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    adjacency: dict[int, list[int]] = {}
    for first, second in boundary_edges:
        a, b = int(first), int(second)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    seen: set[int] = set()
    components: list[np.ndarray] = []
    for seed in sorted(adjacency):
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    components.sort(key=lambda item: (-len(item), tuple(item[:1])))
    return components


def fit_opening(vertices: np.ndarray, component: np.ndarray) -> OpeningFit:
    """Fit a plane and mean radius to one boundary component."""

    vertices = _finite_array(vertices, name="vertices", columns=3).astype(np.float64, copy=False)
    indices = np.asarray(component, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 3:
        raise ValueError("opening component must contain at least three vertices")
    if indices.min() < 0 or indices.max() >= len(vertices):
        raise ValueError("opening component index is outside the vertex array")
    points = vertices[indices]
    center = points.mean(axis=0)
    centered = points - center
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    axis = basis[-1]
    axis /= np.linalg.norm(axis)
    signed = centered @ axis
    planar = centered - np.outer(signed, axis)
    radius = float(np.linalg.norm(planar, axis=1).mean())
    return OpeningFit(
        center=center,
        axis=axis,
        radius_m=radius,
        planarity_m=float(np.abs(signed).max()),
        vertex_count=len(indices),
    )


def select_tube_openings(
    vertices: np.ndarray,
    components: list[np.ndarray],
    mesh_centroid: np.ndarray,
) -> tuple[OpeningFit, OpeningFit]:
    """Select the two dominant planar boundaries and orient axes outward."""

    centroid = np.asarray(mesh_centroid, dtype=np.float64)
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ValueError("mesh_centroid must be a finite 3-vector")
    fits = [fit_opening(vertices, component) for component in components]
    candidates = [fit for fit in fits if fit.planarity_m <= _MAX_OPENING_PLANARITY_M]
    candidates.sort(key=lambda fit: (-fit.vertex_count, fit.planarity_m, *fit.center.tolist()))
    if len(candidates) < 2:
        raise ValueError("at least two planar boundary components are required")
    selected = []
    for fit in candidates[:2]:
        axis = fit.axis
        if float(axis @ (fit.center - centroid)) < 0.0:
            axis = -axis
        selected.append(
            OpeningFit(
                center=fit.center,
                axis=axis,
                radius_m=fit.radius_m,
                planarity_m=fit.planarity_m,
                vertex_count=fit.vertex_count,
            )
        )
    return selected[0], selected[1]


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized XYZW quaternion."""

    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return the active rotation matrix for a normalized XYZW quaternion."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must be a finite XYZW 4-vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= _EPSILON:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def solve_horizontal_axes(
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    body_center: np.ndarray,
) -> np.ndarray:
    """Return a local-to-world XYZW rotation that makes both axes horizontal."""

    axes = []
    for name, value in (("axis_a", axis_a), ("axis_b", axis_b)):
        axis = np.asarray(value, dtype=np.float64)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= _EPSILON:
            raise ValueError(f"{name} must be a finite non-zero 3-vector")
        axes.append(axis / np.linalg.norm(axis))
    first, second = axes
    body = np.asarray(body_center, dtype=np.float64)
    if body.shape != (3,) or not np.isfinite(body).all():
        raise ValueError("body_center must be a finite 3-vector")

    vertical = np.cross(first, second)
    if np.linalg.norm(vertical) <= 1.0e-8:
        vertical = body - first * float(body @ first)
        if np.linalg.norm(vertical) <= 1.0e-8:
            helper = np.array([0.0, 0.0, 1.0])
            if abs(float(first @ helper)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            vertical = helper - first * float(helper @ first)
    vertical /= np.linalg.norm(vertical)
    if float(vertical @ body) > 0.0:
        vertical = -vertical

    world_x_local = first - vertical * float(first @ vertical)
    world_x_local /= np.linalg.norm(world_x_local)
    world_y_local = np.cross(vertical, world_x_local)
    world_y_local /= np.linalg.norm(world_y_local)
    rotation = np.vstack((world_x_local, world_y_local, vertical))
    if np.linalg.det(rotation) < 0.0:
        rotation[1] *= -1.0

    transformed = np.vstack((rotation @ first, rotation @ second))
    elevations = np.degrees(np.arcsin(np.clip(np.abs(transformed[:, 2]), 0.0, 1.0)))
    if float(elevations.max()) > 1.0:
        raise ValueError(f"opening axes cannot both be horizontal: {elevations.tolist()}")
    return _matrix_to_quaternion_xyzw(rotation)


def load_new_stomach_manifest(path: str | Path) -> dict:
    """Load and minimally validate the immutable new-stomach asset manifest."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "asset", "geometry", "boundary_components", "selected_openings"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"new stomach manifest is missing keys: {sorted(missing)}")
    return manifest

