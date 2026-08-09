"""Deterministic preprocessing of explicitly selected stomach surface meshes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class MeshInput:
    """One USD mesh converted into plain immutable preprocessing inputs."""

    prim_path: str
    vertices: np.ndarray
    face_vertex_counts: np.ndarray
    face_vertex_indices: np.ndarray
    world_transform: np.ndarray


@dataclass(frozen=True)
class ReferenceMesh:
    """Welded world-space coverage surface and triangle incidence."""

    vertices_world: np.ndarray
    triangles: np.ndarray
    incident_triangles: tuple[tuple[int, ...], ...]
    selected_prim_paths: tuple[str, ...]
    weld_tolerance_m: float
    geometry_sha256: str


def _validated(mesh: MeshInput) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    counts = np.asarray(mesh.face_vertex_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(mesh.face_vertex_indices, dtype=np.int64).reshape(-1)
    transform = np.asarray(mesh.world_transform, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{mesh.prim_path}: vertices must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError(f"{mesh.prim_path}: world transform must have shape (4, 4)")
    if not np.isfinite(vertices).all() or not np.isfinite(transform).all():
        raise ValueError(f"{mesh.prim_path}: nonfinite geometry or transform")
    if counts.size == 0 or np.any(counts != 3):
        raise ValueError(f"{mesh.prim_path}: only triangular faces are supported")
    if int(counts.sum()) != indices.size:
        raise ValueError(f"{mesh.prim_path}: face counts do not match index count")
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(vertices)):
        raise ValueError(f"{mesh.prim_path}: face index is outside the vertex array")
    return vertices, counts, indices, transform


def _transform_points(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Gf.Matrix4d stores translation in row 3 and transforms row vectors.
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    transformed = homogeneous @ matrix
    if np.any(np.abs(transformed[:, 3]) < np.finfo(np.float64).eps):
        raise ValueError("world transform produced a zero homogeneous coordinate")
    return transformed[:, :3] / transformed[:, 3:4]


def _weld(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("weld tolerance must be finite and positive")
    order = sorted(range(len(points)), key=lambda index: (*points[index].tolist(), index))
    cells: dict[tuple[int, int, int], list[int]] = {}
    welded: list[np.ndarray] = []
    remap = np.empty(len(points), dtype=np.int64)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    for original_index in order:
        point = points[original_index]
        cell = tuple(np.floor(point / tolerance).astype(np.int64).tolist())
        matches: list[int] = []
        for offset in offsets:
            neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
            for welded_index in cells.get(neighbor, ()):
                if float(np.linalg.norm(point - welded[welded_index])) <= tolerance:
                    matches.append(welded_index)
        if matches:
            welded_index = min(matches)
        else:
            welded_index = len(welded)
            welded.append(point.copy())
            cells.setdefault(cell, []).append(welded_index)
        remap[original_index] = welded_index
    return np.asarray(welded, dtype=np.float64), remap


def _hash(
    vertices: np.ndarray,
    triangles: np.ndarray,
    prim_paths: tuple[str, ...],
    tolerance: float,
) -> str:
    payload = {
        "vertices_world": np.round(vertices, 12).tolist(),
        "triangles": triangles.tolist(),
        "selected_prim_paths": list(prim_paths),
        "weld_tolerance_m": tolerance,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preprocess_reference_mesh(
    meshes: Iterable[MeshInput],
    selected_prim_paths: Sequence[str],
    weld_tolerance_m: float = 1.0e-6,
) -> ReferenceMesh:
    """Build a deterministic reference mesh from an explicit prim selection."""
    selected_paths = tuple(sorted(set(str(path) for path in selected_prim_paths)))
    if not selected_paths:
        raise ValueError("selected prim path list is empty")
    by_path = {mesh.prim_path: mesh for mesh in meshes}
    missing = [path for path in selected_paths if path not in by_path]
    if missing:
        raise ValueError(f"selected prims not found: {missing}")

    point_blocks: list[np.ndarray] = []
    triangle_blocks: list[np.ndarray] = []
    point_offset = 0
    for path in selected_paths:
        vertices, _, indices, transform = _validated(by_path[path])
        world = _transform_points(vertices, transform)
        point_blocks.append(world)
        triangles = indices.reshape(-1, 3) + point_offset
        triangle_blocks.append(triangles)
        point_offset += len(world)
    all_points = np.concatenate(point_blocks, axis=0)
    source_triangles = np.concatenate(triangle_blocks, axis=0)
    welded, remap = _weld(all_points, weld_tolerance_m)
    triangles = remap[source_triangles]
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    ):
        raise ValueError("welding produced a degenerate triangle")

    incident: list[list[int]] = [[] for _ in range(len(welded))]
    for triangle_index, triangle in enumerate(triangles):
        for vertex_index in triangle:
            incident[int(vertex_index)].append(triangle_index)
    incident_tuple = tuple(tuple(sorted(set(values))) for values in incident)
    welded.setflags(write=False)
    triangles.setflags(write=False)
    return ReferenceMesh(
        vertices_world=welded,
        triangles=triangles,
        incident_triangles=incident_tuple,
        selected_prim_paths=selected_paths,
        weld_tolerance_m=float(weld_tolerance_m),
        geometry_sha256=_hash(welded, triangles, selected_paths, float(weld_tolerance_m)),
    )
