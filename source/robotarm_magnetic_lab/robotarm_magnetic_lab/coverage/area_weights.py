"""Area-weighted target vertices for stomach coverage."""

from __future__ import annotations

import hashlib

import numpy as np

from .reference_mesh import ReferenceMesh


def target_vertex_area_weights(
    reference: ReferenceMesh,
    target_triangle_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Distribute one third of each selected triangle area to its vertices."""
    triangles = np.asarray(reference.triangles, dtype=np.int64)
    if target_triangle_indices is None:
        selected = np.arange(len(triangles), dtype=np.int64)
    else:
        selected = np.asarray(target_triangle_indices, dtype=np.int64).reshape(-1)
    if len(selected) == 0:
        raise ValueError("coverage target contains no triangles")
    if int(selected.min()) < 0 or int(selected.max()) >= len(triangles):
        raise ValueError("coverage target triangle index is out of range")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("coverage target triangle indices must be unique")
    target_triangles = triangles[selected]
    vertices = np.asarray(reference.vertices_world, dtype=np.float64)
    points = vertices[target_triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("coverage target contains a non-finite or degenerate triangle")
    weights = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(weights, target_triangles.reshape(-1), np.repeat(areas / 3.0, 3))
    if not np.isclose(weights.sum(), areas.sum(), rtol=1.0e-12, atol=1.0e-15):
        raise RuntimeError("vertex area weights do not conserve target triangle area")
    weights.setflags(write=False)
    return weights


def weights_sha256(weights: np.ndarray) -> str:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    return hashlib.sha256(values.tobytes()).hexdigest()
