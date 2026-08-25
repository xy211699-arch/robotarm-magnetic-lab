"""World-axis-aligned stomach entrance region calibration primitives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .reference_mesh import ReferenceMesh


ENTRANCE_REGION_SCHEMA = "robotarm_magnetic_lab.stomach_entrance_region"
ENTRANCE_REGION_VERSION = 1


@dataclass(frozen=True)
class EntranceSelection:
    """Triangle selection and measurements for one calibrated AABB."""

    triangle_indices: np.ndarray
    triangle_count: int
    area_m2: float
    connected_components: int


def _triangle_overlaps_aabb(triangle: np.ndarray, center: np.ndarray, half_size: np.ndarray) -> bool:
    """Exact triangle/AABB overlap using the separating-axis theorem."""
    vertices = np.asarray(triangle, dtype=np.float64).reshape(3, 3) - center
    edges = (vertices[1] - vertices[0], vertices[2] - vertices[1], vertices[0] - vertices[2])
    axes = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.cross(edges[0], edges[1]),
    ]
    box_axes = np.eye(3, dtype=np.float64)
    axes.extend(np.cross(box_axis, edge) for box_axis in box_axes for edge in edges)
    for axis in axes:
        if float(np.dot(axis, axis)) <= 1.0e-24:
            continue
        projection = vertices @ axis
        radius = float(np.dot(half_size, np.abs(axis)))
        if float(projection.min()) > radius or float(projection.max()) < -radius:
            return False
    return True


def triangles_intersecting_aabb(
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    center_world_m: np.ndarray,
    size_world_m: np.ndarray,
) -> np.ndarray:
    """Return indices of all triangles that intersect a positive-sized AABB."""
    vertices = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    center = np.asarray(center_world_m, dtype=np.float64).reshape(3)
    size = np.asarray(size_world_m, dtype=np.float64).reshape(3)
    if not np.isfinite(center).all() or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ValueError("AABB center must be finite and all sizes must be finite and positive")
    half_size = 0.5 * size
    selected = [
        index
        for index, face in enumerate(faces)
        if _triangle_overlaps_aabb(vertices[face], center, half_size)
    ]
    return np.asarray(selected, dtype=np.int64)


def _component_count(triangles: np.ndarray, selected_indices: np.ndarray) -> int:
    """Count face components connected by a complete shared mesh edge."""
    selected = [int(index) for index in np.asarray(selected_indices, dtype=np.int64).reshape(-1)]
    if not selected:
        return 0
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index in selected:
        face = triangles[face_index]
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            edge_faces.setdefault(edge, []).append(face_index)
    adjacency = {face_index: set() for face_index in selected}
    for face_indices in edge_faces.values():
        for face_index in face_indices:
            adjacency[face_index].update(other for other in face_indices if other != face_index)
    components = 0
    remaining = set(selected)
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            neighbors = adjacency[current] & remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
    return components


def select_entrance_triangles(
    reference: ReferenceMesh,
    center_world_m: np.ndarray,
    size_world_m: np.ndarray,
) -> EntranceSelection:
    """Select intersecting faces and report area and edge-connected components."""
    selected = triangles_intersecting_aabb(
        reference.vertices_world,
        reference.triangles,
        center_world_m,
        size_world_m,
    )
    faces = reference.triangles[selected]
    if len(faces):
        points = reference.vertices_world[faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
        )
        area_m2 = float(areas.sum())
    else:
        area_m2 = 0.0
    return EntranceSelection(
        triangle_indices=selected,
        triangle_count=int(len(selected)),
        area_m2=area_m2,
        connected_components=_component_count(reference.triangles, selected),
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def entrance_region_record(
    reference: ReferenceMesh,
    center_world_m: np.ndarray,
    size_world_m: np.ndarray,
    selection: EntranceSelection,
    *,
    stomach_asset_identifier: str,
    operator_confirmation: str = "pending",
) -> dict[str, Any]:
    """Build a deterministic, versioned record whose hash invalidates derived poses."""
    payload: dict[str, Any] = {
        "schema": ENTRANCE_REGION_SCHEMA,
        "version": ENTRANCE_REGION_VERSION,
        "coordinate_frame": "world",
        "selection_rule": "triangle_aabb_intersection_sat",
        "connectivity_rule": "shared_complete_triangle_edge",
        "center_world_m": np.asarray(center_world_m, dtype=np.float64).reshape(3).tolist(),
        "size_world_m": np.asarray(size_world_m, dtype=np.float64).reshape(3).tolist(),
        "stomach_asset_identifier": str(stomach_asset_identifier),
        "stomach_geometry_sha256": reference.geometry_sha256,
        "selected_surface_prim_paths": list(reference.selected_prim_paths),
        "selected_triangle_count": selection.triangle_count,
        "selected_area_m2": selection.area_m2,
        "connected_component_count": selection.connected_components,
        "operator_confirmation": str(operator_confirmation),
    }
    # Confirmation and save time are execution evidence, not geometry inputs.
    # Keeping them outside this hash makes the derived-pose invalidation key
    # stable when an operator confirms an otherwise unchanged box.
    hash_payload = {
        key: value for key, value in payload.items() if key != "operator_confirmation"
    }
    payload["config_sha256"] = _canonical_hash(hash_payload)
    payload["saved_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def save_entrance_region(path: Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
