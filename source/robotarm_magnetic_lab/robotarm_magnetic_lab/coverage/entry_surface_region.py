"""Stable capsule anchor and connected stomach-surface entry-region geometry."""

from __future__ import annotations

import hashlib
import heapq
from itertools import combinations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .reference_mesh import ReferenceMesh


ANCHOR_SCHEMA = "robotarm_magnetic_lab.task009b_entry_anchor"
REGION_SCHEMA = "robotarm_magnetic_lab.task009b_entry_region"
CONFIG_VERSION = 1
ENTRY_RADII_M = tuple(float(radius_mm) / 1000.0 for radius_mm in range(10, 81, 5))


@dataclass(frozen=True)
class ClosestSurfacePoint:
    triangle_index: int
    point_world_m: np.ndarray
    distance_m: float


@dataclass(frozen=True)
class SurfaceRegion:
    triangle_indices: np.ndarray
    vertex_indices: np.ndarray
    area_m2: float
    connected_components: int
    radius_m: float


def closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Return the Euclidean closest point on a triangle (Ericson regions)."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    a, b, c = np.asarray(triangle, dtype=np.float64).reshape(3, 3)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()
    bp = p - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        edge = c - b
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * edge
    denominator = 1.0 / (va + vb + vc)
    v, w = vb * denominator, vc * denominator
    return a + ab * v + ac * w


def nearest_surface_point(reference: ReferenceMesh, point_world_m: np.ndarray) -> ClosestSurfacePoint:
    """Find the nearest triangle and true closest surface point, never a nearest vertex proxy."""
    point = np.asarray(point_world_m, dtype=np.float64).reshape(3)
    best_index = -1
    best_point = None
    best_squared = np.inf
    for index, face in enumerate(reference.triangles):
        candidate = closest_point_on_triangle(point, reference.vertices_world[face])
        squared = float(np.dot(candidate - point, candidate - point))
        if squared < best_squared:
            best_index, best_point, best_squared = index, candidate, squared
    if best_point is None:
        raise ValueError("reference stomach surface has no triangles")
    return ClosestSurfacePoint(best_index, best_point, float(np.sqrt(best_squared)))


def shared_edge_adjacency(reference: ReferenceMesh) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Build deterministic face adjacency weighted by centroid-to-centroid surface steps."""
    faces = reference.triangles
    centroids = reference.vertices_world[faces].mean(axis=1)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(face_index)
    adjacency: list[dict[int, float]] = [dict() for _ in range(len(faces))]
    for face_indices in edge_faces.values():
        for first, second in combinations(sorted(set(face_indices)), 2):
            weight = float(np.linalg.norm(centroids[first] - centroids[second]))
            if weight <= 0.0:
                continue
            adjacency[first][second] = min(adjacency[first].get(second, np.inf), weight)
            adjacency[second][first] = min(adjacency[second].get(first, np.inf), weight)
    return tuple(tuple(sorted(values.items())) for values in adjacency)


def geodesic_face_distances(
    adjacency: tuple[tuple[tuple[int, float], ...], ...], seed_triangle: int
) -> np.ndarray:
    """Compute approximate intrinsic distances over the shared-edge face graph."""
    if seed_triangle < 0 or seed_triangle >= len(adjacency):
        raise ValueError("seed triangle is outside the adjacency graph")
    distances = np.full(len(adjacency), np.inf, dtype=np.float64)
    distances[seed_triangle] = 0.0
    queue: list[tuple[float, int]] = [(0.0, seed_triangle)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        for neighbor, weight in adjacency[current]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _component_count(faces: np.ndarray, selected: np.ndarray) -> int:
    selected_set = {int(value) for value in np.asarray(selected).reshape(-1)}
    if not selected_set:
        return 0
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index in selected_set:
        face = faces[face_index]
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(face_index)
    adjacency = {face_index: set() for face_index in selected_set}
    for values in edge_faces.values():
        for face_index in values:
            adjacency[face_index].update(other for other in values if other != face_index)
    count = 0
    while selected_set:
        count += 1
        stack = [selected_set.pop()]
        while stack:
            neighbors = adjacency[stack.pop()] & selected_set
            selected_set.difference_update(neighbors)
            stack.extend(neighbors)
    return count


def surface_region_from_distances(
    reference: ReferenceMesh, distances_m: np.ndarray, radius_m: float
) -> SurfaceRegion:
    """Select the connected intrinsic disk at a frozen geodesic radius."""
    if radius_m <= 0.0 or not np.isfinite(radius_m):
        raise ValueError("entry radius must be finite and positive")
    distances = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    if len(distances) != len(reference.triangles):
        raise ValueError("geodesic distance count must equal triangle count")
    selected = np.flatnonzero(distances <= radius_m).astype(np.int64)
    faces = reference.triangles[selected]
    vertices = np.unique(faces.reshape(-1)) if len(faces) else np.empty(0, dtype=np.int64)
    if len(faces):
        points = reference.vertices_world[faces]
        area = float(
            (0.5 * np.linalg.norm(
                np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
            )).sum()
        )
    else:
        area = 0.0
    return SurfaceRegion(
        triangle_indices=selected,
        vertex_indices=vertices,
        area_m2=area,
        connected_components=_component_count(reference.triangles, selected),
        radius_m=float(radius_m),
    )


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def anchor_record(
    *,
    default_pose_xyzw: np.ndarray,
    release_pose_xyzw: np.ndarray,
    settled_pose_xyzw: np.ndarray,
    stable_detection: dict[str, Any],
    stomach_geometry_sha256: str,
    capsule_asset_identifier: str,
) -> dict[str, Any]:
    payload = {
        "schema": ANCHOR_SCHEMA,
        "version": CONFIG_VERSION,
        "default_pose_world_xyzw": np.asarray(default_pose_xyzw, dtype=np.float64).reshape(7).tolist(),
        "release_pose_world_xyzw": np.asarray(release_pose_xyzw, dtype=np.float64).reshape(7).tolist(),
        "settled_pose_world_xyzw": np.asarray(settled_pose_xyzw, dtype=np.float64).reshape(7).tolist(),
        "stable_detection": stable_detection,
        "stomach_geometry_sha256": str(stomach_geometry_sha256),
        "capsule_asset_identifier": str(capsule_asset_identifier),
        "operator_confirmation": "confirmed_by_Y_key",
    }
    hash_payload = {key: value for key, value in payload.items() if key != "operator_confirmation"}
    payload["config_sha256"] = _hash(hash_payload)
    payload["saved_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def region_record(
    *,
    anchor_config_sha256: str,
    settled_pose_xyzw: np.ndarray,
    closest: ClosestSurfacePoint,
    region: SurfaceRegion,
    stomach_geometry_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": REGION_SCHEMA,
        "version": CONFIG_VERSION,
        "anchor_config_sha256": str(anchor_config_sha256),
        "settled_pose_world_xyzw": np.asarray(settled_pose_xyzw, dtype=np.float64).reshape(7).tolist(),
        "closest_surface_point_world_m": closest.point_world_m.tolist(),
        "capsule_to_surface_distance_m": closest.distance_m,
        "seed_triangle_index": closest.triangle_index,
        "geodesic_radius_m": region.radius_m,
        "selected_triangle_indices": region.triangle_indices.tolist(),
        "selected_vertex_indices": region.vertex_indices.tolist(),
        "selected_triangle_count": int(len(region.triangle_indices)),
        "selected_vertex_count": int(len(region.vertex_indices)),
        "selected_area_m2": region.area_m2,
        "connected_component_count": region.connected_components,
        "stomach_geometry_sha256": str(stomach_geometry_sha256),
        "operator_confirmation": "confirmed_by_Enter_key",
    }
    hash_payload = {key: value for key, value in payload.items() if key != "operator_confirmation"}
    payload["config_sha256"] = _hash(hash_payload)
    payload["saved_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def save_and_reload(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    if loaded != record:
        raise RuntimeError(f"saved configuration failed exact reload validation: {output}")
    return loaded


def load_and_validate(path: Path, expected_schema: str) -> dict[str, Any]:
    """Load a saved configuration and verify its schema and deterministic hash."""
    source = Path(path)
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("schema") != expected_schema:
        raise ValueError(
            f"configuration schema mismatch: expected {expected_schema!r}, "
            f"got {record.get('schema')!r}"
        )
    expected_hash = str(record.get("config_sha256", ""))
    hash_payload = {
        key: value
        for key, value in record.items()
        if key not in ("config_sha256", "saved_utc", "operator_confirmation")
    }
    actual_hash = _hash(hash_payload)
    if actual_hash != expected_hash:
        raise ValueError(
            f"configuration hash mismatch for {source}: {expected_hash} != {actual_hash}"
        )
    return record
