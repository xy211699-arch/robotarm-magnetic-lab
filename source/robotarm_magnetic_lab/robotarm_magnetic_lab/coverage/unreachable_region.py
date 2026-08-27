"""Frozen operator-authored unreachable stomach-surface masks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .area_weights import target_vertex_area_weights, weights_sha256
from .entry_surface_region import (
    geodesic_face_distances,
    shared_edge_adjacency,
    surface_region_from_distances,
)
from .reference_mesh import ReferenceMesh


UNREACHABLE_REGION_SCHEMA = "robotarm_magnetic_lab.task009b_unreachable_region"
UNREACHABLE_REGION_VERSION = 1
UNREACHABLE_RADII_M = tuple(float(radius_mm) / 1000.0 for radius_mm in range(10, 81, 5))


@dataclass(frozen=True)
class UnreachableSeed:
    """One operator-selected surface seed and its intrinsic expansion radius."""

    triangle_index: int
    point_world_m: np.ndarray
    radius_m: float


@dataclass(frozen=True)
class UnreachableMask:
    """Validated union mask and the complementary reachable target."""

    excluded_triangle_indices: np.ndarray
    reachable_triangle_indices: np.ndarray
    excluded_vertex_indices: np.ndarray
    excluded_area_m2: float
    reachable_area_m2: float
    excluded_area_fraction: float
    config_sha256: str


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _triangle_areas(reference: ReferenceMesh) -> np.ndarray:
    points = reference.vertices_world[reference.triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("stomach target contains a non-finite or degenerate triangle")
    return areas


def build_unreachable_mask(
    reference: ReferenceMesh,
    seeds: Iterable[UnreachableSeed],
) -> tuple[UnreachableMask, tuple[np.ndarray, ...]]:
    """Expand every seed on the shared-edge graph and return their exact union."""
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("at least one unreachable-region seed is required")
    adjacency = shared_edge_adjacency(reference)
    per_seed_faces: list[np.ndarray] = []
    for seed in seed_values:
        triangle_index = int(seed.triangle_index)
        radius_m = float(seed.radius_m)
        if triangle_index < 0 or triangle_index >= len(reference.triangles):
            raise ValueError("unreachable seed triangle is outside the stomach mesh")
        if radius_m not in UNREACHABLE_RADII_M:
            raise ValueError("unreachable seed radius must use a frozen 5 mm level from 10 to 80 mm")
        distances = geodesic_face_distances(adjacency, triangle_index)
        per_seed_faces.append(
            surface_region_from_distances(reference, distances, radius_m).triangle_indices
        )
    excluded = np.unique(np.concatenate(per_seed_faces)).astype(np.int64)
    all_faces = np.arange(len(reference.triangles), dtype=np.int64)
    reachable = np.setdiff1d(all_faces, excluded, assume_unique=True)
    if len(reachable) == 0:
        raise ValueError("unreachable mask excludes the complete stomach target")
    areas = _triangle_areas(reference)
    excluded_area = float(areas[excluded].sum())
    reachable_area = float(areas[reachable].sum())
    total_area = excluded_area + reachable_area
    excluded_vertices = np.unique(reference.triangles[excluded].reshape(-1)).astype(np.int64)
    empty_hash = ""
    mask = UnreachableMask(
        excluded_triangle_indices=excluded,
        reachable_triangle_indices=reachable,
        excluded_vertex_indices=excluded_vertices,
        excluded_area_m2=excluded_area,
        reachable_area_m2=reachable_area,
        excluded_area_fraction=excluded_area / total_area,
        config_sha256=empty_hash,
    )
    return mask, tuple(per_seed_faces)


def unreachable_region_record(
    *,
    reference: ReferenceMesh,
    seeds: Iterable[UnreachableSeed],
    reason: str,
    operator: str,
) -> dict[str, Any]:
    """Create a reviewable, immutable unreachable-region configuration."""
    seed_values = tuple(seeds)
    mask, per_seed_faces = build_unreachable_mask(reference, seed_values)
    raw_weights = target_vertex_area_weights(reference)
    reachable_weights = target_vertex_area_weights(
        reference, mask.reachable_triangle_indices
    )
    reason_value = str(reason).strip()
    operator_value = str(operator).strip()
    if not reason_value:
        raise ValueError("an anatomical/physical unreachable reason is required")
    if not operator_value:
        raise ValueError("operator identifier is required")
    payload: dict[str, Any] = {
        "schema": UNREACHABLE_REGION_SCHEMA,
        "version": UNREACHABLE_REGION_VERSION,
        "status": "frozen",
        "operator": operator_value,
        "reason": reason_value,
        "selection_policy": (
            "operator-selected physical/anatomical unreachable surfaces only; "
            "current-controller failures are not sufficient"
        ),
        "stomach_geometry_sha256": reference.geometry_sha256,
        "seeds": [
            {
                "seed_id": index,
                "surface_point_world_m": np.asarray(seed.point_world_m, dtype=np.float64)
                .reshape(3)
                .tolist(),
                "seed_triangle_index": int(seed.triangle_index),
                "geodesic_radius_m": float(seed.radius_m),
                "selected_triangle_indices": faces.tolist(),
                "selected_triangle_count": int(len(faces)),
            }
            for index, (seed, faces) in enumerate(zip(seed_values, per_seed_faces, strict=True))
        ],
        "excluded_triangle_indices": mask.excluded_triangle_indices.tolist(),
        "excluded_vertex_indices": mask.excluded_vertex_indices.tolist(),
        "excluded_triangle_count": int(len(mask.excluded_triangle_indices)),
        "excluded_vertex_count": int(len(mask.excluded_vertex_indices)),
        "excluded_area_m2": mask.excluded_area_m2,
        "excluded_area_fraction": mask.excluded_area_fraction,
        "reachable_triangle_indices": mask.reachable_triangle_indices.tolist(),
        "reachable_triangle_count": int(len(mask.reachable_triangle_indices)),
        "reachable_vertex_count": int(np.count_nonzero(reachable_weights > 0.0)),
        "reachable_area_m2": mask.reachable_area_m2,
        "raw_target_triangle_count": int(len(reference.triangles)),
        "raw_target_vertex_count": int(len(reference.vertices_world)),
        "raw_target_area_m2": float(raw_weights.sum()),
        "raw_vertex_weights_sha256": weights_sha256(raw_weights),
        "reachable_vertex_weights_sha256": weights_sha256(reachable_weights),
    }
    payload["config_sha256"] = _hash(payload)
    payload["saved_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def seeds_from_record(record: dict[str, Any]) -> tuple[UnreachableSeed, ...]:
    """Restore editable seeds from a validated or newly-created record."""
    return tuple(
        UnreachableSeed(
            triangle_index=int(item["seed_triangle_index"]),
            point_world_m=np.asarray(item["surface_point_world_m"], dtype=np.float64),
            radius_m=float(item["geodesic_radius_m"]),
        )
        for item in record["seeds"]
    )


def save_and_reload_unreachable(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    if loaded != record:
        raise RuntimeError(f"saved unreachable configuration failed reload: {output}")
    return loaded


def load_unreachable_mask(path: Path, reference: ReferenceMesh) -> tuple[dict[str, Any], UnreachableMask]:
    """Validate hashes and recompute the mask instead of trusting stored face lists."""
    source = Path(path)
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("schema") != UNREACHABLE_REGION_SCHEMA:
        raise ValueError(f"unreachable-region schema mismatch: {record.get('schema')!r}")
    if record.get("status") != "frozen":
        raise ValueError("unreachable-region configuration is not frozen")
    if record.get("stomach_geometry_sha256") != reference.geometry_sha256:
        raise ValueError("unreachable-region configuration does not match stomach geometry")
    expected_hash = str(record.get("config_sha256", ""))
    hash_payload = {
        key: value for key, value in record.items() if key not in ("config_sha256", "saved_utc")
    }
    if _hash(hash_payload) != expected_hash:
        raise ValueError("unreachable-region configuration hash mismatch")
    seeds = seeds_from_record(record)
    computed, _ = build_unreachable_mask(reference, seeds)
    if computed.excluded_triangle_indices.tolist() != record["excluded_triangle_indices"]:
        raise ValueError("stored unreachable triangle union differs from recomputed geodesic union")
    if computed.reachable_triangle_indices.tolist() != record["reachable_triangle_indices"]:
        raise ValueError("stored reachable triangle complement differs from recomputed result")
    reachable_weights = target_vertex_area_weights(reference, computed.reachable_triangle_indices)
    if weights_sha256(reachable_weights) != record["reachable_vertex_weights_sha256"]:
        raise ValueError("stored reachable area weights differ from recomputed weights")
    mask = UnreachableMask(
        excluded_triangle_indices=computed.excluded_triangle_indices,
        reachable_triangle_indices=computed.reachable_triangle_indices,
        excluded_vertex_indices=computed.excluded_vertex_indices,
        excluded_area_m2=computed.excluded_area_m2,
        reachable_area_m2=computed.reachable_area_m2,
        excluded_area_fraction=computed.excluded_area_fraction,
        config_sha256=expected_hash,
    )
    return record, mask
