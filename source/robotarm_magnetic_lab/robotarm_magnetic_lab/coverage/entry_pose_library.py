"""Deterministic geometry and records for the TASK-009B entry pose library."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from robotarm_magnetic_lab.runtime.quaternion_conventions import normalized_xyzw

from .reference_mesh import ReferenceMesh
from .visibility import triangle_normals


POSE_LIBRARY_SCHEMA = "robotarm_magnetic_lab.task009b_entry_pose_library"
POSE_LIBRARY_MANIFEST_SCHEMA = "robotarm_magnetic_lab.task009b_entry_pose_library_manifest"
POSE_LIBRARY_VERSION = 1
SPLIT_COUNTS = {"train": 1000, "validation": 100, "test": 100}
SPLIT_BASE_SEEDS = {"train": 910_001, "validation": 920_001, "test": 930_001}
LIVE_RELOAD_COUNT_PER_SPLIT = 20
CAPSULE_RADIUS_M = 0.0065
CAPSULE_CYLINDER_HEIGHT_M = 0.012
INITIAL_SURFACE_CLEARANCE_M = 0.0003
INWARD_NORMAL_SIGN = -1
MIN_UNORIENTED_AXIS_ANGLE_DEG = 45.0


@dataclass(frozen=True)
class SurfacePoseCandidate:
    pose_world_xyzw: np.ndarray
    surface_point_world_m: np.ndarray
    surface_normal_world: np.ndarray
    triangle_index: int
    barycentric: np.ndarray
    tangent_azimuth_rad: float
    camera_end_sign: int
    roll_rad: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_candidate_seed(split: str, attempt_index: int) -> int:
    if split not in SPLIT_BASE_SEEDS:
        raise ValueError(f"unknown pose-library split {split!r}")
    attempt = int(attempt_index)
    if attempt < 0:
        raise ValueError("attempt_index must be non-negative")
    sequence = np.random.SeedSequence([SPLIT_BASE_SEEDS[split], attempt])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def triangle_areas(reference: ReferenceMesh, triangle_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(triangle_indices, dtype=np.int64).reshape(-1)
    if len(indices) == 0:
        raise ValueError("entry region contains no triangles")
    if int(indices.min()) < 0 or int(indices.max()) >= len(reference.triangles):
        raise ValueError("entry triangle index is outside the reference mesh")
    points = reference.vertices_world[reference.triangles[indices]]
    areas = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("entry region contains a non-finite or degenerate triangle")
    return areas


def _matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(rotation).all() or not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1.0e-9
    ) or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-9):
        raise ValueError("candidate rotation matrix must be a finite proper rotation")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = normalized_xyzw(np.asarray([x, y, z, w], dtype=np.float64))
    return -quaternion if quaternion[3] < 0.0 else quaternion


def sample_surface_pose(
    reference: ReferenceMesh,
    entry_triangle_indices: np.ndarray,
    seed: int,
    *,
    inward_normal_sign: int = INWARD_NORMAL_SIGN,
    center_offset_m: float = CAPSULE_RADIUS_M + INITIAL_SURFACE_CLEARANCE_M,
) -> SurfacePoseCandidate:
    """Area-sample one tangent capsule pose from the confirmed entry region."""
    indices = np.asarray(entry_triangle_indices, dtype=np.int64).reshape(-1)
    areas = triangle_areas(reference, indices)
    rng = np.random.default_rng(int(seed))
    local_face = int(rng.choice(len(indices), p=areas / areas.sum()))
    triangle_index = int(indices[local_face])
    vertices = reference.vertices_world[reference.triangles[triangle_index]]
    first, second = rng.random(2)
    root = math.sqrt(float(first))
    barycentric = np.asarray(
        [1.0 - root, root * (1.0 - float(second)), root * float(second)],
        dtype=np.float64,
    )
    point = barycentric @ vertices
    normal = triangle_normals(reference, normal_sign=inward_normal_sign)[triangle_index]
    reference_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, reference_axis))) > 0.9:
        reference_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    tangent_first = np.cross(normal, reference_axis)
    tangent_first /= np.linalg.norm(tangent_first)
    tangent_second = np.cross(normal, tangent_first)
    tangent_second /= np.linalg.norm(tangent_second)
    azimuth = float(rng.uniform(0.0, math.pi))
    camera_end_sign = 1 if int(rng.integers(0, 2)) else -1
    capsule_axis = camera_end_sign * (
        math.cos(azimuth) * tangent_first + math.sin(azimuth) * tangent_second
    )
    capsule_axis /= np.linalg.norm(capsule_axis)
    x_zero = normal.copy()
    y_zero = np.cross(capsule_axis, x_zero)
    y_zero /= np.linalg.norm(y_zero)
    roll = float(rng.uniform(0.0, 2.0 * math.pi))
    x_axis = math.cos(roll) * x_zero + math.sin(roll) * y_zero
    y_axis = -math.sin(roll) * x_zero + math.cos(roll) * y_zero
    rotation = np.column_stack((x_axis, y_axis, capsule_axis))
    quaternion = _matrix_to_xyzw(rotation)
    center = point + float(center_offset_m) * normal
    return SurfacePoseCandidate(
        pose_world_xyzw=np.concatenate((center, quaternion)),
        surface_point_world_m=point,
        surface_normal_world=normal,
        triangle_index=triangle_index,
        barycentric=barycentric,
        tangent_azimuth_rad=azimuth,
        camera_end_sign=camera_end_sign,
        roll_rad=roll,
    )


def unoriented_axis_angle_deg(axis_world: np.ndarray) -> float:
    axis = np.asarray(axis_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or norm <= 1.0e-12:
        raise ValueError("capsule axis must be finite and nonzero")
    cosine = abs(float(axis[2] / norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pose_fingerprint(pose_world_xyzw: np.ndarray) -> str:
    pose = np.asarray(pose_world_xyzw, dtype=np.float64).reshape(7)
    return hashlib.sha256(np.round(pose, 9).tobytes()).hexdigest()


def stable_record_is_valid(record: dict[str, Any]) -> bool:
    pose = np.asarray(record.get("pose_world_xyzw", ()), dtype=np.float64)
    return bool(
        pose.shape == (7,)
        and np.isfinite(pose).all()
        and record.get("stable") is True
        and record.get("camera_inside_lumen") is True
        and float(record.get("unoriented_axis_angle_deg", -math.inf))
        >= MIN_UNORIENTED_AXIS_ANGLE_DEG
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> tuple[int, str]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return output.stat().st_size, file_sha256(output)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
