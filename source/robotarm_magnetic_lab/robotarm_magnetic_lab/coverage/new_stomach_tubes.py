"""Review-only tube masks for the new stomach's two open cylindrical ends.

This selects authored surface triangles, not a fitted solid or a visibility
shortcut.  The candidate is deliberately not a frozen coverage target until
the operator accepts the rendered overlay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference_mesh import ReferenceMesh


@dataclass(frozen=True)
class TubeCandidate:
    labels: np.ndarray
    inward_directions: np.ndarray
    directional_support: np.ndarray
    per_tube_face_indices: tuple[np.ndarray, np.ndarray]
    per_tube_component_counts: tuple[int, int]
    excluded_face_indices: np.ndarray
    reachable_face_indices: np.ndarray
    per_tube_area_m2: tuple[float, float]
    excluded_area_m2: float
    reachable_area_m2: float


def _face_geometry(reference: ReferenceMesh) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(reference.vertices_world, dtype=np.float64)[
        np.asarray(reference.triangles, dtype=np.int64)
    ]
    centers = points.mean(axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    if not np.isfinite(centers).all() or not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("new stomach contains invalid surface triangles")
    return centers, areas


def _component_count(triangles: np.ndarray, selected: np.ndarray) -> int:
    """Count edge-connected selected patches, not vertex-only touches."""
    parent = np.arange(len(selected), dtype=np.int64)
    edge_owner: dict[tuple[int, int], int] = {}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for local, face_index in enumerate(selected):
        first, second, third = triangles[face_index]
        for start, end in ((first, second), (second, third), (third, first)):
            key = (min(int(start), int(end)), max(int(start), int(end)))
            prior = edge_owner.setdefault(key, local)
            if prior != local:
                parent[root(local)] = root(prior)
    return len({root(index) for index in range(len(selected))})


def infer_inward_direction(
    face_centers: np.ndarray,
    opening_center: np.ndarray,
    opening_axis: np.ndarray,
    *,
    sampling_length_m: float = 0.010,
    sampling_radius_m: float = 0.012,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Choose the side with actual near-mouth tube wall triangles.

    The fitted opening axis's sign is not assumed to mean 'towards stomach'.
    A twofold support margin rejects ambiguous mouths.
    """
    centers = np.asarray(face_centers, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(opening_center, dtype=np.float64).reshape(3)
    axis = np.asarray(opening_axis, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(axis))
    if not np.isfinite(centers).all() or not np.isfinite(origin).all() or not np.isfinite(length):
        raise ValueError("non-finite tube calibration geometry")
    if length < 1.0e-12 or sampling_length_m <= 0 or sampling_radius_m <= 0:
        raise ValueError("invalid tube calibration axis or sampling dimensions")
    axis /= length
    relative = centers - origin
    axial = relative @ axis
    radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
    near = radial <= sampling_radius_m
    plus = int(np.count_nonzero(near & (axial > 0) & (axial <= sampling_length_m)))
    minus = int(np.count_nonzero(near & (axial < 0) & (axial >= -sampling_length_m)))
    if max(plus, minus) < 20 or max(plus, minus) < 2 * max(1, min(plus, minus)):
        raise ValueError(f"tube inward direction is ambiguous: +={plus}, -={minus}")
    return (axis if plus > minus else -axis), (plus, minus)


def candidate_tube_mask(
    reference: ReferenceMesh,
    opening_centers_world_m: np.ndarray,
    opening_axes_world: np.ndarray,
    *,
    tube_length_m: float = 0.025,
    radial_limit_m: float = 0.012,
    mouth_buffer_m: float = 0.001,
) -> TubeCandidate:
    """Select only the first cylindrical 25 mm from each open end by default."""
    openings = np.asarray(opening_centers_world_m, dtype=np.float64)
    axes = np.asarray(opening_axes_world, dtype=np.float64)
    if openings.shape != (2, 3) or axes.shape != (2, 3):
        raise ValueError("new stomach must have exactly two fitted tube mouths")
    if tube_length_m <= 0 or radial_limit_m <= 0 or mouth_buffer_m < 0:
        raise ValueError("tube dimensions must be positive")
    centers, areas = _face_geometry(reference)
    labels = np.zeros(len(centers), dtype=np.uint8)
    directions = []
    supports = []
    face_groups = []
    group_areas = []
    group_components = []
    for index, (mouth, axis) in enumerate(zip(openings, axes, strict=True), start=1):
        direction, support = infer_inward_direction(centers, mouth, axis)
        relative = centers - mouth
        axial = relative @ direction
        radial = np.linalg.norm(relative - axial[:, None] * direction, axis=1)
        selected = np.flatnonzero(
            (axial >= -mouth_buffer_m)
            & (axial <= tube_length_m)
            & (radial <= radial_limit_m)
        )
        if not len(selected) or np.any(labels[selected]):
            raise ValueError("tube selection is empty or overlaps the other tube")
        labels[selected] = index
        directions.append(direction)
        supports.append(support)
        face_groups.append(selected)
        group_areas.append(float(areas[selected].sum()))
        components = _component_count(reference.triangles, selected)
        if components != 1:
            raise ValueError(f"tube {index} selection has {components} disconnected patches")
        group_components.append(components)
    excluded = np.flatnonzero(labels != 0)
    reachable = np.flatnonzero(labels == 0)
    if not len(reachable):
        raise ValueError("tube mask excludes the entire stomach")
    return TubeCandidate(
        labels=labels,
        inward_directions=np.stack(directions),
        directional_support=np.asarray(supports, dtype=np.int64),
        per_tube_face_indices=(face_groups[0], face_groups[1]),
        per_tube_component_counts=(group_components[0], group_components[1]),
        excluded_face_indices=excluded,
        reachable_face_indices=reachable,
        per_tube_area_m2=(group_areas[0], group_areas[1]),
        excluded_area_m2=float(areas[excluded].sum()),
        reachable_area_m2=float(areas[reachable].sum()),
    )
