"""Review-only, surface-geodesic exclusion for the small duct marked by the operator.

The accepted new-stomach asset has a cluster of small circular mesh boundaries
in the annotated right-hand region.  Distances follow unique mesh edges, so
this selects the duct surface rather than every face inside a 2-D screenshot
rectangle or an arbitrary 3-D box.  It is not an active coverage target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from robotarm_magnetic_lab.geometry.stomach_orientation import (
    extract_boundary_components,
    fit_opening,
)

from .new_stomach_tubes import _component_count
from .reference_mesh import ReferenceMesh


@dataclass(frozen=True)
class MarkedDuctSeed:
    name: str
    center_from_min_m: tuple[float, float, float]
    radius_m: float
    surface_distance_m: float


# Centers and radii are measured on the accepted new-stomach collision mesh.
# The chosen distances end at the transition from the small cylindrical
# branches into the gastric wall; all three expansions form one connected patch.
MARKED_DUCT_SEEDS = (
    MarkedDuctSeed("upper_opening", (0.173217, 0.080109, 0.081799), 0.0045, 0.020),
    MarkedDuctSeed("side_opening", (0.167836, 0.051387, 0.077508), 0.0035, 0.015),
    MarkedDuctSeed("marked_cylinder", (0.185270, 0.060100, 0.068420), 0.002931, 0.0125),
)


@dataclass(frozen=True)
class MarkedDuctCandidate:
    face_indices: np.ndarray
    per_seed_face_counts: tuple[int, ...]
    matched_boundary_centers_world_m: np.ndarray
    area_m2: float
    connected_components: int
    bounds_from_min_m: np.ndarray


def _unique_edge_graph(vertices: np.ndarray, triangles: np.ndarray) -> csr_matrix:
    """Weighted vertex graph; a shared triangle edge must be counted once."""
    edges = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]),
        axis=0,
    ).astype(np.int64, copy=False)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("invalid edges in new-stomach coverage mesh")
    return coo_matrix(
        (
            np.concatenate((lengths, lengths)),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()


def candidate_marked_duct_mask(
    reference: ReferenceMesh,
    *,
    seeds: tuple[MarkedDuctSeed, ...] = MARKED_DUCT_SEEDS,
) -> MarkedDuctCandidate:
    vertices = np.asarray(reference.vertices_world, dtype=np.float64)
    triangles = np.asarray(reference.triangles, dtype=np.int64)
    if not seeds or len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("marked duct needs seeds and a nonempty mesh")
    if not np.isfinite(vertices).all():
        raise ValueError("non-finite new-stomach vertices")
    boundaries = extract_boundary_components(vertices, triangles)
    fits = [fit_opening(vertices, component) for component in boundaries]
    graph = _unique_edge_graph(vertices, triangles)
    origin = vertices.min(axis=0)
    selected = np.zeros(len(triangles), dtype=bool)
    matched: set[int] = set()
    centers: list[np.ndarray] = []
    counts: list[int] = []
    for seed in seeds:
        if seed.surface_distance_m <= 0 or seed.radius_m <= 0:
            raise ValueError("marked-duct seed dimensions must be positive")
        expected = origin + np.asarray(seed.center_from_min_m, dtype=np.float64)
        matches = [
            (index, float(np.linalg.norm(fit.center - expected)))
            for index, fit in enumerate(fits)
            if index not in matched
            and abs(fit.radius_m - seed.radius_m) <= 0.001
            and np.linalg.norm(fit.center - expected) <= 0.001
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot uniquely match marked-duct opening {seed.name}")
        index = matches[0][0]
        matched.add(index)
        centers.append(fits[index].center)
        distances = dijkstra(
            graph,
            directed=False,
            indices=boundaries[index],
            min_only=True,
            limit=seed.surface_distance_m + 1.0e-9,
        )
        seed_faces = np.all(
            np.asarray(distances)[triangles] <= seed.surface_distance_m + 1.0e-9,
            axis=1,
        )
        counts.append(int(np.count_nonzero(seed_faces)))
        selected |= seed_faces
    face_indices = np.flatnonzero(selected)
    if len(face_indices) == 0:
        raise ValueError("marked duct selection is empty")
    components = _component_count(triangles, face_indices)
    if components != 1:
        raise ValueError(f"marked duct has {components} disconnected surface patches")
    faces = vertices[triangles[face_indices]]
    areas = 0.5 * np.linalg.norm(
        np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
    )
    centers_from_min = faces.mean(axis=1) - origin
    bounds = np.stack((centers_from_min.min(axis=0), centers_from_min.max(axis=0)))
    if not (
        np.all(bounds[0] >= (0.160, 0.040, 0.050))
        and np.all(bounds[1] <= (0.205, 0.085, 0.090))
    ):
        raise ValueError("marked-duct exclusion escaped the annotated region")
    return MarkedDuctCandidate(
        face_indices=face_indices,
        per_seed_face_counts=tuple(counts),
        matched_boundary_centers_world_m=np.stack(centers),
        area_m2=float(areas.sum()),
        connected_components=components,
        bounds_from_min_m=bounds,
    )
