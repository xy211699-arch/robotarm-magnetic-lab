"""Privileged P0 stomach-surface coverage evaluation."""

from .reference_mesh import MeshInput, ReferenceMesh, preprocess_reference_mesh
from .records import CoverageRecordWriter, artifact_inventory, deployable_fields
from .accumulator import CoverageAccumulator, CoverageUpdate
from .visibility import (
    ScalarFirstHitRaycaster,
    WarpFirstHitRaycaster,
    candidate_vertices,
    visible_from_first_hits,
)

__all__ = [
    "CoverageAccumulator",
    "CoverageRecordWriter",
    "CoverageUpdate",
    "MeshInput",
    "ReferenceMesh",
    "ScalarFirstHitRaycaster",
    "WarpFirstHitRaycaster",
    "candidate_vertices",
    "artifact_inventory",
    "deployable_fields",
    "preprocess_reference_mesh",
    "visible_from_first_hits",
]
