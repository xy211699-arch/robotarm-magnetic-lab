"""Privileged P0 stomach-surface coverage evaluation."""

from .reference_mesh import MeshInput, ReferenceMesh, preprocess_reference_mesh
from .records import CoverageRecordWriter, artifact_inventory, deployable_fields
from .runtime import RecordedFrameClock, assert_coverage_consistency
from .accumulator import CoverageAccumulator, CoverageUpdate
from .area_weights import target_vertex_area_weights, weights_sha256
from .unreachable_region import (
    UNREACHABLE_REGION_SCHEMA,
    UnreachableMask,
    UnreachableSeed,
    build_unreachable_mask,
    load_unreachable_mask,
    unreachable_region_record,
)
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
    "target_vertex_area_weights",
    "weights_sha256",
    "UNREACHABLE_REGION_SCHEMA",
    "UnreachableMask",
    "UnreachableSeed",
    "build_unreachable_mask",
    "load_unreachable_mask",
    "unreachable_region_record",
    "MeshInput",
    "ReferenceMesh",
    "RecordedFrameClock",
    "ScalarFirstHitRaycaster",
    "WarpFirstHitRaycaster",
    "candidate_vertices",
    "artifact_inventory",
    "assert_coverage_consistency",
    "deployable_fields",
    "preprocess_reference_mesh",
    "visible_from_first_hits",
]
