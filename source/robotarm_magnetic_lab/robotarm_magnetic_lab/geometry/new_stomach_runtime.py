"""Single audited geometry contract for every new-stomach runtime consumer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom

from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, ReferenceMesh, preprocess_reference_mesh

from .stomach_geometry_audit import GeometryAudit, audit_geometry_alignment


# The converted asset contains two valid 0.222 um edges.  The legacy 1 um
# coverage weld would collapse them and create degenerate faces.  This smaller
# tolerance preserves every authored triangle while still merging numerical
# duplicates well below the asset's minimum edge length.
NEW_STOMACH_WELD_TOLERANCE_M = 1.0e-7


@dataclass(frozen=True)
class NewStomachRuntimeGeometry:
    """Audited world-space mesh shared by planning, wall and coverage code."""

    stomach_root_path: str
    collision_mesh_path: str
    world_transform: np.ndarray
    reference: ReferenceMesh
    audit: GeometryAudit
    orientation_config_sha256: str
    asset_usd_sha256: str

    @property
    def geometry_sha256(self) -> str:
        return self.reference.geometry_sha256

    def consumer_hashes(self) -> dict[str, str]:
        """Return the required five-way geometry identity evidence."""

        return {
            name: self.geometry_sha256
            for name in ("collision", "planning", "wall_surface", "coverage", "visual_alignment")
        }

    @classmethod
    def from_stage(
        cls,
        stage: Usd.Stage,
        config: Any,
        stomach_root_path: str = "/World/envs/env_0/Stomach",
    ) -> "NewStomachRuntimeGeometry":
        if not bool(getattr(config, "confirmed", False)):
            raise ValueError("new stomach orientation must be user-confirmed before runtime binding")
        audit = audit_geometry_alignment(stage, config, stomach_root_path)
        mesh_prim = stage.GetPrimAtPath(audit.collision_mesh_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        transform = np.asarray(
            UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=np.float64,
        )
        mesh_input = MeshInput(
            prim_path=audit.collision_mesh_path,
            vertices=np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64),
            face_vertex_counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64),
            face_vertex_indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64),
            world_transform=transform,
            orientation=str(mesh.GetOrientationAttr().Get() or "rightHanded"),
        )
        reference = preprocess_reference_mesh(
            [mesh_input],
            [audit.collision_mesh_path],
            weld_tolerance_m=NEW_STOMACH_WELD_TOLERANCE_M,
        )
        return cls(
            stomach_root_path=stomach_root_path,
            collision_mesh_path=audit.collision_mesh_path,
            world_transform=transform,
            reference=reference,
            audit=audit,
            orientation_config_sha256=str(config.config_sha256),
            asset_usd_sha256=str(config.asset_usd_sha256),
        )


def validate_model_specific_inputs(
    stomach_model: str,
    *,
    unreachable_region_path: Path | None = None,
    legacy_initial_pose: bool = False,
) -> None:
    """Prevent old-model coordinates or topology indices entering new mode."""

    if stomach_model not in {"legacy", "new_v1"}:
        raise ValueError(f"unknown stomach model: {stomach_model}")
    if stomach_model == "new_v1" and unreachable_region_path is not None:
        raise ValueError("new_v1 rejects the legacy unreachable-region configuration")
    if stomach_model == "new_v1" and legacy_initial_pose:
        raise ValueError("new_v1 rejects legacy initial coordinates")
