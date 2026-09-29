"""Runtime audit for the migrated stomach's transform, collider and texture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from .stomach_orientation import quaternion_xyzw_to_matrix


@dataclass(frozen=True)
class GeometryAudit:
    """Serializable evidence that every runtime geometry contract agrees."""

    stomach_root_path: str
    collision_mesh_path: str
    collision_mesh_count: int
    collision_approximation: str
    collision_orientation: str
    geometry_sha256: str
    config_sha256: str
    asset_usd_sha256: str
    world_bounds_min_m: tuple[float, float, float]
    world_bounds_max_m: tuple[float, float, float]
    opening_elevation_deg: tuple[float, float]
    texture_count: int
    texture_paths: tuple[str, ...]
    root_transform_error_max: float

    def to_json(self) -> dict:
        return asdict(self)


def author_colored_segments(stage, path: str, segments, colors, width: float) -> UsdGeom.BasisCurves:
    """Author linear colored debug segments with Kit-compatible primvars."""

    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([2] * len(segments))
    curve.CreatePointsAttr([Gf.Vec3f(*point) for segment in segments for point in segment])
    curve.CreateWidthsAttr([width] * len(segments))
    curve.SetWidthsInterpolation(UsdGeom.Tokens.uniform)
    color_primvar = curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
    color_primvar.Set([Gf.Vec3f(*color) for color in colors])
    return curve


def _expected_world_matrix(config) -> np.ndarray:
    rotation = quaternion_xyzw_to_matrix(np.asarray(config.rotation_xyzw, dtype=np.float64))
    scale = np.asarray(config.scale, dtype=np.float64)
    expected = np.eye(4, dtype=np.float64)
    expected[:3, :3] = rotation.T * scale[:, None]
    expected[3, :3] = np.asarray(config.position_world_m, dtype=np.float64)
    return expected


def _enabled_collision_meshes(root: Usd.Prim) -> list[Usd.Prim]:
    meshes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is not False:
            meshes.append(prim)
    return meshes


def _texture_paths(root: Usd.Prim) -> tuple[str, ...]:
    found: list[str] = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdShade.Shader):
            continue
        for shader_input in UsdShade.Shader(prim).GetInputs():
            value = shader_input.Get()
            if not isinstance(value, Sdf.AssetPath) or not value.path:
                continue
            resolved = value.resolvedPath or value.path
            if not Path(resolved).is_file():
                raise ValueError(f"texture cannot be resolved: {value.path}")
            found.append(str(Path(resolved).resolve()))
    if not found:
        raise ValueError("texture input is missing from the stomach shader")
    return tuple(sorted(set(found)))


def audit_geometry_alignment(stage: Usd.Stage, config, stomach_root_path: str) -> GeometryAudit:
    """Validate the exact runtime transform and a single static triangle collider."""

    root = stage.GetPrimAtPath(stomach_root_path)
    if not root.IsValid():
        raise ValueError(f"stomach root is missing: {stomach_root_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    actual_root = np.asarray(cache.GetLocalToWorldTransform(root), dtype=np.float64)
    if not np.isfinite(actual_root).all():
        raise ValueError("stomach root transform must be finite")
    expected_root = _expected_world_matrix(config)
    transform_error = float(np.abs(actual_root - expected_root).max())
    # Isaac Lab authors the configured pose as float32 on the runtime stage.
    # Keep the audit strict while accepting that expected quantization.
    if transform_error > 1.0e-6:
        raise ValueError(
            "stomach root transform mismatch: "
            f"max error {transform_error}; actual={actual_root.tolist()}; "
            f"expected={expected_root.tolist()}"
        )

    collision_meshes = _enabled_collision_meshes(root)
    if len(collision_meshes) != 1:
        raise ValueError(f"exactly one enabled stomach collision mesh is required: {len(collision_meshes)}")
    mesh_prim = collision_meshes[0]
    expected_mesh_path = stomach_root_path + str(config.collision_mesh_suffix)
    if str(mesh_prim.GetPath()) != expected_mesh_path:
        raise ValueError(
            f"collision mesh path mismatch: {mesh_prim.GetPath()} != {expected_mesh_path}"
        )
    mesh = UsdGeom.Mesh(mesh_prim)
    collision_orientation = str(mesh.GetOrientationAttr().Get() or "rightHanded")
    if collision_orientation != "leftHanded":
        raise ValueError(
            "new stomach collider must face the cavity: "
            f"orientation={collision_orientation!r}"
        )
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("collision mesh points and bounds must be finite")
    if triangles.ndim != 1 or not np.isfinite(triangles).all():
        raise ValueError("collision mesh triangle indices must be finite")
    mesh_matrix = np.asarray(cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64)
    world_points = np.c_[points, np.ones(len(points))] @ mesh_matrix
    bounds_min = world_points[:, :3].min(axis=0)
    bounds_max = world_points[:, :3].max(axis=0)
    if not np.isfinite(bounds_min).all() or not np.isfinite(bounds_max).all():
        raise ValueError("collision mesh world bounds must be finite")
    if not np.allclose(bounds_min, config.world_bounds_min_m, atol=1.0e-6) or not np.allclose(
        bounds_max, config.world_bounds_max_m, atol=1.0e-6
    ):
        raise ValueError(
            f"collision mesh world bounds mismatch: {bounds_min.tolist()}..{bounds_max.tolist()}"
        )

    axes = np.asarray(config.opening_axes_world, dtype=np.float64)
    if axes.shape != (2, 3) or not np.isfinite(axes).all():
        raise ValueError("two finite opening axes are required")
    elevations = np.degrees(np.arcsin(np.clip(np.abs(axes[:, 2]), 0.0, 1.0)))
    if float(elevations.max()) > 1.0:
        raise ValueError(f"both opening axes must be horizontal: {elevations.tolist()}")
    centers = np.asarray(config.opening_centers_world_m, dtype=np.float64)
    if centers.shape != (2, 3) or not np.isfinite(centers).all():
        raise ValueError("two finite opening centers are required")

    textures = _texture_paths(root)
    approximation = UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get()
    if approximation != "none":
        raise ValueError(f"stomach collision approximation must be none, got {approximation!r}")
    digest = hashlib.sha256(points.tobytes() + triangles.tobytes() + mesh_matrix.tobytes()).hexdigest()
    return GeometryAudit(
        stomach_root_path=stomach_root_path,
        collision_mesh_path=str(mesh_prim.GetPath()),
        collision_mesh_count=1,
        collision_approximation=approximation,
        collision_orientation=collision_orientation,
        geometry_sha256=digest,
        config_sha256=str(config.config_sha256),
        asset_usd_sha256=str(config.asset_usd_sha256),
        world_bounds_min_m=tuple(float(item) for item in bounds_min),
        world_bounds_max_m=tuple(float(item) for item in bounds_max),
        opening_elevation_deg=tuple(float(item) for item in elevations),
        texture_count=len(textures),
        texture_paths=textures,
        root_transform_error_max=transform_error,
    )
