from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from robotarm_magnetic_lab.geometry.stomach_geometry_audit import (
    audit_geometry_alignment,
    author_colored_segments,
)


ROOT_PATH = "/World/envs/env_0/Stomach"


def _stage(tmp_path: Path):
    stage_path = tmp_path / "fixture.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    root = UsdGeom.Xform.Define(stage, ROOT_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    mesh = UsdGeom.Mesh.Define(stage, ROOT_PATH + "/geometry/mesh")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-0.5, -0.5, 0.0),
            Gf.Vec3f(0.5, -0.5, 0.0),
            Gf.Vec3f(0.0, 0.5, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateOrientationAttr("leftHanded")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    texture = tmp_path / "texture.png"
    texture.write_bytes(b"texture")
    shader = UsdShade.Shader.Define(stage, ROOT_PATH + "/Looks/Shader")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    stage.GetRootLayer().Save()
    config = SimpleNamespace(
        position_world_m=(1.0, 2.0, 3.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
        collision_mesh_suffix="/geometry/mesh",
        opening_centers_world_m=((1.0, 2.0, 3.0), (1.1, 2.1, 3.0)),
        opening_axes_world=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        world_bounds_min_m=(0.5, 1.5, 3.0),
        world_bounds_max_m=(1.5, 2.5, 3.0),
        config_sha256="fixture",
        asset_usd_sha256="fixture-asset",
    )
    return stage, config


def test_audit_accepts_one_aligned_triangle_mesh_and_existing_texture(tmp_path):
    stage, config = _stage(tmp_path)

    audit = audit_geometry_alignment(stage, config, ROOT_PATH)

    assert audit.collision_mesh_path == ROOT_PATH + "/geometry/mesh"
    assert audit.collision_mesh_count == 1
    assert audit.collision_orientation == "leftHanded"
    assert audit.opening_elevation_deg == pytest.approx((0.0, 0.0))
    assert audit.texture_count == 1


def test_audit_accepts_runtime_float32_pose_quantization(tmp_path):
    stage, config = _stage(tmp_path)
    exact = np.array([1.1767840832248901, -0.8666323805012753, 0.497086653472885])
    rounded = exact.astype(np.float32).astype(np.float64)
    stage.GetPrimAtPath(ROOT_PATH).GetAttribute("xformOp:translate").Set(Gf.Vec3d(*rounded))
    config.position_world_m = tuple(exact)
    config.opening_centers_world_m = (tuple(exact), tuple(exact + [0.1, 0.1, 0.0]))
    config.world_bounds_min_m = tuple(exact + [-0.5, -0.5, 0.0])
    config.world_bounds_max_m = tuple(exact + [0.5, 0.5, 0.0])

    audit = audit_geometry_alignment(stage, config, ROOT_PATH)

    assert audit.root_transform_error_max < 1.0e-6


def test_author_colored_segments_uses_primvar_interpolation(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "curves.usda"))
    author_colored_segments(
        stage,
        "/World/DebugAxes",
        [((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))],
        [(1.0, 0.0, 0.0)],
        0.01,
    )

    curve = UsdGeom.BasisCurves.Get(stage, "/World/DebugAxes")
    assert curve.GetDisplayColorPrimvar().GetInterpolation() == UsdGeom.Tokens.uniform


def test_audit_rejects_root_transform_mismatch(tmp_path):
    stage, config = _stage(tmp_path)
    stage.GetPrimAtPath(ROOT_PATH).GetAttribute("xformOp:translate").Set(Gf.Vec3d(2.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="stomach root transform"):
        audit_geometry_alignment(stage, config, ROOT_PATH)


def test_audit_rejects_nonhorizontal_opening_axis(tmp_path):
    stage, config = _stage(tmp_path)
    config.opening_axes_world = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="horizontal"):
        audit_geometry_alignment(stage, config, ROOT_PATH)


def test_audit_rejects_duplicate_enabled_collision_meshes(tmp_path):
    stage, config = _stage(tmp_path)
    duplicate = UsdGeom.Mesh.Define(stage, ROOT_PATH + "/duplicate")
    duplicate.CreatePointsAttr([Gf.Vec3f(0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    duplicate.CreateFaceVertexCountsAttr([3])
    duplicate.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(duplicate.GetPrim()).CreateCollisionEnabledAttr(True)

    with pytest.raises(ValueError, match="exactly one enabled"):
        audit_geometry_alignment(stage, config, ROOT_PATH)


def test_audit_rejects_missing_texture(tmp_path):
    stage, config = _stage(tmp_path)
    shader = UsdShade.Shader(stage.GetPrimAtPath(ROOT_PATH + "/Looks/Shader"))
    shader.GetInput("diffuse_texture").Set(Sdf.AssetPath(str(tmp_path / "missing.png")))

    with pytest.raises(ValueError, match="texture"):
        audit_geometry_alignment(stage, config, ROOT_PATH)


def test_audit_rejects_nonfinite_mesh_bounds(tmp_path):
    stage, config = _stage(tmp_path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(ROOT_PATH + "/geometry/mesh"))
    mesh.GetPointsAttr().Set(
        [Gf.Vec3f(float("nan"), 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )

    with pytest.raises(ValueError, match="finite"):
        audit_geometry_alignment(stage, config, ROOT_PATH)
