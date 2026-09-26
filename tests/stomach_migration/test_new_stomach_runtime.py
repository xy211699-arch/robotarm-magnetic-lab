from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from robotarm_magnetic_lab.geometry.new_stomach_runtime import (
    NewStomachRuntimeGeometry,
    validate_model_specific_inputs,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ball_envelope import (
    stomach_collision_mesh,
)


ROOT = "/World/envs/env_0/Stomach"
MESH = ROOT + "/geometry/mesh"


def _stage(tmp_path: Path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "runtime.usda"))
    root = UsdGeom.Xform.Define(stage, ROOT)
    root.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    mesh = UsdGeom.Mesh.Define(stage, MESH)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    texture = tmp_path / "texture.png"
    texture.write_bytes(b"texture")
    shader = UsdShade.Shader.Define(stage, ROOT + "/Looks/Shader")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    config = SimpleNamespace(
        position_world_m=(1.0, 2.0, 3.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
        collision_mesh_suffix="/geometry/mesh",
        opening_centers_world_m=((1.0, 2.0, 3.0), (1.1, 2.1, 3.0)),
        opening_axes_world=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        world_bounds_min_m=(1.0, 2.0, 3.0),
        world_bounds_max_m=(2.0, 3.0, 3.0),
        config_sha256="orientation-hash",
        asset_usd_sha256="asset-hash",
        confirmed=True,
    )
    return stage, config


def test_runtime_geometry_binds_all_consumers_to_one_world_mesh(tmp_path):
    stage, config = _stage(tmp_path)

    runtime = NewStomachRuntimeGeometry.from_stage(stage, config, ROOT)

    assert runtime.collision_mesh_path == MESH
    assert runtime.reference.selected_prim_paths == (MESH,)
    assert runtime.consumer_hashes() == {
        "collision": runtime.geometry_sha256,
        "planning": runtime.geometry_sha256,
        "wall_surface": runtime.geometry_sha256,
        "coverage": runtime.geometry_sha256,
        "visual_alignment": runtime.geometry_sha256,
    }
    assert stomach_collision_mesh(stage, configured_path=MESH) == MESH
    np.testing.assert_allclose(runtime.reference.vertices_world, [[1, 2, 3], [1, 3, 3], [2, 2, 3]])


def test_runtime_rejects_unconfirmed_or_wrong_collision_mesh(tmp_path):
    stage, config = _stage(tmp_path)
    config.confirmed = False
    with pytest.raises(ValueError, match="confirmed"):
        NewStomachRuntimeGeometry.from_stage(stage, config, ROOT)
    with pytest.raises(RuntimeError, match="configured stomach collision"):
        stomach_collision_mesh(stage, configured_path=ROOT + "/wrong")


def test_new_mode_rejects_legacy_coordinates_and_unreachable_mask():
    with pytest.raises(ValueError, match="unreachable"):
        validate_model_specific_inputs("new_v1", unreachable_region_path=Path("old.json"))
    with pytest.raises(ValueError, match="initial"):
        validate_model_specific_inputs("new_v1", legacy_initial_pose=True)
    validate_model_specific_inputs("legacy", unreachable_region_path=Path("old.json"), legacy_initial_pose=True)


def test_screening_cli_keeps_legacy_default_and_explicit_new_mode():
    script = (Path(__file__).parents[2] / "scripts/magnetic_following/screen_vector_random.py").read_text()
    assert "--stomach-model" in script
    assert "choices=('legacy','new_v1')" in script
    assert "default='legacy'" in script
    assert "RobotarmMagneticNewStomachLabEnvCfg" in script
