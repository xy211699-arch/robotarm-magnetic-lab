#!/usr/bin/env python3
"""Verify TASK-008 live geometry, dynamics and equivalent-wrench semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))
HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0")
parser.add_argument("--output", type=Path, default=Path("/tmp/task008-preflight.json"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
args_cli.enable_cameras = True
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import robotarm_magnetic_lab.tasks  # noqa: E402,F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force_macro import (
    DynamicForceMacroConfig,
    PointForce,
    equivalent_com_wrench,
)


FORBIDDEN = ("write_root_pose", "write_root_velocity", "set_transforms", "set_velocities")


def source_scan() -> dict:
    files = [
        ROOT / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_macro_action.py",
        ROOT / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/dynamic_force_macro_runner.py",
    ]
    hits = {name: [] for name in FORBIDDEN}
    for path in files:
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN:
            if name in text:
                hits[name].append(str(path))
    if any(hits.values()):
        raise RuntimeError(f"forbidden runtime state writer found: {hits}")
    return {"files": [str(path) for path in files], "forbidden_hits": hits}


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    with launch_simulation(cfg, args_cli):
        env = gym.make(args_cli.task, cfg=cfg)
        env.reset()
        base = env.unwrapped
        term = base.action_manager.get_term("dynamic_force_macro")
        capsule = base.scene["capsule"]
        import omni.usd
        from pxr import PhysxSchema, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        capsule_path = "/World/envs/env_0/Scene/MagneticDemo/target_magnet"
        capsule_prim = stage.GetPrimAtPath(capsule_path)
        if not capsule_prim.IsValid():
            raise RuntimeError(f"capsule prim is unavailable: {capsule_path}")
        enabled_colliders = []
        capsule_shapes = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = UsdPhysics.CollisionAPI(prim)
            enabled = api.GetCollisionEnabledAttr().Get()
            if enabled is False:
                continue
            path = str(prim.GetPath())
            enabled_colliders.append(path)
            if path.startswith(capsule_path) and prim.IsA(UsdGeom.Capsule):
                shape = UsdGeom.Capsule(prim)
                capsule_shapes.append(
                    {
                        "prim_path": path,
                        "radius_m": float(shape.GetRadiusAttr().Get()),
                        "cylinder_height_m": float(shape.GetHeightAttr().Get()),
                        "axis": str(shape.GetAxisAttr().Get() or "Z"),
                    }
                )
        if len(capsule_shapes) != 1:
            raise RuntimeError(f"expected one enabled capsule shape, found {capsule_shapes}")
        live_shape = capsule_shapes[0]
        if not math.isclose(live_shape["radius_m"], 0.0065, abs_tol=1.0e-9):
            raise RuntimeError(f"unexpected live capsule radius: {live_shape}")
        if not math.isclose(live_shape["cylinder_height_m"], 0.012, abs_tol=1.0e-9):
            raise RuntimeError(f"unexpected live capsule cylinder height: {live_shape}")
        mass = float(capsule.data.body_mass.torch.reshape(-1)[0].item())
        inertia = capsule.data.body_inertia.torch.reshape(-1).detach().cpu().numpy().tolist()
        local_com_pose = capsule.data.body_com_pose_b.torch[0, 0].detach().cpu().numpy()
        link = capsule.data.root_link_pos_w.torch[0].detach().cpu().numpy()
        com = capsule.data.root_com_pos_w.torch[0].detach().cpu().numpy()
        config = DynamicForceMacroConfig()
        camera_offset = tuple(float(value) for value in base.scene["capsule_camera"].cfg.offset.pos)
        if camera_offset[2] >= 0.0:
            raise RuntimeError(f"camera is not authored on capsule local -Z: {camera_offset}")
        body_api = PhysxSchema.PhysxRigidBodyAPI(capsule_prim)
        body_ccd = body_api.GetEnableCCDAttr().Get()
        rigid_api = UsdPhysics.RigidBodyAPI(capsule_prim)
        kinematic = rigid_api.GetKinematicEnabledAttr().Get()
        gravity_disabled = body_api.GetDisableGravityAttr().Get()
        physics_scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        if len(physics_scenes) != 1:
            raise RuntimeError(f"expected one physics scene, found {len(physics_scenes)}")
        scene_ccd = PhysxSchema.PhysxSceneAPI(physics_scenes[0]).GetEnableCCDAttr().Get()
        if not rigid_api or kinematic is True or gravity_disabled is True:
            raise RuntimeError(
                f"capsule must be dynamic with gravity: rigid={bool(rigid_api)} "
                f"kinematic={kinematic} gravity_disabled={gravity_disabled}"
            )
        if body_ccd is not True or scene_ccd is not True:
            raise RuntimeError(f"CCD contract failed: body={body_ccd} scene={scene_ccd}")
        table_colliders = [path for path in enabled_colliders if not path.startswith(capsule_path)]
        if not table_colliders:
            raise RuntimeError("no enabled table/world collider found")
        fixture = (
            PointForce("camera", np.array([0.0, 0.0, -0.006]), np.array([0.0, 1.0, 0.0])),
            PointForce("other", np.array([0.0, 0.0, 0.006]), np.array([0.0, 1.0, 0.0])),
        )
        force, torque = equivalent_com_wrench(fixture, np.zeros(3))
        payload = {
            "schema": "task008_preflight_v1",
            "task": args_cli.task,
            "geometry": {
                "radius_m": live_shape["radius_m"],
                "cylinder_height_m": live_shape["cylinder_height_m"],
                "total_length_m": live_shape["cylinder_height_m"] + 2 * live_shape["radius_m"],
                "shape_axis_local": live_shape["axis"],
                "long_axis_local": [0.0, 0.0, 1.0],
                "hemisphere_centers_local_z_m": [-0.006, 0.006],
                "camera_side_local_axis_sign": -1,
                "camera_offset_m": list(camera_offset),
                "center_of_mass_local_pose": local_com_pose.tolist(),
                "link_origin_world_m": link.tolist(),
                "com_world_m": com.tolist(),
            },
            "physics": {
                "mass_kg": mass,
                "inertia": inertia,
                "gravity_enabled": gravity_disabled is not True,
                "kinematic_enabled": kinematic is True,
                "body_ccd": bool(body_ccd),
                "scene_ccd": bool(scene_ccd),
                "physics_device": str(cfg.sim.device),
                "table_colliders": table_colliders,
                "stomach_collider_contract": (
                    "/World/envs/env_0/Stomach/ConvertedSource/Environment/"
                    "Stomach/Physics_Collision_Mesh/Stomach"
                ),
            },
            "clocks_hz": {"physics": 240, "environment": 60, "camera": 30, "actor": 1},
            "wrench_api": {
                "selected_path": "equivalent_com_wrench",
                "reason": "permanent wrench composer exposes one indexed force/torque/position tuple per body",
                "two_force_fixture_force": force.tolist(),
                "two_force_fixture_torque": torque.tolist(),
            },
            "runtime_source_scan": source_scan(),
            "action_terms": list(base.action_manager.active_terms),
            "dynamic_body": term.mass_kg > 0.0,
        }
        env.close()
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args_cli.output.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f"TASK008_PREFLIGHT_PASS output={args_cli.output.resolve()} bytes={len(text.encode())} sha256={digest}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
