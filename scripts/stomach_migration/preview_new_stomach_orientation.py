#!/usr/bin/env python3
"""Render audited front, side and top previews of the migrated stomach pose."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)


import numpy as np
import omni.usd
from omni.replicator.core.functional import write_image
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg

from robotarm_magnetic_lab.geometry.stomach_geometry_audit import (
    audit_geometry_alignment,
    author_colored_segments,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_env_cfg import (
    NEW_STOMACH_GEOMETRY,
    RobotarmMagneticNewStomachLabEnvCfg,
)


STOMACH_ROOT = "/World/envs/env_0/Stomach"
DEBUG_ROOT = "/World/NewStomachOrientationDebug"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sphere(stage, path, center, radius, color):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    transform = UsdGeom.Xformable(sphere)
    transform.AddTranslateOp().Set(Gf.Vec3d(*center))


def _draw_orientation(stage):
    center = (
        np.asarray(NEW_STOMACH_GEOMETRY.world_bounds_min_m)
        + np.asarray(NEW_STOMACH_GEOMETRY.world_bounds_max_m)
    ) / 2.0
    world_segments = [
        (center, center + np.array([0.08, 0.0, 0.0])),
        (center, center + np.array([0.0, 0.08, 0.0])),
        (center, center + np.array([0.0, 0.0, 0.08])),
    ]
    author_colored_segments(
        stage,
        DEBUG_ROOT + "/WorldAxes",
        world_segments,
        ((1.0, 0.05, 0.05), (0.05, 1.0, 0.05), (0.05, 0.2, 1.0)),
        0.002,
    )
    opening_colors = ((1.0, 0.45, 0.0), (0.85, 0.0, 1.0))
    opening_segments = []
    for index, (opening_center, opening_axis, color) in enumerate(
        zip(
            NEW_STOMACH_GEOMETRY.opening_centers_world_m,
            NEW_STOMACH_GEOMETRY.opening_axes_world,
            opening_colors,
            strict=True,
        )
    ):
        start = np.asarray(opening_center)
        end = start + 0.07 * np.asarray(opening_axis)
        opening_segments.append((start, end))
        _sphere(stage, DEBUG_ROOT + f"/Opening{index}Center", start, 0.004, color)
        _sphere(stage, DEBUG_ROOT + f"/Opening{index}Outward", end, 0.007, color)
    author_colored_segments(stage, DEBUG_ROOT + "/OpeningAxes", opening_segments, opening_colors, 0.003)


def _render_previews(env, output: Path):
    bounds_min = np.asarray(NEW_STOMACH_GEOMETRY.world_bounds_min_m)
    bounds_max = np.asarray(NEW_STOMACH_GEOMETRY.world_bounds_max_m)
    center = (bounds_min + bounds_max) / 2.0
    views = {
        "front": (center + np.array([0.0, -0.36, 0.08]), center),
        "side": (center + np.array([0.38, 0.0, 0.08]), center),
        "top": (center + np.array([0.0, 0.0, 0.45]), center),
    }
    cameras = {name: env.scene[f"new_stomach_{name}_camera"] for name in views}
    for name, (eye, target) in views.items():
        cameras[name].set_world_poses_from_view(
            np.asarray(eye, dtype=np.float32)[None, :],
            np.asarray(target, dtype=np.float32)[None, :],
        )
    # Each view owns a distinct render product.  One physics/render step is
    # enough to propagate all Fabric poses to RTX without stale-frame reuse.
    env.sim.step(render=True)
    env.scene.update(env.sim.get_physics_dt())
    image_paths = []
    for name, camera in cameras.items():
        camera.update(env.sim.get_physics_dt(), force_recompute=True)
        image = camera.data.output["rgb"][0]
        if image is None:
            raise RuntimeError(f"preview image is missing: {name}")
        image_array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        if image_array.ndim != 3 or not np.isfinite(image_array).all():
            raise RuntimeError(f"preview image is missing or non-finite: {name}")
        path = output / f"new_stomach_{name}.png"
        write_image(path=str(path), data=image_array)
        image_paths.append(path)
    if len({_sha256(path) for path in image_paths}) != len(image_paths):
        raise RuntimeError("preview camera returned duplicate frames after changing viewpoints")
    return image_paths


def main() -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = RobotarmMagneticNewStomachLabEnvCfg()
    cfg.sim.device = args.device
    cfg.sim.render_interval = 1
    cfg.scene.num_envs = 1
    for view_name in ("front", "side", "top"):
        setattr(
            cfg.scene,
            f"new_stomach_{view_name}_camera",
            CameraCfg(
                prim_path=f"/World/NewStomachPreview/{view_name}",
                update_period=0.0,
                width=960,
                height=540,
                data_types=["rgb"],
                update_latest_camera_pose=True,
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    horizontal_aperture=36.0,
                    focus_distance=0.2,
                    clipping_range=(0.005, 5.0),
                ),
            ),
        )
    env = None
    try:
        with launch_simulation(cfg, args):
            env = ManagerBasedRLEnv(cfg=cfg)
            stage = omni.usd.get_context().get_stage()
            audit = audit_geometry_alignment(stage, NEW_STOMACH_GEOMETRY, STOMACH_ROOT)
            _draw_orientation(stage)
            for _ in range(3):
                env.sim.render()
            images = _render_previews(env, output)
            record = audit.to_json()
            record.update(
                status="needs_input",
                confirmation_required=(
                    "Confirm that both marked tube axes are horizontal and the stomach placement is acceptable."
                ),
                preview_files=[
                    {
                        "path": str(path),
                        "byte_size": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in images
                ],
            )
            audit_path = output / "orientation_audit.json"
            audit_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            print("NEW_STOMACH_ORIENTATION", json.dumps(record, sort_keys=True), flush=True)
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
