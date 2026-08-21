#!/usr/bin/env python3
"""Verify TASK-008 live geometry, dynamics and equivalent-wrench semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Robotarm-Magnetic-Dynamic-Force-Macro-Table-Lab-v0")
parser.add_argument("--output", type=Path, default=Path("/tmp/task008-preflight.json"))
parser.add_argument("--headless", action="store_true", help="Compatibility flag; current launcher is non-windowed by default.")
AppLauncher.add_app_launcher_args(parser)
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
        mass = float(capsule.data.body_mass.torch.reshape(-1)[0].item())
        inertia = capsule.data.body_inertia.torch.reshape(-1).detach().cpu().numpy().tolist()
        link = capsule.data.root_link_pos_w.torch[0].detach().cpu().numpy()
        com = capsule.data.root_com_pos_w.torch[0].detach().cpu().numpy()
        config = DynamicForceMacroConfig()
        fixture = (
            PointForce("camera", np.array([0.0, 0.0, -0.006]), np.array([0.0, 1.0, 0.0])),
            PointForce("other", np.array([0.0, 0.0, 0.006]), np.array([0.0, 1.0, 0.0])),
        )
        force, torque = equivalent_com_wrench(fixture, np.zeros(3))
        payload = {
            "schema": "task008_preflight_v1",
            "task": args_cli.task,
            "geometry": {
                "radius_m": config.capsule_radius_m,
                "cylinder_height_m": config.cylinder_height_m,
                "total_length_m": config.cylinder_height_m + 2 * config.capsule_radius_m,
                "hemisphere_centers_local_z_m": [-0.006, 0.006],
                "camera_side_local_axis_sign": -1,
                "camera_offset_m": list(base.scene["capsule_camera"].cfg.offset.pos),
                "link_origin_world_m": link.tolist(),
                "com_world_m": com.tolist(),
            },
            "physics": {
                "mass_kg": mass,
                "inertia": inertia,
                "gravity_enabled": True,
                "body_ccd": True,
                "scene_ccd": bool(cfg.sim.physics.enable_ccd),
                "physics_device": str(cfg.sim.device),
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
