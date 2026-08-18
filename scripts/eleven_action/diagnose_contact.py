#!/usr/bin/env python3
"""诊断 TASK-005 平面场景的胶囊接触报告链路。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(ROOT / "scripts" / "eleven_action"))


def main() -> int:
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    import gymnasium as gym
    import numpy as np
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from calibrate_eleven_action import reset_flat_trial, run_one_action
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
        load_dynamic_profile,
    )
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action.geometry import (
        capsule_axis_world,
    )

    env_cfg = parse_env_cfg(
        "Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0",
        device=args.device,
        num_envs=1,
    )
    with launch_simulation(env_cfg, args):
        env = gym.make("Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0", cfg=env_cfg)
        try:
            import omni.usd
            from pxr import Usd, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            rigid_paths = []
            collision_paths = []
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Scene")):
                path = prim.GetPath().pathString
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_paths.append(path)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision_paths.append(path)
            print(f"ELEVEN_ACTION_RIGID_PATHS {rigid_paths}", flush=True)
            print(f"ELEVEN_ACTION_COLLISION_PATHS {collision_paths}", flush=True)
            term = reset_flat_trial(
                env,
                load_dynamic_profile(),
                tilt_deg=67.0,
                azimuth_deg=137.0,
                roll_deg=211.0,
                settle_steps=12,
            )
            state_short = term._read_state()
            short_axis = capsule_axis_world(state_short)
            short_swing = state_short.angular_velocity_world_rad_s - (
                state_short.angular_velocity_world_rad_s @ short_axis
            ) * short_axis
            short_hold_error = np.degrees(
                np.arccos(np.clip(float(short_axis @ term.controller._hold_axis), -1.0, 1.0))
            )
            print(
                "ELEVEN_ACTION_SETTLE_STATE steps=12 "
                f"linear_speed={np.linalg.norm(state_short.linear_velocity_world_m_s):.9f} "
                f"angular_speed={np.linalg.norm(state_short.angular_velocity_world_rad_s):.9f} "
                f"swing_speed={np.linalg.norm(short_swing):.9f} "
                f"hold_error_deg={short_hold_error:.9f} "
                f"torque={term.telemetry.torque_world_nm.tolist()}",
                flush=True,
            )
            no_request = torch.full((1, 1), -1.0, device=env.unwrapped.device)
            for _ in range(108):
                env.step(no_request)
            state_long = term._read_state()
            long_axis = capsule_axis_world(state_long)
            long_swing = state_long.angular_velocity_world_rad_s - (
                state_long.angular_velocity_world_rad_s @ long_axis
            ) * long_axis
            long_hold_error = np.degrees(
                np.arccos(np.clip(float(long_axis @ term.controller._hold_axis), -1.0, 1.0))
            )
            print(
                "ELEVEN_ACTION_SETTLE_STATE steps=120 "
                f"linear_speed={np.linalg.norm(state_long.linear_velocity_world_m_s):.9f} "
                f"angular_speed={np.linalg.norm(state_long.angular_velocity_world_rad_s):.9f} "
                f"swing_speed={np.linalg.norm(long_swing):.9f} "
                f"hold_error_deg={long_hold_error:.9f} "
                f"torque={term.telemetry.torque_world_nm.tolist()}",
                flush=True,
            )
            for milestone in (300, 600):
                for _ in range(milestone - int(term._physics_substep / 4)):
                    env.step(no_request)
                state_milestone = term._read_state()
                axis_milestone = capsule_axis_world(state_milestone)
                swing_milestone = state_milestone.angular_velocity_world_rad_s - (
                    state_milestone.angular_velocity_world_rad_s @ axis_milestone
                ) * axis_milestone
                hold_error = np.degrees(
                    np.arccos(
                        np.clip(float(axis_milestone @ term.controller._hold_axis), -1.0, 1.0)
                    )
                )
                print(
                    f"ELEVEN_ACTION_SETTLE_STATE steps={milestone} "
                    f"linear_speed={np.linalg.norm(state_milestone.linear_velocity_world_m_s):.9f} "
                    f"angular_speed={np.linalg.norm(state_milestone.angular_velocity_world_rad_s):.9f} "
                    f"swing_speed={np.linalg.norm(swing_milestone):.9f} "
                    f"hold_error_deg={hold_error:.9f} "
                    f"torque={term.telemetry.torque_world_nm.tolist()}",
                    flush=True,
                )
            sensor = env.unwrapped.scene["capsule_contact"]
            print(
                "ELEVEN_ACTION_CONTACT_BUFFER_SHAPES "
                f"positions={tuple(sensor.data.contact_pos_w.torch.shape)} "
                f"force_matrix={tuple(sensor.data.force_matrix_w.torch.shape)} "
                f"net_forces={tuple(sensor.data.net_forces_w.torch.shape)}",
                flush=True,
            )
            print(
                "ELEVEN_ACTION_CONTACT_FORCE_MATRIX "
                f"{sensor.data.force_matrix_w.torch.detach().cpu().numpy().reshape(-1, 3).tolist()}",
                flush=True,
            )
            print(f"ELEVEN_ACTION_CONTACT_DIAGNOSTICS {term.contact_diagnostics}", flush=True)
            recent = term.controller.contact_history.recent_contacts(
                current_substep=term._physics_substep,
                last_n_substeps=12,
            )
            print(
                "ELEVEN_ACTION_CONTACT_REGIONS "
                f"{[(sample.region.value, sample.axial_coordinate_m) for sample in recent]}",
                flush=True,
            )
            print(
                "ELEVEN_ACTION_CONTACT_NET_FORCE "
                f"{sensor.data.net_forces_w.torch.detach().cpu().numpy().reshape(-1, 3).tolist()}",
                flush=True,
            )
            print(f"ELEVEN_ACTION_CONTACT_MOVE {run_one_action(env, term, 9)}", flush=True)
        finally:
            env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
