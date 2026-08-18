#!/usr/bin/env python3
"""在正式门禁暴露的固定最差姿态上诊断授权轴增益。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(ROOT / "scripts" / "eleven_action"))

VIEW_CASES = (
    (1, 36.625, 161.464, 85.393),
    (2, 49.509, 136.910, 124.651),
    (3, 48.169, 308.257, 229.737),
    (4, 32.143, 339.572, 128.367),
    (5, 73.648, 191.803, 36.261),
    (6, 53.594, 228.440, 122.222),
    (7, 86.615, 256.739, 256.176),
    (8, 61.467, 252.848, 8.539),
)
HOLD_CASES = (
    (66.697, 56.964, 173.881),
    (62.156, 314.278, 125.330),
)


def main() -> int:
    if "--headless" in sys.argv:
        sys.argv.remove("--headless")
        os.environ["HEADLESS"] = "1"
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    launcher = AppLauncher(args)
    simulation_app = launcher.app

    import gymnasium as gym
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from calibrate_eleven_action import reset_flat_trial, run_one_action
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
        load_dynamic_profile,
    )

    env_cfg = parse_env_cfg(
        "Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0",
        device=args.device,
        num_envs=1,
    )
    env_cfg.observations.policy.rgb = None
    env_cfg.scene.capsule_camera = None
    env_cfg.sim.render_interval = 240
    base = load_dynamic_profile()
    with launch_simulation(env_cfg, args):
        env = gym.make("Template-Robotarm-Magnetic-Eleven-Action-Flat-Lab-v0", cfg=env_cfg)
        try:
            for kp in (0.005, 0.01, 0.02):
                for kd in (0.0008, 0.0016, 0.0032):
                    profile = replace(base, axis_kp_nm_per_rad=kp, axis_kd_nms_per_rad=kd)
                    rows = []
                    for action_id, tilt, azimuth, roll in VIEW_CASES:
                        term = reset_flat_trial(
                            env, profile, tilt_deg=tilt, azimuth_deg=azimuth, roll_deg=roll
                        )
                        rows.append(run_one_action(env, term, action_id))
                    holds = []
                    for tilt, azimuth, roll in HOLD_CASES:
                        term = reset_flat_trial(
                            env, profile, tilt_deg=tilt, azimuth_deg=azimuth, roll_deg=roll
                        )
                        holds.append(run_one_action(env, term, 0))
                    max_view_error = max(abs(row["angle_delta_deg"] - 15.0) for row in rows)
                    max_hold_delta = max(row["angle_delta_deg"] for row in holds)
                    max_drift = max(row["max_support_drift_m"] for row in (*rows, *holds))
                    passed = max_view_error <= 3.0 and max_hold_delta <= 3.0 and max_drift <= 0.002
                    print(
                        "ELEVEN_ACTION_TRACKING_DIAG "
                        f"kp={kp} kd={kd} passed={passed} "
                        f"max_view_error_deg={max_view_error:.6f} "
                        f"max_hold_delta_deg={max_hold_delta:.6f} "
                        f"max_drift_m={max_drift:.9f}",
                        flush=True,
                    )
        finally:
            env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
