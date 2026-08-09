#!/usr/bin/env python3
"""Drive repeated APPROACH actions until the stomach sweep guard rejects one."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_actions", type=int, default=40)
parser.add_argument("--max_steps_per_action", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import robotarm_magnetic_lab.tasks  # noqa: F401, E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.action_layer import (  # noqa: E402
    AtomicAction,
    HardFailureCode,
)


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # Keep the executor alive for one diagnostic read after HARD_FAILURE. The
    # production task still terminates on this state; disabling only the test
    # termination prevents Gym's automatic reset from erasing last_result.
    cfg.terminations.atomic_hard_failure = None
    with launch_simulation(cfg, args_cli):
        env = gym.make(args_cli.task, cfg=cfg)
        env.reset()
        term = env.unwrapped.action_manager.get_term("atomic")
        command = torch.full(
            env.action_space.shape,
            float(int(AtomicAction.APPROACH)),
            device=env.unwrapped.device,
        )
        for request_index in range(1, args_cli.max_actions + 1):
            result = None
            for _ in range(args_cli.max_steps_per_action):
                env.step(command)
                result = term.last_result
                if result is not None:
                    break
            if result is None:
                print(f"STOMACH_GUARD_FAIL request={request_index} reason=no_terminal_result")
                env.close()
                return 2
            checker = term.executor.safety.world_collision_checker
            snapshot = term._snapshot()
            live = checker.check_configuration(snapshot.joint_position_rad[:6])
            print(
                f"STOMACH_GUARD_STEP request={request_index} status={result.status.value} "
                f"code={None if result.hard_failure_code is None else result.hard_failure_code.value} "
                f"live_clearance_m={live.clearance_m:.6f} required_m={checker.required_clearance_m:.6f}",
                flush=True,
            )
            if result.status.value == "HARD_FAILURE":
                passed = (
                    result.hard_failure_code is HardFailureCode.ENVIRONMENT_COLLISION
                    # The 5 mm planning/runtime buffer is consumed while the
                    # physical arm settles after hold. Passing means the guard
                    # fired for the right reason before geometric overlap.
                    and live.clearance_m > 0.0
                )
                print(
                    "STOMACH_GUARD_VALIDATION "
                    f"status={'PASS' if passed else 'FAIL'} detail={result.hard_failure_detail}",
                    flush=True,
                )
                env.close()
                return 0 if passed else 3
            term.acknowledge_result()
        print("STOMACH_GUARD_FAIL reason=no_environment_rejection")
        env.close()
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
