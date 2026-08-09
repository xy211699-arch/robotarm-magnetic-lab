#!/usr/bin/env python3
"""Exercise all 11 atomic IDs in the independent Isaac Lab table task."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))

HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Robotarm-Magnetic-Atomic-Table-Lab-v0")
parser.add_argument("--max_steps_per_action", type=int, default=60)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401, E402
import robotarm_magnetic_lab.tasks  # noqa: F401, E402
from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.action_layer import (  # noqa: E402
    AtomicAction,
)


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    output_path = Path("logs/action_layer/stage1_atomic_results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with launch_simulation(cfg, args_cli):
        env = gym.make(args_cli.task, cfg=cfg)
        for action in AtomicAction:
            env.reset()
            term = env.unwrapped.action_manager.get_term("atomic")
            command = torch.full(
                env.action_space.shape,
                float(int(action)),
                device=env.unwrapped.device,
            )
            result = None
            for _ in range(args_cli.max_steps_per_action):
                _, _, terminated, truncated, _ = env.step(command)
                result = term.last_result
                if result is not None:
                    break
                if bool(terminated[0] or truncated[0]):
                    break
            record = (
                result.to_dict()
                if result is not None
                else {
                    "action_id": int(action),
                    "action": action.name,
                    "status": "NO_TERMINAL_RESULT",
                }
            )
            records.append(record)
            print("ATOMIC_RESULT " + json.dumps(record, sort_keys=True), flush=True)
        env.close()
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    terminal = sum(record["status"] in ("DONE", "HARD_FAILURE") for record in records)
    done = sum(record["status"] == "DONE" for record in records)
    print(
        f"ATOMIC_VALIDATION actions={len(records)} terminal={terminal} "
        f"done={done} log={output_path}"
    )
    return 0 if done == len(AtomicAction) else 1


if __name__ == "__main__":
    raise SystemExit(main())
