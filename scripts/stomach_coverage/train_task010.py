#!/usr/bin/env python3
"""Single TASK-010 training entry for fake contract and Isaac Lab backends."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_runner(args, device):
    from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
    from robotarm_magnetic_lab.learning.task010_critic import Task010Critic
    from robotarm_magnetic_lab.learning.task010_runner import Task010OnPolicyRunner
    from robotarm_magnetic_lab.runtime.task010_config import load_task010_config

    config = load_task010_config(args.config)
    snapshot = json.loads(Path(args.config).read_text(encoding="utf-8"))
    audit = Path(args.dependency_audit)
    if not audit.is_file():
        raise FileNotFoundError(f"TASK-010 dependency audit not found: {audit}")
    dependency_hash = _sha256(audit)
    ppo_kwargs = {
        "num_learning_epochs": config.ppo.num_learning_epochs,
        "num_mini_batches": config.ppo.num_mini_batches,
        "clip_param": config.ppo.clip_param,
        "value_loss_coef": config.ppo.value_loss_coef,
        "learning_rate": config.ppo.learning_rate,
        "desired_kl": config.ppo.desired_kl,
        "max_grad_norm": config.ppo.max_grad_norm,
        "entropy_coef": config.ppo.entropy_coef,
    }
    runner = Task010OnPolicyRunner(
        Task010Actor(), Task010Critic(), output_dir=args.output_dir,
        config_hash=config.config_sha256, config_snapshot=snapshot,
        dependency_audit_hash=dependency_hash,
        seed=int(config.training.seed if args.seed is None else args.seed),
        device=device, ppo_kwargs=ppo_kwargs,
    )
    if args.resume_checkpoint is not None:
        runner.load(args.resume_checkpoint)
    return config, runner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--validation", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--backend", choices=("fake", "isaac"), default="isaac")
    parser.add_argument(
        "--dependency-audit", type=Path,
        default=Path("/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/task010_cnn_gru/gate0/prerequisites.json"),
    )
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _parser()
    preliminary, _ = parser.parse_known_args()
    if preliminary.backend == "fake":
        args, unknown = parser.parse_known_args()
        if any(not item.startswith("--kit_args") for item in unknown):
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        config, runner = _build_runner(args, "cpu")
        updates = config.training.max_updates if args.max_updates is None else int(args.max_updates)
        interval = config.checkpoints.rolling_interval if args.save_interval is None else int(args.save_interval)
        runner.learn_fake(
            num_updates=updates, rollout_steps=config.ppo.rollout_steps,
            num_envs=config.training.num_envs, save_interval=interval,
        )
        return

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    import gymnasium as gym
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg

    config, runner = _build_runner(args, args.device)
    cfg = parse_env_cfg(config.task_id, device=args.device, num_envs=config.training.num_envs)
    updates = config.training.max_updates if args.max_updates is None else int(args.max_updates)
    interval = config.checkpoints.rolling_interval if args.save_interval is None else int(args.save_interval)
    with launch_simulation(app, cfg):
        env = gym.make(config.task_id, cfg=cfg).unwrapped
        try:
            runner.learn_environment(env, num_updates=updates, rollout_steps=config.ppo.rollout_steps, save_interval=interval)
        finally:
            env.close()
    app.close()


if __name__ == "__main__":
    main()
