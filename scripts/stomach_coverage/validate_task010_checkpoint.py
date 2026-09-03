#!/usr/bin/env python3
"""Run deterministic TASK-010 validation on the frozen twenty validation poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_task010_validation import VALIDATION_POSE_IDS, file_sha256, summarize, validation_batches


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--visual-condition",
        choices=("normal", "blind", "donor", "first_frame"),
        default="normal",
    )
    result.add_argument("--experiment-config", type=Path)
    result.add_argument("--training-seed", type=int)
    result.add_argument("--save-feature-bank", type=Path)
    result.add_argument("--donor-bank", type=Path)
    return result


def main() -> None:
    from isaaclab.app import AppLauncher

    command = parser()
    AppLauncher.add_app_launcher_args(command)
    args = command.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import robotarm_magnetic_lab.tasks  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
    from robotarm_magnetic_lab.runtime.task010_config import load_task010_config
    from robotarm_magnetic_lab.runtime.task010_feature_bank import (
        load_pose_feature_sequence,
        manifest_sha256,
        save_pose_feature_sequence,
    )
    from robotarm_magnetic_lab.runtime.task010_visual_dependence_config import (
        VISUAL_DEPENDENCE_CONFIG_PATH,
        load_visual_dependence_config,
    )
    from robotarm_magnetic_lab.runtime.task010_visual_intervention import (
        Task010VisualIntervention,
    )

    frozen = load_task010_config(args.config)
    experiment = None
    experiment_config_path = args.experiment_config
    if experiment_config_path is None and args.visual_condition != "normal":
        experiment_config_path = VISUAL_DEPENDENCE_CONFIG_PATH
    if experiment_config_path is not None:
        experiment = load_visual_dependence_config(experiment_config_path)
    if args.visual_condition == "donor" and args.donor_bank is None:
        raise RuntimeError("donor condition requires --donor-bank")
    if args.visual_condition != "donor" and args.donor_bank is not None:
        raise RuntimeError("--donor-bank is only valid with donor condition")
    if args.visual_condition != "normal" and args.save_feature_bank is not None:
        raise RuntimeError("--save-feature-bank is only valid with normal condition")
    if args.save_feature_bank is not None and experiment is None:
        raise RuntimeError("--save-feature-bank requires --experiment-config")
    record = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if record.get("config_hash") != frozen.config_sha256:
        raise RuntimeError("TASK-010 validation checkpoint/config mismatch")
    checkpoint_hash = file_sha256(args.checkpoint)
    training_seed = args.training_seed
    if training_seed is None:
        training_seed = record.get("seed")
    checkpoint_update = record.get("current_update")
    if experiment is not None and checkpoint_update not in (750, 1000):
        raise RuntimeError("visual-dependence validation requires update 750 or 1000")
    experiment_config_sha256 = (
        experiment.config_sha256 if experiment is not None else None
    )
    donor_manifest_sha256 = None
    if args.donor_bank is not None:
        donor_manifest_sha256 = manifest_sha256(Path(args.donor_bank))
    if args.visual_condition == "donor" and experiment is None:
        raise RuntimeError("donor condition requires --experiment-config")
    actor = Task010Actor().to(args.device)
    actor.load_state_dict(record["actor"], strict=True)
    actor.eval()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "pose_records.jsonl"
    trajectory_path = output / "coverage_trajectories.jsonl"
    for stale in (records_path, trajectory_path):
        if stale.exists():
            stale.unlink()

    cfg = parse_env_cfg(frozen.task_id, device=args.device, num_envs=12)
    cfg.pose_split = "validation"
    cfg.explicit_pose_ids = tuple(validation_batches()[0])
    with launch_simulation(app, cfg):
        env = gym.make(frozen.task_id, cfg=cfg).unwrapped
        try:
            for batch_index, batch in enumerate(validation_batches()):
                padded = tuple(batch) + tuple(batch[:1]) * (12 - len(batch))
                env.cfg.explicit_pose_ids = padded
                observations, extras = env.reset(seed=frozen.training.seed + batch_index)
                actual = tuple(extras["task009d0_reset"]["pose_ids"])
                if actual[: len(batch)] != tuple(batch):
                    raise RuntimeError("TASK-010 validation reset returned different pose IDs")
                actor.reset()
                first_frame = Task010VisualIntervention(
                    "first_frame", num_envs=len(batch), feature_dim=512
                )
                saved_features = None
                if args.save_feature_bank is not None:
                    saved_features = torch.empty(
                        (len(batch), 1200, 512), dtype=torch.float32
                    )
                donor_features = None
                if args.visual_condition == "donor":
                    assert experiment is not None
                    donor_tensors = []
                    for pose_id in batch:
                        donor_pose_id = experiment.donor_pose_by_target[pose_id]
                        donor_tensors.append(
                            load_pose_feature_sequence(
                                Path(args.donor_bank),
                                donor_pose_id,
                                {
                                    "pose_id": donor_pose_id,
                                    "training_seed": training_seed,
                                    "checkpoint_update": checkpoint_update,
                                    "checkpoint_sha256": checkpoint_hash,
                                    "base_config_sha256": frozen.config_sha256,
                                    "visual_dependence_config_sha256": experiment_config_sha256,
                                    "feature_steps": 1200,
                                    "feature_dim": 512,
                                },
                            )
                        )
                    donor_features = torch.stack(donor_tensors, dim=0)
                actor_totals = torch.zeros((len(batch),), device=args.device)
                alpha_totals = torch.zeros_like(actor_totals)
                mode_counts = torch.zeros((len(batch), 6), device=args.device)
                coverage_trajectories = torch.empty((len(batch), 1201), device=args.device, dtype=torch.float64)
                coverage_trajectories[:, 0] = torch.as_tensor(
                    extras["task009d0_reset"]["initial_coverage"][: len(batch)],
                    device=args.device,
                    dtype=torch.float64,
                )
                terminal = None
                for step_index in range(1200):
                    target_observation = observations["policy"][: len(batch)]
                    actor_input = observations["policy"].clone()
                    if args.visual_condition == "normal":
                        replacement = target_observation[:, :512]
                    elif args.visual_condition == "blind":
                        replacement = torch.zeros_like(target_observation[:, :512])
                    elif args.visual_condition == "donor":
                        assert donor_features is not None
                        replacement = donor_features[:, step_index, :].to(args.device)
                    elif args.visual_condition == "first_frame":
                        replacement = first_frame.apply(target_observation[:, :512])
                    else:
                        raise AssertionError(args.visual_condition)
                    actor_input[: len(batch), :512] = replacement.to(
                        device=actor_input.device, dtype=actor_input.dtype
                    )
                    if saved_features is not None:
                        saved_features[:, step_index, :] = (
                            target_observation[:, :512].detach().cpu()
                        )
                    # Keep only the Actor forward pass gradient-free.  Wrapping
                    # env.step() in inference_mode makes mutable coverage state
                    # into inference tensors, which cannot be reset in-place
                    # before the second (8-pose) validation batch.
                    with torch.no_grad():
                        action = actor(actor_input, stochastic_output=False)
                    observations, reward, terminated, truncated, step_extras = env.step(action)
                    actor_totals += reward[: len(batch)]
                    alpha_totals += action[: len(batch), 1]
                    mode_counts.scatter_add_(1, action[: len(batch), :1].long(), torch.ones((len(batch), 1), device=args.device))
                    terminal_boundary = bool(
                        torch.any(terminated[: len(batch)] | truncated[: len(batch)]).item()
                    )
                    # Manager extras persist across an explicit reset.  A
                    # terminal-audit key can therefore belong to the previous
                    # validation batch; consume it only when this step itself
                    # carries a terminal/truncation flag.
                    terminal_audit = (
                        step_extras.get("task009d0_terminal_audit")
                        if terminal_boundary
                        else None
                    )
                    runtime = env._task009d0_coverage_runtime
                    current_coverage = (
                        terminal_audit["reachable_coverage"]
                        if terminal_audit is not None
                        # Snapshot the authoritative float64 cumulative mask.
                        # Both latest_update and the reward tracker can retain a
                        # cached row value across the explicit 12->8 batch reset.
                        else runtime._snapshot(
                            runtime.reachable_accumulator
                        ).coverage_fraction
                    )
                    coverage_trajectories[:, step_index + 1] = current_coverage[: len(batch)]
                    if terminal_boundary:
                        terminal = terminal_audit
                if terminal is None:
                    raise RuntimeError("TASK-010 validation did not reach true terminal")
                feature_bank_manifest_sha256 = None
                if args.save_feature_bank is not None and saved_features is not None:
                    assert experiment is not None
                    for row, pose_id in enumerate(batch):
                        save_pose_feature_sequence(
                            Path(args.save_feature_bank),
                            {
                                "pose_id": pose_id,
                                "training_seed": training_seed,
                                "checkpoint_update": checkpoint_update,
                                "checkpoint_sha256": checkpoint_hash,
                                "base_config_sha256": frozen.config_sha256,
                                "visual_dependence_config_sha256": experiment_config_sha256,
                                "feature_steps": 1200,
                                "feature_dim": 512,
                            },
                            saved_features[row],
                        )
                    feature_bank_manifest_sha256 = manifest_sha256(
                        Path(args.save_feature_bank)
                    )
                for row, pose_id in enumerate(batch):
                    payload = {
                        "schema": "robotarm_magnetic_lab.task010_validation_pose",
                        "pose_id": pose_id,
                        "formal_steps": 1200,
                        "checkpoint_sha256": checkpoint_hash,
                        "config_sha256": frozen.config_sha256,
                        "visual_condition": args.visual_condition,
                        "training_seed": training_seed,
                        "checkpoint_update": checkpoint_update,
                        "donor_pose_id": (
                            experiment.donor_pose_by_target[pose_id]
                            if args.visual_condition == "donor" and experiment is not None
                            else None
                        ),
                        "previous_action_source": "target_environment",
                        "feature_bank_manifest_sha256": (
                            feature_bank_manifest_sha256
                            if args.save_feature_bank is not None
                            else donor_manifest_sha256
                        ),
                        "experiment_config_sha256": experiment_config_sha256,
                        "final_coverage": float(terminal["reachable_coverage"][row].item()),
                        "total_reward": float(actor_totals[row].item()),
                        "mean_alpha": float(alpha_totals[row].item() / 1200.0),
                        "mode_fraction": (mode_counts[row] / 1200.0).cpu().tolist(),
                        "failure_reason": None,
                    }
                    with records_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(payload, sort_keys=True) + "\n")
                    trajectory = {
                        "schema": "robotarm_magnetic_lab.task010_validation_coverage_trajectory",
                        "pose_id": pose_id,
                        "checkpoint_sha256": checkpoint_hash,
                        "config_sha256": frozen.config_sha256,
                        "visual_condition": args.visual_condition,
                        "training_seed": training_seed,
                        "checkpoint_update": checkpoint_update,
                        "experiment_config_sha256": experiment_config_sha256,
                        "control_hz": 10,
                        "coverage_fraction": coverage_trajectories[row].cpu().tolist(),
                    }
                    with trajectory_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(trajectory, sort_keys=True) + "\n")
        finally:
            env.close()
    result = summarize(records_path, checkpoint_sha256=checkpoint_hash, config_sha256=frozen.config_sha256)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    app.close()


if __name__ == "__main__":
    main()
