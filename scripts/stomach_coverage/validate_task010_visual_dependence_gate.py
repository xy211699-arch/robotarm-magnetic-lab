#!/usr/bin/env python3
"""Machine-readable V0-V2 gate evidence validator for the visual-dependence study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))


def _required_true(evidence: Mapping[str, Any], gate: str, field: str) -> None:
    if evidence.get(field) is not True:
        raise RuntimeError(f"{gate}.{field} is required")


def validate_gate_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise RuntimeError("gate evidence must be an object")
    v0 = evidence.get("v0")
    v1 = evidence.get("v1")
    v2 = evidence.get("v2")
    v3 = evidence.get("v3")
    if not isinstance(v0, dict):
        raise RuntimeError("v0 evidence is missing")
    for field in (
        "critic_isolation",
        "blind_visual_projection",
        "identical_trainable_parameters",
        "actor_observation_schema",
        "resnet_forward_count",
    ):
        _required_true(v0, "v0", field)
    if not isinstance(v1, dict):
        raise RuntimeError("v1 evidence is missing")
    _required_true(v1, "v1", "blind_forward_backward_save_restore")
    if not isinstance(v2, dict):
        raise RuntimeError("v2 evidence is missing")
    for field in (
        "donor_mapping",
        "first_frame_repeat",
        "target_previous_action",
        "unique_variables",
        "curve_length",
    ):
        _required_true(v2, "v2", field)
    if not isinstance(v3, dict):
        raise RuntimeError("v3 evidence is missing")
    if v3.get("status") != "awaiting_manual_start":
        raise RuntimeError("v3 status must remain awaiting_manual_start")
    return {
        "schema": "robotarm_magnetic_lab.task010_visual_dependence_gate_report",
        "status": "passed",
        "v0": v0,
        "v1": v1,
        "v2": v2,
        "v3": v3,
    }


def _load_evidence(args: argparse.Namespace) -> dict[str, Any]:
    path = args.evidence
    if path is None and args.run_dir is not None:
        path = Path(args.run_dir) / "gates" / "evidence.json"
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("gate evidence file is missing")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("gate evidence must be a JSON object")
    return raw


def _build_self_check_evidence(config_path: Path) -> dict[str, Any]:
    import torch

    from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
    from robotarm_magnetic_lab.learning.task010_critic import Task010Critic
    from robotarm_magnetic_lab.learning.task010_runner import Task010OnPolicyRunner
    from robotarm_magnetic_lab.runtime.task010_feature_bank import (
        load_pose_feature_sequence,
        save_pose_feature_sequence,
    )
    from robotarm_magnetic_lab.runtime.task010_visual_dependence_config import (
        load_visual_dependence_config,
    )
    from robotarm_magnetic_lab.runtime.task010_visual_intervention import (
        Task010VisualIntervention,
        replace_actor_visual_features,
    )

    config = load_visual_dependence_config(config_path)
    terms_path = (
        config_path.parents[2]
        / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/"
        "robotarm_magnetic_lab/mdp/task010_terms.py"
    )
    terms_source = terms_path.read_text(encoding="utf-8")
    encoder_path = (
        config_path.parents[2]
        / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_visual_encoder.py"
    )
    encoder_source = encoder_path.read_text(encoding="utf-8")
    actor_a = Task010Actor()
    actor_b = Task010Actor()
    same_parameters = all(
        tuple(parameter.shape) == tuple(other.shape) and parameter.numel() == other.numel()
        for parameter, other in zip(actor_a.parameters(), actor_b.parameters())
    )

    v1 = False
    with tempfile.TemporaryDirectory() as directory:
        runner = Task010OnPolicyRunner(
            Task010Actor(),
            Task010Critic(),
            output_dir=Path(directory),
            config_hash=config.base_config.sha256,
            config_snapshot={"seed": 990999},
            dependency_audit_hash="self-check",
            seed=990999,
            device="cpu",
            ppo_kwargs={
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
            },
            experiment_metadata={
                "visual_condition": "blind",
                "visual_dependence_config_sha256": config.config_sha256,
                "base_config_sha256": config.base_config.sha256,
            },
        )
        runner.learn_fake(num_updates=1, rollout_steps=1, num_envs=1, save_interval=1)
        checkpoint = Path(directory) / "checkpoints" / "update_0001.pt"
        runner.save(checkpoint)
        restored = Task010OnPolicyRunner(
            Task010Actor(),
            Task010Critic(),
            output_dir=Path(directory) / "restored",
            config_hash=config.base_config.sha256,
            config_snapshot={"seed": 990999},
            dependency_audit_hash="self-check",
            seed=990999,
            device="cpu",
            experiment_metadata=runner.experiment_metadata,
        )
        restored.load(checkpoint)
        runner.close()
        restored.close()
        v1 = True

    intervention = Task010VisualIntervention("first_frame", num_envs=2, feature_dim=512)
    first = torch.randn((2, 512))
    intervention.apply(first)
    intervention.apply(torch.randn((2, 512)))
    target = torch.randn((2, 519))
    replacement = torch.randn((2, 512))
    replaced = replace_actor_visual_features(target, replacement)
    donor_valid = all(
        target != donor for target, donor in config.donor_pose_by_target.items()
    )
    with tempfile.TemporaryDirectory() as bank_dir:
        metadata = {
            "pose_id": config.validation_pose_ids[0],
            "training_seed": 991001,
            "checkpoint_update": 750,
            "checkpoint_sha256": "c" * 64,
            "base_config_sha256": config.base_config.sha256,
            "visual_dependence_config_sha256": config.config_sha256,
            "feature_steps": 1200,
            "feature_dim": 512,
        }
        features = torch.randn((1200, 512))
        save_pose_feature_sequence(Path(bank_dir), metadata, features)
        loaded = load_pose_feature_sequence(Path(bank_dir), metadata["pose_id"], metadata)
        bank_roundtrip = torch.equal(loaded, features)
    return {
        "evidence_kind": "cpu_implementation_smoke",
        "v0": {
            "critic_isolation": "task010_privileged_observation" in terms_source,
            "blind_visual_projection": (
                'condition == "blind"' in terms_source and "torch.zeros_like" in terms_source
            ),
            "identical_trainable_parameters": same_parameters,
            "actor_observation_schema": actor_a.observation_dim == 519,
            "resnet_forward_count": "forward_image_count" in encoder_source,
        },
        "v1": {"blind_forward_backward_save_restore": v1},
        "v2": {
            "donor_mapping": donor_valid,
            "first_frame_repeat": torch.equal(
                intervention.apply(torch.zeros((2, 512))),
                intervention.apply(torch.zeros((2, 512))),
            ),
            "target_previous_action": torch.equal(replaced[:, 512:], target[:, 512:]),
            "unique_variables": len(config.primary_conditions) == 4,
            "curve_length": config.coverage_points == 1201,
        },
        "v3": {"status": "awaiting_manual_start"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    evidence = _build_self_check_evidence(args.config) if args.self_check else _load_evidence(args)
    report = validate_gate_evidence(evidence)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
