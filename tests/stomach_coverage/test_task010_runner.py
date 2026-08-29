from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from robotarm_magnetic_lab.learning.task010_actor import Task010Actor
from robotarm_magnetic_lab.learning.task010_critic import Task010Critic
from robotarm_magnetic_lab.learning.task010_runner import Task010OnPolicyRunner


def _runner(tmp_path: Path, *, seed: int = 991000, config_hash: str = "config"):
    return Task010OnPolicyRunner(
        Task010Actor(), Task010Critic(), output_dir=tmp_path,
        config_hash=config_hash, config_snapshot={"seed": seed},
        dependency_audit_hash="dependency", seed=seed, device="cpu",
    )


def _draw_all_rngs():
    return (random.random(), float(np.random.random()), torch.rand(3))


def test_checkpoint_roundtrip_restores_rng_and_update(tmp_path: Path):
    runner = _runner(tmp_path / "first")
    runner.current_update = 2
    runner.total_transitions = 1536
    path = tmp_path / "update_0002.pt"
    runner.save(path)
    expected = _draw_all_rngs()
    restored = _runner(tmp_path / "second", seed=1)
    restored.load(path)
    assert restored.current_update == 2
    assert restored.total_transitions == 1536
    assert restored.actor.get_hidden_state().numel() == 0
    actual = _draw_all_rngs()
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_resume_rejects_config_hash_mismatch(tmp_path: Path):
    first = _runner(tmp_path / "first", config_hash="a")
    checkpoint = tmp_path / "checkpoint.pt"
    first.save(checkpoint)
    with pytest.raises(ValueError, match="config hash mismatch"):
        _runner(tmp_path / "second", config_hash="b").load(checkpoint)


def test_fake_training_writes_one_fsynced_metric_per_update(tmp_path: Path):
    runner = _runner(tmp_path)
    runner.learn_fake(num_updates=2, rollout_steps=4, num_envs=4)
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in metrics] == [1, 2]
    assert all(row["all_finite"] for row in metrics)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "runner_initialized"


def test_checkpoint_contains_required_contract_fields(tmp_path: Path):
    runner = _runner(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    runner.save(checkpoint)
    record = torch.load(checkpoint, weights_only=False)
    for key in (
        "actor", "critic", "optimizer", "current_update", "total_transitions",
        "rng", "config_hash", "config_snapshot", "git_commit",
        "dependency_audit_hash", "actor_observation_schema_sha256", "action_schema_sha256",
    ):
        assert key in record


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA map-location regression")
def test_rng_restore_accepts_checkpoint_state_mapped_to_cuda(tmp_path: Path):
    runner = _runner(tmp_path)
    state = runner._rng_state()
    state["torch_cpu"] = state["torch_cpu"].cuda()
    state["torch_cuda"] = [item.cuda() for item in state["torch_cuda"]]
    runner._restore_rng(state)
