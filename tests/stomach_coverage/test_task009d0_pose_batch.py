from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robotarm_magnetic_lab.runtime.task009d0_config import load_task009d0_config
from robotarm_magnetic_lab.runtime.task009d0_pose_batch import (
    Task009D0PoseBatchSampler,
    derived_env_episode_seed,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task009d0/vectorized_training_candidates_v1.json"


def _sampler(split: str, seed: int = 990009) -> Task009D0PoseBatchSampler:
    return Task009D0PoseBatchSampler.from_config(
        load_task009d0_config(CONFIG_PATH),
        authorized_split=split,
        training_seed=seed,
        repository_root=ROOT,
    )


def test_training_sampler_is_reproducible_and_rejects_split_leakage():
    first = _sampler("train")
    second = _sampler("train")
    ids = np.asarray([0, 1, 7], dtype=np.int64)
    episodes = np.asarray([3, 3, 3], dtype=np.int64)
    first_batch = first.sample(ids, episodes)
    second_batch = second.sample(ids, episodes)
    assert first_batch.pose_ids == second_batch.pose_ids
    np.testing.assert_array_equal(first_batch.rng_seeds, second_batch.rng_seeds)
    assert len(set(first_batch.rng_seeds.tolist())) == len(ids)
    with pytest.raises(ValueError, match="training sampler cannot access"):
        first.resolve_explicit(np.asarray([0]), ["validation-0006"])


def test_validation_loader_accepts_only_explicit_validation_ids():
    loader = _sampler("validation")
    batch = loader.resolve_explicit(np.asarray([0]), ["validation-0006"])
    assert batch.pose_ids == ("validation-0006",)
    assert batch.poses_world_xyzw.shape == (1, 7)
    with pytest.raises(ValueError, match="explicit pose split mismatch"):
        loader.resolve_explicit(np.asarray([0]), ["test-0001"])
    with pytest.raises(ValueError, match="sampling is available only"):
        loader.sample(np.asarray([0]), np.asarray([0]))


def test_seed_derivation_is_environment_and_episode_specific():
    values = {
        derived_env_episode_seed(990009, env_id, episode)
        for env_id in (0, 1, 7)
        for episode in (0, 3)
    }
    assert len(values) == 6


def test_pose_batch_rejects_shape_and_duplicate_rows():
    loader = _sampler("train")
    with pytest.raises(ValueError, match="one episode index"):
        loader.sample(np.asarray([0, 1]), np.asarray([0]))
    with pytest.raises(ValueError, match="unique"):
        loader.sample(np.asarray([0, 0]), np.asarray([0, 1]))
