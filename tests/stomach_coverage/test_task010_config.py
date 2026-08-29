from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotarm_magnetic_lab.runtime.task010_config import (
    TASK010_CONFIG_PATH,
    canonical_config_sha256,
    load_task010_config,
)


def test_development_config_is_frozen():
    cfg = load_task010_config(TASK010_CONFIG_PATH)
    assert (cfg.training.seed, cfg.training.num_envs) == (991000, 12)
    assert (cfg.ppo.rollout_steps, cfg.training.max_updates) == (64, 1000)
    assert cfg.validation.updates == (250, 500, 750, 1000)
    assert cfg.model.resnet_weights == "IMAGENET1K_V1"
    assert cfg.model.actor_observation_dim == 519
    assert cfg.model.critic_observation_dim == 65
    assert cfg.ppo.gamma == 0.999
    assert cfg.ppo.lam == 0.95
    assert cfg.checkpoints.rolling_interval == 50
    assert cfg.checkpoints.permanent_updates == (250, 500, 750, 1000)
    assert cfg.config_sha256 == canonical_config_sha256(cfg)


def test_unknown_or_overridden_frozen_field_is_rejected(tmp_path: Path):
    raw = json.loads(TASK010_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["augmentation"] = {"random_crop": True}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="augmentation must remain disabled"):
        load_task010_config(path)


def test_frozen_action_and_clock_contract():
    cfg = load_task010_config(TASK010_CONFIG_PATH)
    assert cfg.clocks.physics_hz == 240
    assert cfg.clocks.control_hz == 10
    assert cfg.clocks.physics_steps_per_action == 24
    assert cfg.episode.formal_steps == 1200
    assert cfg.action.mode_ids == (0, 1, 2, 3, 4, 5)
    assert cfg.action.force_ratio_mg["MOVE"] == (0.7, 1.4)
    assert cfg.action.force_ratio_mg["VIEW"] == (0.2, 0.5)
    assert cfg.action.force_ratio_mg["UP"] == (0.8, 1.05)
