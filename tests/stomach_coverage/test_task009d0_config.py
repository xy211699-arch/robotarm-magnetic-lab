from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotarm_magnetic_lab.runtime.task009d0_config import load_task009d0_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task009d0/vectorized_training_candidates_v1.json"


def test_candidate_config_freezes_non_destructive_contract():
    cfg = load_task009d0_config(CONFIG_PATH)
    assert cfg["num_env_candidates"] == [1, 2, 4, 8]
    assert cfg["episode"]["formal_steps"] == 1200
    assert cfg["episode"]["hold_steps"] == 10
    assert cfg["camera"] == {
        "width": 1280,
        "height": 720,
        "hz": 10,
        "fov_deg": 120.0,
    }
    assert cfg["coverage"]["max_distance_m"] == 0.07
    assert cfg["benchmark"]["warmup_steps"] == 50
    assert cfg["benchmark"]["measured_steps"] == 300
    assert cfg["benchmark"]["repeats"] == 3
    assert cfg["benchmark"]["minimum_free_memory_fraction"] == 0.20
    assert cfg["benchmark"]["near_tie_fraction"] == 0.10


def test_candidate_config_rejects_unknown_top_level_key(tmp_path):
    record = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    record["misspelled_contract"] = True
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_task009d0_config(path)


def test_frozen_loader_requires_selected_num_envs(tmp_path):
    record = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "not-frozen.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="selected_num_envs"):
        load_task009d0_config(path, frozen=True)
