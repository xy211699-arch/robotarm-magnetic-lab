from pathlib import Path

import json
import pytest

from robotarm_magnetic_lab.runtime.task010_config import load_task010_config
from robotarm_magnetic_lab.runtime.task010_visual_dependence_config import (
    load_visual_dependence_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/task010/visual_dependence_v1.json"
)
BASE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/task010/cnn_gru_development_v1.json"
)


def test_visual_dependence_matrix_is_frozen():
    cfg = load_visual_dependence_config(CONFIG_PATH)
    base = load_task010_config(BASE_CONFIG_PATH)
    assert cfg.formal_seeds == (991001, 991002, 991003)
    assert cfg.validation_pose_ids == base.validation.pose_ids
    assert cfg.primary_update == 750
    assert cfg.sensitivity_update == 1000
    assert cfg.training_conditions == ("blind",)
    assert cfg.primary_conditions == ("normal", "blind", "donor", "first_frame")
    assert cfg.sensitivity_conditions == ("normal", "blind")
    assert cfg.episode_steps == 1200
    assert cfg.coverage_points == 1201
    assert cfg.bootstrap_replicates == 10000


def test_donor_mapping_is_derangement_in_frozen_order():
    cfg = load_visual_dependence_config(CONFIG_PATH)
    expected = dict(
        zip(cfg.validation_pose_ids, cfg.validation_pose_ids[1:] + cfg.validation_pose_ids[:1])
    )
    assert cfg.donor_pose_by_target == expected
    assert all(target != donor for target, donor in expected.items())


def test_changed_or_unknown_config_field_is_rejected(tmp_path):
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["test_pose_ids"] = ["forbidden"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field"):
        load_visual_dependence_config(path)
