from __future__ import annotations

import pytest
import torch

from robotarm_magnetic_lab.runtime.task010_feature_bank import (
    load_pose_feature_sequence,
    save_pose_feature_sequence,
)


def _metadata(pose_id="pose-01"):
    return {
        "pose_id": pose_id,
        "training_seed": 991001,
        "checkpoint_update": 750,
        "checkpoint_sha256": "c" * 64,
        "base_config_sha256": "b" * 64,
        "visual_dependence_config_sha256": "v" * 64,
        "feature_steps": 1200,
        "feature_dim": 512,
    }


def test_feature_bank_round_trip_requires_1200_by_512(tmp_path):
    features = torch.randn(1200, 512)
    path = save_pose_feature_sequence(
        tmp_path, _metadata("pose-01"), features
    )
    assert path.name == "pose-01.pt"
    assert torch.equal(
        load_pose_feature_sequence(tmp_path, "pose-01", _metadata("pose-01")),
        features,
    )


def test_feature_bank_rejects_wrong_shape(tmp_path):
    features = torch.randn(1199, 512)
    with pytest.raises(ValueError, match="1200, 512"):
        save_pose_feature_sequence(tmp_path, _metadata("bad"), features)


def test_feature_bank_rejects_metadata_mismatch(tmp_path):
    features = torch.randn(1200, 512)
    save_pose_feature_sequence(tmp_path, _metadata("pose-01"), features)
    wrong = _metadata("pose-01")
    wrong["checkpoint_update"] = 1000
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_pose_feature_sequence(tmp_path, "pose-01", wrong)
