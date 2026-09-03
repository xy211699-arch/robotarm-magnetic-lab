from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stomach_coverage"))

import train_task010  # noqa: E402


def _required_args():
    return [
        "--config",
        str(ROOT / "configs/task010/cnn_gru_development_v1.json"),
        "--output-dir",
        "/tmp/task010-visual-condition-cli",
    ]


def test_training_cli_accepts_only_normal_and_blind():
    assert (
        train_task010._parser().parse_args(
            ["--visual-condition", "blind", *_required_args()]
        ).visual_condition
        == "blind"
    )
    with pytest.raises(SystemExit):
        train_task010._parser().parse_args(
            ["--visual-condition", "donor", *_required_args()]
        )


def test_default_training_condition_is_normal():
    args = train_task010._parser().parse_args(_required_args())
    assert args.visual_condition == "normal"
