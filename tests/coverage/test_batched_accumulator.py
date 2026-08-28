from __future__ import annotations

import numpy as np
import pytest
import torch

from robotarm_magnetic_lab.coverage.accumulator import CoverageAccumulator
from robotarm_magnetic_lab.coverage.batched_accumulator import BatchedCoverageAccumulator


def test_duplicate_frame_is_ignored_per_row_and_reset_is_isolated():
    acc = BatchedCoverageAccumulator(
        torch.tensor([1.0, 2.0, 3.0]), num_envs=2, device="cpu"
    )
    acc.update(
        torch.tensor([10, 10]),
        torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool),
    )
    second = acc.update(
        torch.tensor([10, 11]),
        torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.bool),
    )
    assert second.updated.tolist() == [False, True]
    assert second.visible_count.tolist() == [0, 1]
    assert second.newly_covered_count.tolist() == [0, 1]
    before = acc.mask[1].clone()
    acc.reset_rows(torch.tensor([0]))
    assert not acc.mask[0].any()
    torch.testing.assert_close(acc.mask[1], before)


def test_decreasing_frame_id_and_invalid_shapes_are_rejected():
    acc = BatchedCoverageAccumulator(torch.ones(3), num_envs=2, device="cpu")
    acc.update(torch.tensor([2, 2]), torch.zeros((2, 3), dtype=torch.bool))
    with pytest.raises(RuntimeError, match="decreased"):
        acc.update(torch.tensor([1, 3]), torch.zeros((2, 3), dtype=torch.bool))
    with pytest.raises(ValueError, match="shapes"):
        acc.update(torch.tensor([3]), torch.zeros((2, 3), dtype=torch.bool))


def test_each_row_matches_legacy_numpy_accumulator():
    weights = np.asarray([0.125, 0.25, 0.5, 1.0, 2.0], dtype=np.float64)
    visible = torch.tensor(
        [
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
            [[0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 1]],
            [[0, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 0, 0, 1]],
            [[1, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 1, 0]],
            [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0]],
        ],
        dtype=torch.bool,
    )
    batched = BatchedCoverageAccumulator(
        torch.from_numpy(weights), num_envs=3, device="cpu"
    )
    legacy = [CoverageAccumulator(len(weights), weights) for _ in range(3)]
    for frame_id, masks in enumerate(visible):
        result = batched.update(torch.full((3,), frame_id), masks)
        for env_id in range(3):
            indices = torch.nonzero(masks[env_id], as_tuple=False).reshape(-1).tolist()
            old = legacy[env_id].update(frame_id, indices)
            np.testing.assert_array_equal(batched.mask[env_id].numpy(), legacy[env_id].mask)
            assert result.visible_area_m2[env_id].item() == pytest.approx(
                old.visible_area_m2, abs=1.0e-15
            )
            assert result.newly_covered_area_m2[env_id].item() == pytest.approx(
                old.newly_covered_area_m2, abs=1.0e-15
            )
            assert result.cumulative_area_m2[env_id].item() == pytest.approx(
                old.cumulative_area_m2, abs=1.0e-15
            )


def test_weights_require_finite_nonnegative_positive_total():
    for weights in (
        torch.tensor([]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, float("nan")]),
    ):
        with pytest.raises(ValueError, match="weights"):
            BatchedCoverageAccumulator(weights, num_envs=2, device="cpu")
