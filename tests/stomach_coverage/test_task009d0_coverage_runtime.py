from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robotarm_magnetic_lab.coverage.reference_mesh import ReferenceMesh
from robotarm_magnetic_lab.runtime.task009d0_coverage_runtime import (
    Task009D0CoverageRuntime,
    Task009D0RgbSynchronizer,
)


class _FakeCamera:
    def __init__(self, frame):
        self.frame = torch.as_tensor(frame, dtype=torch.int64)
        self._ALL_ENV_MASK = torch.ones(len(frame), dtype=torch.bool)
        self.forced = 0

    def _update_buffers_impl(self, _mask):
        self.frame += 1
        self.forced += 1


def _reference():
    return ReferenceMesh(
        vertices_world=np.asarray(
            [[0.0, 0.0, 0.03], [0.01, 0.0, 0.03], [0.0, 0.01, 0.03]],
            dtype=np.float64,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
        incident_triangles=((0,), (0,), (0,)),
        selected_prim_paths=("/Inner",),
        weld_tolerance_m=1.0e-6,
        geometry_sha256="fixture",
    )


def test_all_stale_frames_are_forced_once_and_partial_staleness_fails():
    sync = Task009D0RgbSynchronizer(num_envs=2)
    camera = _FakeCamera([4, 4])
    assert sync.observe(boundary=1, camera=camera).tolist() == [4, 4]
    assert sync.observe(boundary=1, camera=camera).tolist() == [4, 4]
    assert camera.forced == 0
    camera.frame = torch.tensor([4, 4])
    assert sync.observe(boundary=2, camera=camera).tolist() == [5, 5]
    assert camera.forced == 1
    camera.frame = torch.tensor([6, 5])
    with pytest.raises(RuntimeError, match="partial camera advancement"):
        sync.observe(boundary=3, camera=camera)


def test_environment_local_translation_c0_and_row_reset_are_isolated():
    calls = []

    def visibility(centers_local, axes):
        calls.append((centers_local.clone(), axes.clone()))
        torch.testing.assert_close(centers_local[0], centers_local[1])
        return torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)

    runtime = Task009D0CoverageRuntime(
        reference_local=_reference(),
        env_origins=torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        raw_vertex_weights=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        reachable_vertex_weights=torch.tensor([1.0, 0.0, 3.0], dtype=torch.float64),
        device="cpu",
        visibility_override=visibility,
    )
    centers_world = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    axes = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    initial = runtime.capture_initial(
        frame_ids=torch.tensor([10, 10]),
        camera_centers_world=centers_world,
        optical_axes_world=axes,
    )
    assert initial.initial
    assert initial.new_coverage_reward_m2.tolist() == [0.0, 0.0]
    assert initial.reachable.coverage_fraction.tolist() == [0.25, 0.25]

    def second_visibility(centers_local, axes):
        return torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.bool)

    runtime.visibility_override = second_visibility
    update = runtime.update_boundary(
        boundary=1,
        frame_ids=torch.tensor([11, 11]),
        camera_centers_world=centers_world,
        optical_axes_world=axes,
    )
    assert update.new_coverage_reward_m2.tolist() == [3.0, 3.0]
    assert runtime.update_boundary(
        boundary=1,
        frame_ids=torch.tensor([11, 11]),
        camera_centers_world=centers_world,
        optical_axes_world=axes,
    ) is update
    runtime.reset_rows(torch.tensor([0]))
    assert not runtime.reachable_accumulator.mask[0].any()
    assert runtime.reachable_accumulator.mask[1].any()


def test_stabilizing_boundary_suppresses_accumulation():
    runtime = Task009D0CoverageRuntime(
        reference_local=_reference(),
        env_origins=torch.zeros((1, 3)),
        raw_vertex_weights=torch.ones(3, dtype=torch.float64),
        reachable_vertex_weights=torch.ones(3, dtype=torch.float64),
        device="cpu",
        visibility_override=lambda *_: torch.ones((1, 3), dtype=torch.bool),
    )
    update = runtime.update_boundary(
        boundary=1,
        frame_ids=torch.tensor([1]),
        camera_centers_world=torch.zeros((1, 3)),
        optical_axes_world=torch.tensor([[0.0, 0.0, 1.0]]),
        stabilizing=True,
    )
    assert update.stabilizing
    assert not runtime.reachable_accumulator.mask.any()
