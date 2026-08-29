from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from robotarm_magnetic_lab.coverage.reference_mesh import ReferenceMesh
from robotarm_magnetic_lab.runtime.task009d0_coverage_runtime import Task009D0CoverageRuntime
from robotarm_magnetic_lab.runtime.task010_privileged import TASK010_CRITIC_SLICES, Task010PrivilegedBuilder
from robotarm_magnetic_lab.runtime.task010_recovery import Task010RecoveryTracker


def _runtime() -> Task009D0CoverageRuntime:
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float64
    )
    reference = ReferenceMesh(
        vertices_world=vertices,
        triangles=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        incident_triangles=((0,), (0, 1), (0, 1), (1,)),
        selected_prim_paths=("/Inner",), weld_tolerance_m=1.0e-6, geometry_sha256="fixture",
    )
    runtime = Task009D0CoverageRuntime(
        reference_local=reference, env_origins=torch.zeros((2, 3)),
        raw_vertex_weights=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64),
        reachable_vertex_weights=torch.tensor([1.0, 1.0, 0.0, 3.0], dtype=torch.float64),
        device="cpu", visibility_override=lambda *_: torch.ones((2, 4), dtype=torch.bool),
    )
    runtime.capture_initial(
        boundary=0, frame_ids=torch.tensor([1, 1]),
        camera_centers_world=torch.zeros((2, 3)), optical_axes_world=torch.tensor([[0, 0, 1], [0, 0, 1]]),
    )
    return runtime


def test_coverage_grid_is_area_weighted_and_empty_cells_are_zero():
    runtime = _runtime()
    runtime.reachable_accumulator.reset_rows(torch.tensor([0, 1]))
    runtime.reachable_accumulator.update(
        torch.tensor([2, 2]),
        torch.tensor([[True, False, False, False], [False, True, False, True]]),
    )
    grid = runtime.coverage_grid_3x3x3()
    assert grid.shape == (2, 27)
    assert torch.isfinite(grid).all()
    assert (grid >= 0).all() and (grid <= 1).all()
    assert (grid[0] > 0).sum().item() == 1
    assert (grid[1] > 0).sum().item() == 2


class _Scene(dict):
    def __init__(self, *args, env_origins):
        super().__init__(*args)
        self.env_origins = env_origins


def test_privileged_schema_is_exact_complete_and_contact_safe():
    runtime = _runtime()
    pose = torch.tensor([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    velocity = torch.zeros((2, 6))
    capsule = SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=pose), root_com_vel_w=SimpleNamespace(torch=velocity)))
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros((2, 1, 3))))
    scene = _Scene({"capsule": capsule, "capsule_contact": sensor}, env_origins=torch.zeros((2, 3)))
    action = SimpleNamespace(previous_action_features=torch.zeros((2, 7)))
    env = SimpleNamespace(
        num_envs=2, device="cpu", scene=scene, _formal_step=10,
        _task009d0_coverage_runtime=runtime,
        action_manager=SimpleNamespace(get_term=lambda _name: action),
    )
    recovery = Task010RecoveryTracker(2).update(
        pose[:, :3], pose[:, 3:7], torch.zeros(2), dt_s=0.1
    )
    observation = Task010PrivilegedBuilder().build(env, recovery)
    assert observation.shape == (2, 65)
    occupied = [index for value in TASK010_CRITIC_SLICES.values() for index in range(value.start, value.stop)]
    assert occupied == list(range(65))
    assert torch.isfinite(observation).all()
    assert torch.equal(observation[:, TASK010_CRITIC_SLICES["wall_normal"]], torch.zeros((2, 3)))
    assert torch.equal(observation[:, TASK010_CRITIC_SLICES["wall_normal_valid"]], torch.zeros((2, 1)))
