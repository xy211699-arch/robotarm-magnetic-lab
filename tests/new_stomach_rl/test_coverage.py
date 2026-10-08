"""Independent sampling clocks and exact existing geometric visibility gates."""
import numpy as np
import pytest
import torch

from robotarm_magnetic_lab.coverage.new_stomach_rl_coverage import (
    DualRateCoverage, GeometryVisibility, optical_from_capsule, validate_frozen_target,
)
from robotarm_magnetic_lab.coverage.reference_mesh import MeshInput, preprocess_reference_mesh


class FixtureVisibility:
    device = 'cpu'
    weights = torch.tensor([1., 2., 0.], dtype=torch.float64)
    def __call__(self, pose):
        return pose.to(torch.bool)


def test_dual_clocks_exact_counts_monotonic_and_reset():
    runtime = DualRateCoverage(FixtureVisibility(), 1, geometry_optics=lambda x: x)
    runtime.initialize(torch.tensor([[True, False, True]]))
    for tick in range(1, 28801):
        row = runtime.update_geometry_at_physics_tick(tick, torch.tensor([[False, True, True]]))
        if tick % 24:
            assert row is None
        if tick % 240 == 0:
            runtime.observe_camera_frame(tick // 240, tick / 240, torch.tensor([[False, True, True]]))
    assert len(runtime.c10_records) == 1201
    assert len(runtime.c1_records) == 121
    assert runtime.c10_records[-1]['coverage'] == [1.]
    runtime.reset([0])
    assert runtime.c10.mask.sum() == runtime.c1.mask.sum() == 0


def test_duplicate_boundary_is_not_counted_twice_and_c1_cannot_run_at_10hz():
    runtime = DualRateCoverage(FixtureVisibility(), 1, geometry_optics=lambda x: x)
    pose = torch.tensor([[True, False, False]])
    runtime.initialize(pose)
    runtime.update_geometry_at_physics_tick(24, pose)
    assert runtime.update_geometry_at_physics_tick(24, pose) is None
    with pytest.raises(ValueError, match='1Hz'):
        runtime.observe_camera_frame(1, .1, pose)
    with pytest.raises(ValueError):
        runtime.update_geometry_at_physics_tick(23, pose)


def test_c1_and_c10_use_same_common_boundary_visibility():
    runtime = DualRateCoverage(FixtureVisibility(), 1, geometry_optics=lambda x: x)
    pose = torch.tensor([[True, False, False]])
    runtime.initialize(pose)
    for tick in range(24, 241, 24):
        runtime.update_geometry_at_physics_tick(tick, pose)
    runtime.observe_camera_frame(1, 1., pose)
    assert torch.equal(runtime.c10.mask, runtime.c1.mask)
    with pytest.raises(RuntimeError, match='same boundary'):
        runtime.observe_camera_frame(2, 2., torch.tensor([[False, True, False]]))


def test_capsule_optics_use_xyzw_and_actual_rotated_offset():
    pose = torch.tensor([[1., 2., 3., 0., np.sqrt(.5), 0., np.sqrt(.5)]], dtype=torch.float64)
    center, axis = optical_from_capsule(pose)
    assert torch.allclose(center, torch.tensor([[1.-.0127, 2., 3.]], dtype=torch.float64))
    assert torch.allclose(axis, torch.tensor([[-1., 0., 0.]], dtype=torch.float64), atol=1e-12)


def test_frozen_target_rejects_old_geometry_before_raycast():
    with pytest.raises(ValueError, match='geometry'):
        validate_frozen_target('old-stomach', np.zeros(429029, dtype=np.uint8))


def test_existing_fov_distance_occlusion_normal_and_excluded_weights():
    vertices = np.array([[-.01, -.01, .03], [.01, -.01, .03], [0., .01, .03]])
    ref = preprocess_reference_mesh([MeshInput('/test', vertices, np.array([3]), np.array([0, 2, 1]), np.eye(4))], ['/test'])
    class Hits:
        occluded = False
        def query(self, centers, candidate_mask=None, target_vertices_local=None):
            distances = torch.linalg.vector_norm(target_vertices_local[None]-centers[:, None], dim=-1)
            if self.occluded:
                distances -= .005
            return distances, torch.zeros_like(distances, dtype=torch.int64)
    hits = Hits()
    visibility = GeometryVisibility(ref, np.array([1., 0., 1.]), hits, 'cpu')
    optical = (torch.zeros((1,3)), torch.tensor([[0.,0.,1.]]))
    assert visibility(optical).tolist() == [[True, False, True]]
    hits.occluded = True
    assert not visibility(optical).any()


def test_live_mount_reads_current_transform_without_rgb_or_frame_update(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import warp as wp
    monkeypatch.setattr(wp.config, 'kernel_cache_dir', str(tmp_path / 'warp'))
    from robotarm_magnetic_lab.runtime.new_stomach_rl_optics import LiveMountedOptics
    class View:
        calls = 0
        def get_world_poses(self):
            self.calls += 1
            return (SimpleNamespace(torch=torch.tensor([[0.,0.,-.0127]])),
                SimpleNamespace(torch=torch.tensor([[0.,0.,-1.,0.]])))
    class Camera:
        num_instances = 1
        frame = 7
        cfg = SimpleNamespace(offset=SimpleNamespace(pos=(0.,0.,-.0127), rot=(0.,1.,0.,0.)))
        _view = View()
        @property
        def data(self):
            raise AssertionError('10Hz geometry must not request an RGB sensor update')
    camera = Camera()
    sync_calls = []
    optics = LiveMountedOptics(camera, 'cpu', lambda: sync_calls.append(True))
    pose = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]])
    center, axis = optics(pose)
    assert camera.frame == 7 and camera._view.calls == 1 and len(sync_calls) == 1
    assert torch.allclose(axis, torch.tensor([[0.,0.,-1.]], dtype=torch.float64))
    pose[0,0] = .01
    with pytest.raises(RuntimeError, match='stale'):
        optics(pose)


def test_missing_rgb_is_repaired_once_only_at_one_second_boundary():
    from types import SimpleNamespace
    from robotarm_magnetic_lab.runtime.new_stomach_rl_rgb import PolicyRGBBoundary
    class Camera:
        cfg = SimpleNamespace(update_period=1.)
        frame = torch.tensor([10])
        data = SimpleNamespace(output={'rgb': torch.zeros((1,2,2,3), dtype=torch.uint8)})
        _ALL_ENV_MASK = object()
        calls = 0
        def _update_buffers_impl(self, mask):
            assert mask is self._ALL_ENV_MASK
            self.frame += 1; self.calls += 1
    camera = Camera()
    runtime = PolicyRGBBoundary(1, 'cpu')
    _, initial = runtime.sample(0, 0, camera)
    _, next_frame = runtime.sample(1, 240, camera)
    assert next_frame.item() == initial.item()+1 and camera.calls == 1
    with pytest.raises(ValueError):
        runtime.sample(1, 240, camera)
    assert camera.calls == 1
    with pytest.raises(ValueError):
        runtime.sample(2, 264, camera)
