"""Synchronous per-environment RGB and area-coverage runtime for TASK-009D0."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import torch

from robotarm_magnetic_lab.coverage.area_weights import target_vertex_area_weights
from robotarm_magnetic_lab.coverage.batched_accumulator import (
    BatchedCoverageAccumulator,
    BatchedCoverageUpdate,
)
from robotarm_magnetic_lab.coverage.batched_visibility import (
    BatchedWarpFirstHitRaycaster,
    batched_candidate_mask,
    build_incident_face_table,
    visible_from_batched_first_hits,
)
from robotarm_magnetic_lab.coverage.reference_mesh import ReferenceMesh


def _tensor(value) -> torch.Tensor:
    return getattr(value, "torch", value)


def translation_ulp_transform_candidates(matrix):
    """Yield deterministic ±1-ULP translation candidates, identity first."""
    import numpy as np

    source = np.asarray(matrix, dtype=np.float64)
    if source.shape != (4, 4) or not np.isfinite(source).all():
        raise ValueError("reference transform must be a finite 4x4 matrix")
    offsets = sorted(
        product((-1, 0, 1), repeat=3),
        key=lambda item: (sum(abs(value) for value in item), item),
    )
    for offset in offsets:
        candidate = source.copy()
        for axis, direction in enumerate(offset):
            if direction:
                candidate[3, axis] = np.nextafter(
                    candidate[3, axis],
                    np.inf if direction > 0 else -np.inf,
                )
        yield offset, candidate


class Task009D0RgbSynchronizer:
    """Associate exactly one fresh camera frame with each global boundary."""

    def __init__(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._last_frames = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        self._last_boundary: int | None = None
        self._latest = self._last_frames.clone()
        self.last_forced_capture = False

    @property
    def latest(self) -> torch.Tensor:
        return self._latest.clone()

    def observe(self, boundary: int, camera) -> torch.Tensor:
        boundary = int(boundary)
        if self._last_boundary is not None and boundary < self._last_boundary:
            raise RuntimeError("RGB control boundary decreased")
        if boundary == self._last_boundary:
            return self._latest.clone()
        frames = _tensor(camera.frame).to(device=self.device, dtype=torch.int64).reshape(-1)
        if frames.shape != self._last_frames.shape:
            raise ValueError("camera frame vector does not match num_envs")
        initialized = self._last_frames >= 0
        if initialized.any():
            advanced = frames > self._last_frames
            relevant_advanced = advanced[initialized]
            if relevant_advanced.any() and not relevant_advanced.all():
                raise RuntimeError("partial camera advancement across environments")
            forced = False
            if not relevant_advanced.any():
                camera._update_buffers_impl(camera._ALL_ENV_MASK)
                frames = _tensor(camera.frame).to(
                    device=self.device, dtype=torch.int64
                ).reshape(-1)
                forced = True
            expected = self._last_frames + 1
            if torch.any(frames[initialized] != expected[initialized]).item():
                raise RuntimeError(
                    "policy RGB must advance exactly once per initialized environment"
                )
            self.last_forced_capture = forced
        else:
            self.last_forced_capture = False
        self._last_frames.copy_(frames)
        self._latest.copy_(frames)
        self._last_boundary = boundary
        return frames.clone()

    def reset_rows(self, env_ids: torch.Tensor) -> None:
        rows = env_ids.to(device=self.device, dtype=torch.int64).reshape(-1)
        self._last_frames[rows] = -1
        self._latest[rows] = -1
        self._last_boundary = None


@dataclass(frozen=True)
class Task009D0CoverageBoundary:
    boundary: int
    frame_ids: torch.Tensor
    raw: BatchedCoverageUpdate
    reachable: BatchedCoverageUpdate
    new_coverage_reward_m2: torch.Tensor
    initial: bool
    stabilizing: bool
    candidate_counts: torch.Tensor
    ray_count: int


class Task009D0CoverageRuntime:
    """Own batched raw/reachable masks and exact camera-boundary visibility."""

    def __init__(
        self,
        *,
        reference_local: ReferenceMesh,
        env_origins: torch.Tensor,
        raw_vertex_weights: torch.Tensor,
        reachable_vertex_weights: torch.Tensor,
        device: str | torch.device,
        camera=None,
        visibility_override: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        raycaster: BatchedWarpFirstHitRaycaster | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.reference_local = reference_local
        self.vertices_local = torch.as_tensor(
            reference_local.vertices_world, device=self.device, dtype=torch.float64
        )
        self.env_origins = env_origins.to(device=self.device, dtype=torch.float64)
        if self.env_origins.ndim != 2 or self.env_origins.shape[1] != 3:
            raise ValueError("env_origins must have shape [E,3]")
        self.num_envs = self.env_origins.shape[0]
        self.raw_accumulator = BatchedCoverageAccumulator(
            raw_vertex_weights, self.num_envs, self.device
        )
        self.reachable_accumulator = BatchedCoverageAccumulator(
            reachable_vertex_weights, self.num_envs, self.device
        )
        self.reachable_vertex_mask = reachable_vertex_weights.to(
            device=self.device, dtype=torch.float64
        ).reshape(-1) > 0.0
        self.incident_face_table = build_incident_face_table(
            reference_local.incident_triangles, self.device
        )
        self.camera = camera
        self.rgb_sync = Task009D0RgbSynchronizer(self.num_envs, self.device)
        self.visibility_override = visibility_override
        self.raycaster = raycaster
        self.latest_raw_visible_mask = torch.zeros(
            (self.num_envs, len(self.vertices_local)), dtype=torch.bool, device=self.device
        )
        self.latest_reachable_visible_mask = torch.zeros_like(
            self.latest_raw_visible_mask
        )
        self.latest_update: Task009D0CoverageBoundary | None = None
        self._latest_boundary: int | None = None
        minimum = self.vertices_local.amin(dim=0)
        span = (self.vertices_local.amax(dim=0) - minimum).clamp_min(1.0e-12)
        coordinates = ((self.vertices_local - minimum) / span * 3.0).floor().to(torch.int64).clamp(0, 2)
        self._coverage_grid_cell = coordinates[:, 0] * 9 + coordinates[:, 1] * 3 + coordinates[:, 2]
        self._coverage_grid_total_area = torch.zeros(27, device=self.device, dtype=torch.float64)
        self._coverage_grid_total_area.scatter_add_(
            0, self._coverage_grid_cell, self.reachable_accumulator.weights
        )

    @classmethod
    def from_environment(
        cls,
        env,
        *,
        unreachable_region_path: str | Path,
        surface_prim_path: str | None = None,
    ) -> "Task009D0CoverageRuntime":
        from robotarm_magnetic_lab.coverage.simulator_runtime import (
            DEFAULT_INNER_SURFACE_PATH,
            reference_from_stage,
        )
        from robotarm_magnetic_lab.coverage.unreachable_region import load_unreachable_mask

        base = env.unwrapped if hasattr(env, "unwrapped") else env
        selected_prim_path = surface_prim_path or DEFAULT_INNER_SURFACE_PATH
        origins = _tensor(base.scene.env_origins).to(
            device=base.device, dtype=torch.float64
        )
        # Isaac Lab rewrites a spawned USD default-Prim transform, so obtain
        # geometry from the live stage and remove the complete env_0 clone
        # transform.  Using USD doubles avoids float32 env-origin residue in
        # the frozen geometry hash for 2-D 4/8-environment layouts.
        import json
        import numpy as np
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        env_zero = stage.GetPrimAtPath("/World/envs/env_0")
        if not env_zero.IsValid():
            raise RuntimeError("TASK-009D0 env_0 clone prim is unavailable")
        surface = stage.GetPrimAtPath(selected_prim_path)
        if not surface.IsValid():
            raise RuntimeError(f"approved stomach surface is unavailable: {selected_prim_path}")
        relative_matrix_gf, _ = UsdGeom.XformCache(0.0).ComputeRelativeTransform(
            surface, env_zero
        )
        relative_matrix = np.asarray(
            [
                [float(relative_matrix_gf[row][column]) for column in range(4)]
                for row in range(4)
            ],
            dtype=np.float64,
        )
        expected_hash = json.loads(Path(unreachable_region_path).read_text(encoding="utf-8"))[
            "stomach_geometry_sha256"
        ]
        local_reference = None
        matched_offset = None
        # Gf's one-environment world composition and clone-relative transform
        # can differ by one ULP after the environment grid translation is
        # introduced.  Search only translation neighbors and still require the
        # exact frozen geometry SHA-256; topology or asset changes cannot pass.
        for offset, candidate in translation_ulp_transform_candidates(relative_matrix):
            reference = reference_from_stage(
                selected_prim_path,
                world_transform_override=candidate,
            )
            if reference.geometry_sha256 == expected_hash:
                local_reference = reference
                matched_offset = offset
                break
        if local_reference is None:
            raise RuntimeError(
                "live stomach clone normalization mismatch: "
                f"expected={expected_hash}, relative={reference.geometry_sha256}, "
                f"relative_matrix={relative_matrix.tolist()}, "
                f"env_origin0={origins[0].detach().cpu().tolist()}"
            )
        if matched_offset != (0, 0, 0):
            print(
                "TASK009D0_REFERENCE_NORMALIZED",
                json.dumps(
                    {
                        "translation_ulp_offset": matched_offset,
                        "geometry_sha256": expected_hash,
                    }
                ),
            )
        raw_weights = target_vertex_area_weights(local_reference)
        _, unreachable = load_unreachable_mask(
            Path(unreachable_region_path), local_reference
        )
        reachable_weights = target_vertex_area_weights(
            local_reference, unreachable.reachable_triangle_indices
        )
        raycaster = BatchedWarpFirstHitRaycaster(local_reference, device=str(base.device))
        return cls(
            reference_local=local_reference,
            env_origins=origins,
            raw_vertex_weights=torch.as_tensor(raw_weights),
            reachable_vertex_weights=torch.as_tensor(reachable_weights),
            device=base.device,
            camera=base.scene["capsule_camera"],
            raycaster=raycaster,
        )

    def _camera_inputs(
        self,
        boundary: int,
        frame_ids: torch.Tensor | None,
        camera_centers_world: torch.Tensor | None,
        optical_axes_world: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if frame_ids is None:
            if self.camera is None:
                raise ValueError("frame_ids are required without a live camera")
            _ = self.camera.data.output["rgb"]
            frame_ids = self.rgb_sync.observe(boundary, self.camera)
        if camera_centers_world is None:
            camera_centers_world = _tensor(self.camera.data.pos_w)
        if optical_axes_world is None:
            from isaaclab.utils import math as math_utils

            quaternions = _tensor(self.camera.data.quat_w_ros).to(
                device=self.device, dtype=torch.float64
            )
            local_axis = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float64
            ).expand(self.num_envs, -1)
            optical_axes_world = math_utils.quat_apply(quaternions, local_axis)
        return (
            frame_ids.to(device=self.device, dtype=torch.int64).reshape(-1),
            camera_centers_world.to(device=self.device, dtype=torch.float64),
            optical_axes_world.to(device=self.device, dtype=torch.float64),
        )

    def _visibility(
        self, centers_world: torch.Tensor, axes_world: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        centers_local = centers_world - self.env_origins
        if self.visibility_override is not None:
            visible = self.visibility_override(centers_local, axes_world).to(
                device=self.device, dtype=torch.bool
            )
            if visible.shape != (self.num_envs, len(self.vertices_local)):
                raise ValueError("visibility override must return [E,V]")
            return visible, visible.sum(dim=1), int(visible.numel())
        if self.raycaster is None:
            raise RuntimeError("production visibility requires a batched raycaster")
        candidate, target_distances = batched_candidate_mask(
            self.vertices_local, centers_local, axes_world
        )
        hit_distances, hit_faces = self.raycaster.query(
            centers_local, candidate, self.vertices_local
        )
        visible = visible_from_batched_first_hits(
            self.reference_local,
            centers_local,
            self.vertices_local,
            candidate,
            target_distances,
            hit_distances,
            hit_faces,
            self.incident_face_table,
            normal_sign=-1,
        )
        return visible, candidate.sum(dim=1), self.raycaster.last_ray_count

    def _snapshot(self, accumulator: BatchedCoverageAccumulator) -> BatchedCoverageUpdate:
        mask = accumulator.mask
        weights = accumulator.weights
        area = (mask.to(torch.float64) * weights[None, :]).sum(dim=1)
        zeros_i = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        zeros_f = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        return BatchedCoverageUpdate(
            updated=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            visible_count=zeros_i,
            newly_covered_count=zeros_i.clone(),
            visible_area_m2=zeros_f,
            newly_covered_area_m2=zeros_f.clone(),
            cumulative_area_m2=area,
            coverage_fraction=area / accumulator.total_area_m2,
        )

    def _update(
        self,
        *,
        boundary: int,
        frame_ids: torch.Tensor | None,
        camera_centers_world: torch.Tensor | None,
        optical_axes_world: torch.Tensor | None,
        initial: bool,
        stabilizing: bool,
    ) -> Task009D0CoverageBoundary:
        boundary = int(boundary)
        if self._latest_boundary is not None:
            if boundary == self._latest_boundary:
                return self.latest_update
            if boundary < self._latest_boundary:
                raise RuntimeError("coverage control boundary decreased")
        frames, centers, axes = self._camera_inputs(
            boundary, frame_ids, camera_centers_world, optical_axes_world
        )
        if stabilizing:
            raw_update = self._snapshot(self.raw_accumulator)
            reachable_update = self._snapshot(self.reachable_accumulator)
            candidate_counts = torch.zeros(
                self.num_envs, dtype=torch.int64, device=self.device
            )
            ray_count = 0
        else:
            visible, candidate_counts, ray_count = self._visibility(centers, axes)
            self.latest_raw_visible_mask.copy_(visible)
            self.latest_reachable_visible_mask.copy_(
                visible & self.reachable_vertex_mask[None, :]
            )
            raw_update = self.raw_accumulator.update(frames, visible)
            reachable_update = self.reachable_accumulator.update(
                frames, visible & self.reachable_vertex_mask[None, :]
            )
        reward = torch.zeros_like(reachable_update.newly_covered_area_m2)
        if not initial and not stabilizing:
            reward = reachable_update.newly_covered_area_m2.clone()
        result = Task009D0CoverageBoundary(
            boundary=boundary,
            frame_ids=frames.clone(),
            raw=raw_update,
            reachable=reachable_update,
            new_coverage_reward_m2=reward,
            initial=bool(initial),
            stabilizing=bool(stabilizing),
            candidate_counts=candidate_counts,
            ray_count=int(ray_count),
        )
        self._latest_boundary = boundary
        self.latest_update = result
        return result

    def capture_initial(
        self,
        *,
        boundary: int = 0,
        frame_ids: torch.Tensor | None = None,
        camera_centers_world: torch.Tensor | None = None,
        optical_axes_world: torch.Tensor | None = None,
    ) -> Task009D0CoverageBoundary:
        return self._update(
            boundary=boundary,
            frame_ids=frame_ids,
            camera_centers_world=camera_centers_world,
            optical_axes_world=optical_axes_world,
            initial=True,
            stabilizing=False,
        )

    def update_boundary(
        self,
        *,
        boundary: int,
        frame_ids: torch.Tensor | None = None,
        camera_centers_world: torch.Tensor | None = None,
        optical_axes_world: torch.Tensor | None = None,
        stabilizing: bool = False,
    ) -> Task009D0CoverageBoundary:
        return self._update(
            boundary=boundary,
            frame_ids=frame_ids,
            camera_centers_world=camera_centers_world,
            optical_axes_world=optical_axes_world,
            initial=False,
            stabilizing=stabilizing,
        )

    def reset_rows(self, env_ids: torch.Tensor) -> None:
        rows = env_ids.to(device=self.device, dtype=torch.int64).reshape(-1)
        self.raw_accumulator.reset_rows(rows)
        self.reachable_accumulator.reset_rows(rows)
        self.latest_raw_visible_mask[rows] = False
        self.latest_reachable_visible_mask[rows] = False
        self.rgb_sync.reset_rows(rows)
        self._latest_boundary = None
        self.latest_update = None

    def coverage_grid_3x3x3(self) -> torch.Tensor:
        """Return per-row area-weighted cumulative reachable coverage in 27 cells."""
        weighted = (
            self.reachable_accumulator.mask.to(torch.float64)
            * self.reachable_accumulator.weights.unsqueeze(0)
        )
        result = torch.zeros((self.num_envs, 27), device=self.device, dtype=torch.float64)
        result.scatter_add_(
            1,
            self._coverage_grid_cell.unsqueeze(0).expand(self.num_envs, -1),
            weighted,
        )
        denominator = self._coverage_grid_total_area.unsqueeze(0)
        return torch.where(denominator > 0.0, result / denominator.clamp_min(1.0e-30), torch.zeros_like(result))
