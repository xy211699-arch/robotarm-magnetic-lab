"""Isaac Lab bridge for the privileged P0 coverage evaluator."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .accumulator import CoverageAccumulator, CoverageUpdate
from .area_weights import target_vertex_area_weights, weights_sha256
from .records import CoverageRecordWriter, artifact_inventory
from .reference_mesh import MeshInput, preprocess_reference_mesh
from .runtime import RecordedFrameClock, assert_coverage_consistency
from .visibility import (
    FOV_HALF_ANGLE_DEG,
    HIT_DISTANCE_TOLERANCE_M,
    MAX_OBSERVATION_DISTANCE_M,
    WarpFirstHitRaycaster,
    candidate_vertices,
    visible_from_first_hits,
    camera_facing_first_hits,
)
from robotarm_magnetic_lab.ui.coverage_view import (
    KitCoveragePointCloudView,
    export_coverage_projection,
)


DEFAULT_INNER_SURFACE_PATH = (
    "/World/envs/env_0/Stomach/ConvertedSource/Environment/Stomach/VisualMesh/Stomach"
)


def _quat_xyzw_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 0.0:
        raise ValueError("camera quaternion is zero")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_values(matrix: Any) -> np.ndarray:
    return np.asarray(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def reference_from_stage(prim_path: str = DEFAULT_INNER_SURFACE_PATH):
    """Read only the preflight-approved luminal mesh from the live stage."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"approved inner surface is unavailable: {prim_path}")
    mesh = UsdGeom.Mesh(prim)
    mesh_input = MeshInput(
        prim_path=prim_path,
        vertices=np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64),
        face_vertex_counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64),
        face_vertex_indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64),
        world_transform=_matrix_values(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)),
        orientation=str(mesh.GetOrientationAttr().Get() or "rightHanded"),
    )
    return preprocess_reference_mesh([mesh_input], [prim_path])


class P0CoverageRuntime:
    """Own records, GPU visibility, cumulative state, and debug-only views."""

    def __init__(
        self,
        env: Any,
        output_directory: Path,
        *,
        task_id: str,
        seed: int,
        commit: str,
        branch: str,
        enable_view: bool = False,
        require_camera_facing_normal: bool = False,
        camera_facing_normal_sign: int = 1,
        raycast_device: str | None = None,
        surface_prim_path: str = DEFAULT_INNER_SURFACE_PATH,
    ) -> None:
        self.env = env.unwrapped
        self.camera = self.env.scene["capsule_camera"]
        self.capsule = self.env.scene["capsule"]
        self.reference = reference_from_stage(surface_prim_path)
        self.require_camera_facing_normal = bool(require_camera_facing_normal)
        self.camera_facing_normal_sign = int(camera_facing_normal_sign)
        if self.camera_facing_normal_sign not in (-1, 1):
            raise ValueError("camera_facing_normal_sign must be -1 or +1")
        self.raycast_device = str(raycast_device or self.env.device)
        self.raycaster = WarpFirstHitRaycaster(self.reference, device=self.raycast_device)
        self.vertex_weights = target_vertex_area_weights(self.reference)
        self.accumulator = CoverageAccumulator(
            len(self.reference.vertices_world), vertex_weights=self.vertex_weights
        )
        self.clock = RecordedFrameClock(float(self.camera.cfg.update_period))
        self.trajectory: list[np.ndarray] = []
        self.timings_s: list[float] = []
        self.candidate_counts: list[int] = []
        self.ray_counts: list[int] = []
        self.latest_record: dict[str, Any] | None = None
        self.snapshot_index = 0
        self.view = KitCoveragePointCloudView(self.reference.vertices_world) if enable_view else None
        metadata = {
            "task_id": task_id,
            "seed": int(seed),
            "repository_commit": commit,
            "repository_branch": branch,
            "selected_inner_surface_prims": list(self.reference.selected_prim_paths),
            "geometry_sha256": self.reference.geometry_sha256,
            "weld_tolerance_m": self.reference.weld_tolerance_m,
            "vertex_count": len(self.reference.vertices_world),
            "triangle_count": len(self.reference.triangles),
            "target_vertex_count": int(np.count_nonzero(self.vertex_weights > 0.0)),
            "target_triangle_count": len(self.reference.triangles),
            "target_total_area_m2": self.accumulator.total_area_m2,
            "target_vertex_weights_sha256": weights_sha256(self.vertex_weights),
            "camera": {
                "prim_path": self.camera.cfg.prim_path,
                "width": int(self.camera.cfg.width),
                "height": int(self.camera.cfg.height),
                "update_period_s": float(self.camera.cfg.update_period),
                "fov_full_angle_deg": 2.0 * FOV_HALF_ANGLE_DEG,
                "optical_convention": "ROS; +Z optical axis",
            },
            "coverage": {
                "max_distance_m": MAX_OBSERVATION_DISTANCE_M,
                "half_angle_deg": FOV_HALF_ANGLE_DEG,
                "hit_distance_tolerance_m": HIT_DISTANCE_TOLERANCE_M,
                "ray_backend": "isaaclab.utils.warp.ops.raycast_mesh CUDA first hit",
                "raycast_device": self.raycast_device,
                "require_camera_facing_normal": self.require_camera_facing_normal,
                "camera_facing_normal_sign": self.camera_facing_normal_sign,
                "synchronization": (
                    "Each timing ends after CUDA hit distances and face IDs are copied to CPU; "
                    "the copy synchronizes the query before the elapsed time is recorded."
                ),
            },
            "clocks_hz": {"physics": 240.0, "atomic": 20.0, "recorded": 1.0, "display": 30.0},
            "information_isolation": (
                "Capsule truth, rays and coverage are consumed only by evaluator/telemetry/view code."
            ),
        }
        self.writer = CoverageRecordWriter(Path(output_directory), metadata)

    @property
    def partial_directory(self) -> Path:
        return self.writer.partial_directory

    @property
    def sim_time_s(self) -> float:
        return float(self.env.episode_length_buf[0].item()) * float(self.env.step_dt)

    @property
    def total_sim_time_s(self) -> float:
        """Monotonic simulation time across any manager-driven episode reset."""
        return float(self.env.common_step_counter) * float(self.env.step_dt)

    def capsule_position(self) -> np.ndarray:
        return self.capsule.data.root_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)

    def append_action_event(self, request: Any, event: str, **extra: Any) -> None:
        record = {
            "request_id": int(request.request_id),
            "event": str(event),
            "timestamp_s": float(request.timestamp_s),
            "outcome": request.outcome.value,
            "action_id": None if request.action_id is None else int(request.action_id),
            "device_result": request.device_result,
            **extra,
        }
        self.writer.append_action(record)

    def maybe_update(self) -> CoverageUpdate | None:
        # Reading renderer output forces an outdated 1 Hz camera buffer to complete.
        _ = self.camera.data.output["rgb"]
        frame_value = int(self.camera.frame.torch[0].item())
        timestamp = frame_value * float(self.camera.cfg.update_period)
        frame_id = self.clock.observe(timestamp)
        if frame_id is None:
            return None
        started = time.perf_counter()
        center = self.camera.data.pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        quaternion = self.camera.data.quat_w_ros.torch[0].detach().cpu().numpy().astype(np.float64)
        optical_axis = _quat_xyzw_matrix(quaternion) @ np.asarray([0.0, 0.0, 1.0])
        candidates, target_distances = candidate_vertices(
            self.reference.vertices_world, center, optical_axis
        )
        hit_distances, hit_faces = self.raycaster.query(
            center, self.reference.vertices_world[candidates]
        )
        first_hit_visible = visible_from_first_hits(
            candidates,
            target_distances,
            hit_distances,
            hit_faces,
            self.reference.incident_triangles,
        )
        visible = first_hit_visible.copy()
        normal_facing_count = None
        if self.require_camera_facing_normal:
            normal_facing = camera_facing_first_hits(
                center,
                self.reference.vertices_world[candidates],
                hit_faces,
                self.reference,
                normal_sign=self.camera_facing_normal_sign,
            )
            visible &= normal_facing
            normal_facing_count = int(np.count_nonzero(normal_facing))
        update = self.accumulator.update(frame_id, candidates[visible])
        elapsed = time.perf_counter() - started
        capsule_position = self.capsule_position()
        self.trajectory.append(capsule_position.copy())
        self.timings_s.append(elapsed)
        self.candidate_counts.append(int(len(candidates)))
        self.ray_counts.append(int(len(candidates)))
        record = {
            "frame_id": int(frame_id),
            "timestamp_s": float(timestamp),
            "camera_position_world_m": center.tolist(),
            "camera_quaternion_ros_xyzw": quaternion.tolist(),
            "capsule_position_world_m": capsule_position.tolist(),
            "candidate_count": int(len(candidates)),
            "ray_count": int(len(candidates)),
            "first_hit_visible_count": int(np.count_nonzero(first_hit_visible)),
            "normal_facing_count": normal_facing_count,
            "visible_count": int(update.visible_count),
            "newly_covered_count": int(update.newly_covered_count),
            "cumulative_count": int(update.cumulative_count),
            "vertex_count": len(self.reference.vertices_world),
            "visible_area_m2": float(update.visible_area_m2),
            "newly_covered_area_m2": float(update.newly_covered_area_m2),
            "cumulative_area_m2": float(update.cumulative_area_m2),
            "total_area_m2": float(update.total_area_m2),
            "coverage_fraction": float(update.coverage_fraction),
            "coverage_update_s": float(elapsed),
        }
        self.writer.append_frame(record)
        self.latest_record = record
        print(
            "P0_COVERAGE "
            f"frame={frame_id} covered={update.cumulative_count}/{len(self.reference.vertices_world)} "
            f"percent={100.0 * update.coverage_fraction:.3f} "
            f"new={update.newly_covered_count}",
            flush=True,
        )
        return update

    def update_view(self) -> None:
        if self.view is not None:
            self.view.update(
                self.accumulator.mask,
                self.capsule_position(),
                np.asarray(self.trajectory, dtype=np.float64).reshape(-1, 3),
            )

    def snapshot(self, reason: str) -> dict[str, Any]:
        self.snapshot_index += 1
        stem = f"snapshot_{self.snapshot_index:04d}_{reason}"
        mask = self.accumulator.mask
        capsule_position = self.capsule_position()
        trajectory = np.asarray(self.trajectory, dtype=np.float64).reshape(-1, 3)
        fraction = self.accumulator.coverage_fraction
        png_path = self.partial_directory / f"{stem}.png"
        projection = export_coverage_projection(
            png_path,
            self.reference.vertices_world,
            mask,
            capsule_position,
            trajectory,
            fraction,
            self.sim_time_s,
        )
        metadata = {
            "reason": reason,
            "timestamp_s": self.sim_time_s,
            "cumulative_count": int(mask.sum()),
            "vertex_count": int(len(mask)),
            "coverage_fraction": fraction,
            **projection,
        }
        metadata_path = self.partial_directory / f"{stem}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self.latest_record is not None:
            assert_coverage_consistency(
                mask, self.latest_record, projection, vertex_weights=self.vertex_weights
            )
        return metadata

    def reset(self) -> None:
        self.snapshot("reset")
        self.accumulator.reset()
        self.clock.reset()
        self.trajectory.clear()
        self.latest_record = None
        if self.view is not None:
            self.update_view()

    def finalize(self, reason: str = "exit") -> Path:
        self.snapshot(reason)
        np.save(self.partial_directory / "coverage_mask.npy", self.accumulator.mask)
        np.save(
            self.partial_directory / "trajectory_world_m.npy",
            np.asarray(self.trajectory, dtype=np.float64).reshape(-1, 3),
        )
        np.save(self.partial_directory / "coverage_timings_s.npy", np.asarray(self.timings_s))
        summary = {
            "reason": reason,
            "coverage_updates": len(self.timings_s),
            "coverage_fraction": self.accumulator.coverage_fraction,
            "timing_s": {
                "median": float(np.median(self.timings_s)) if self.timings_s else None,
                "p95": float(np.percentile(self.timings_s, 95)) if self.timings_s else None,
                "maximum": float(np.max(self.timings_s)) if self.timings_s else None,
            },
            "candidate_count": {
                "median": float(np.median(self.candidate_counts)) if self.candidate_counts else None,
                "maximum": max(self.candidate_counts) if self.candidate_counts else None,
            },
            "ray_count": {
                "median": float(np.median(self.ray_counts)) if self.ray_counts else None,
                "maximum": max(self.ray_counts) if self.ray_counts else None,
            },
            "all_updates_under_one_second": bool(self.timings_s and max(self.timings_s) < 1.0),
        }
        (self.partial_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = [path for path in self.partial_directory.rglob("*") if path.is_file()]
        inventory = artifact_inventory(files, root=self.partial_directory)
        (self.partial_directory / "artifact_inventory.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        final = self.writer.finalize()
        if self.view is not None:
            self.view.close()
            self.view = None
        return final
