"""Predictive robot/ASM-to-stomach collision checking for atomic actions.

The capsule is intentionally absent from this module: capsule-to-stomach
contact is part of normal locomotion.  The checker queries the dense XRDF
``world_collision`` spheres for the arm and mounted ASM against the exact
static stomach triangle mesh before and during action execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..controllers.action_layer.kinematics import UrdfXrdfSafetyModel


@wp.kernel
def _closest_mesh_distance_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    faces: wp.array(dtype=int),
):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_no_sign(mesh, point, 10.0)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distances[index] = wp.length(closest - point)
        faces[index] = query.face
    else:
        distances[index] = 1.0e6
        faces[index] = -1


def _matrix_values(matrix) -> np.ndarray:
    return np.asarray(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in counts.tolist():
        face = indices[cursor : cursor + count]
        cursor += count
        for offset in range(1, count - 1):
            triangles.append((int(face[0]), int(face[offset]), int(face[offset + 1])))
    return np.asarray(triangles, dtype=np.int32)


@dataclass(frozen=True)
class WorldClearance:
    clearance_m: float
    sample_index: int
    frame: str
    sphere_index: int
    face_index: int


class StomachMeshCollisionChecker:
    """Unsigned swept sphere-to-surface clearance on the live stomach mesh.

    Unsigned distance is deliberate.  The current stomach is one thin open
    surface, so inside/outside signs are not reliable.  Rejecting overlap from
    either side prevents a robot link from crossing the wall while leaving the
    independently simulated capsule unconstrained.
    """

    def __init__(
        self,
        kinematics: UrdfXrdfSafetyModel,
        *,
        mesh_prim_path: str,
        robot_base_position_world_m: np.ndarray,
        robot_base_rotation_world: np.ndarray,
        device: str,
        required_clearance_m: float,
        trajectory_samples: int,
    ) -> None:
        import omni.usd
        from pxr import UsdGeom

        self.kinematics = kinematics
        self.mesh_prim_path = str(mesh_prim_path)
        self.base_position = np.asarray(
            robot_base_position_world_m, dtype=np.float64
        ).reshape(3)
        self.base_rotation = np.asarray(
            robot_base_rotation_world, dtype=np.float64
        ).reshape(3, 3)
        self.device = str(device)
        self.required_clearance_m = float(required_clearance_m)
        self.trajectory_samples = max(int(trajectory_samples), 2)

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.mesh_prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(
                f"stomach collision mesh is unavailable: {self.mesh_prim_path}"
            )
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        triangles = _triangulate(counts, indices)
        transform = _matrix_values(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        )
        vertices_world = np.c_[vertices, np.ones(len(vertices))] @ transform
        vertices_world = vertices_world[:, :3]
        self.mesh = wp.Mesh(
            points=wp.array(
                vertices_world.astype(np.float32),
                dtype=wp.vec3,
                device=self.device,
            ),
            indices=wp.array(
                triangles.reshape(-1),
                dtype=int,
                device=self.device,
            ),
        )

    def _configuration_spheres(
        self, configuration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        sphere_sets = self.kinematics.environment_world_spheres(configuration)
        if not sphere_sets:
            raise RuntimeError("XRDF world-collision geometry contains no usable frames")
        centers_base = np.concatenate([values[0] for values in sphere_sets.values()])
        radii = np.concatenate([values[1] for values in sphere_sets.values()])
        centers_world = (
            self.base_position + centers_base @ self.base_rotation.T
        )
        frame_names: list[str] = []
        local_indices: list[int] = []
        for frame, (centers, _radii) in sphere_sets.items():
            frame_names.extend([frame] * len(centers))
            local_indices.extend(range(len(centers)))
        return (
            centers_world,
            radii,
            frame_names,
            np.asarray(local_indices, dtype=np.int64),
        )

    def _query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        distances = wp.empty(len(points), dtype=float, device=self.device)
        faces = wp.empty(len(points), dtype=int, device=self.device)
        wp.launch(
            _closest_mesh_distance_kernel,
            dim=len(points),
            inputs=[
                self.mesh.id,
                wp.array(points, dtype=wp.vec3, device=self.device),
                distances,
                faces,
            ],
            device=self.device,
        )
        return (
            distances.numpy().reshape(-1).astype(np.float64),
            faces.numpy().reshape(-1).astype(np.int64),
        )

    def check_configuration(self, arm_configuration_rad: np.ndarray) -> WorldClearance:
        centers, radii, frames, local_indices = self._configuration_spheres(
            arm_configuration_rad
        )
        distances, faces = self._query(centers)
        clearances = distances - radii
        index = int(np.argmin(clearances))
        return WorldClearance(
            clearance_m=float(clearances[index]),
            sample_index=0,
            frame=frames[index],
            sphere_index=int(local_indices[index]),
            face_index=int(faces[index]),
        )

    def validate_path(self, arm_targets_rad: np.ndarray) -> dict[str, object]:
        targets = np.asarray(arm_targets_rad, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != len(self.kinematics.joints):
            raise ValueError("arm_targets_rad has an invalid shape")
        # Atomic trajectories are direct smooth blends.  Resampling the full
        # curve gives a fixed minimum sweep density even if a future planner
        # emits fewer control waypoints.
        sample_phase = np.linspace(0.0, 1.0, self.trajectory_samples)
        source_phase = np.linspace(0.0, 1.0, len(targets))
        samples = np.column_stack(
            [np.interp(sample_phase, source_phase, targets[:, joint]) for joint in range(targets.shape[1])]
        )

        all_centers: list[np.ndarray] = []
        all_radii: list[np.ndarray] = []
        frames: list[str] | None = None
        local_indices: np.ndarray | None = None
        for configuration in samples:
            centers, radii, frame_names, indices = self._configuration_spheres(
                configuration
            )
            all_centers.append(centers)
            all_radii.append(radii)
            if frames is None:
                frames = frame_names
                local_indices = indices
        sphere_count = len(all_radii[0])
        distances, faces = self._query(np.concatenate(all_centers))
        clearances = distances - np.concatenate(all_radii)
        flat_index = int(np.argmin(clearances))
        sample_index = flat_index // sphere_count
        sphere_index = flat_index % sphere_count
        minimum = float(clearances[flat_index])
        assert frames is not None and local_indices is not None
        result = {
            "ok": minimum >= self.required_clearance_m,
            "kind": "CLEAR" if minimum >= self.required_clearance_m else "ENVIRONMENT_COLLISION",
            "sample_index": sample_index,
            "minimum_world_clearance_m": minimum,
            "required_world_clearance_m": self.required_clearance_m,
            "frame": frames[sphere_index],
            "sphere_index": int(local_indices[sphere_index]),
            "face_index": int(faces[flat_index]),
            "mesh_prim_path": self.mesh_prim_path,
        }
        return result
