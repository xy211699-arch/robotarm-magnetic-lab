"""Pose-only fresh mounted-camera transforms, without a sensor/RGB update.

PhysX/Fabric composes the optical mount in float32. Match Camera's two Warp
convention conversions exactly: a 1nm Python/renderer difference can change a
first-hit at a mesh vertex. No quantization or relaxed visibility gate is used.
The fresh transform is checked against the current physical capsule and mount;
a stale hierarchy raises instead of silently using the previous RGB pose.
"""
import numpy as np
import torch
import warp as wp

from robotarm_magnetic_lab.coverage.new_stomach_rl_coverage import optical_from_capsule
from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw


@wp.kernel
def _camera_world_quat(raw: wp.array(dtype=wp.quatf), out: wp.array(dtype=wp.quatf)):
    i = wp.tid()
    # Exactly Camera._camera_update_state_kernel (Isaac Lab 3 local checkout).
    out[i] = raw[i] * wp.quatf(-.5, .5, .5, .5)


class LiveMountedOptics:
    def __init__(self, camera, device, forward_physics=None):
        self.camera, self.device = camera, str(device)
        self.forward_physics = forward_physics
        self._world = wp.empty(camera.num_instances, dtype=wp.quatf, device=device)
        self._ros = wp.empty(camera.num_instances, dtype=wp.quatf, device=device)
        self.records = []

    def __call__(self, current_capsule_pose):
        from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp
        if self.forward_physics is not None:
            # PhysxManager.step does not push transforms to Fabric. Its public
            # forward() does ONLY kinematics/transform sync (no step/render).
            self.forward_physics()
        # Read the live hierarchy, NOT camera.data (which has the 1Hz image).
        hierarchy = getattr(self.camera._view, '_fabric_hierarchy', None)
        if hierarchy is not None:
            # Pose-only parent/child propagation; no render or sensor frame.
            hierarchy.update_world_xforms()
        pos, raw = self.camera._view.get_world_poses()
        wp.launch(_camera_world_quat, dim=self.camera.num_instances,
            inputs=[wp.from_torch(raw.torch.contiguous(), dtype=wp.quatf), self._world], device=self.device)
        convert_camera_frame_orientation_convention_wp(self._world, self._ros, 'world', 'ros')
        quats = wp.to_torch(self._ros).cpu().numpy()
        axes = torch.tensor(np.stack([rotation_matrix_from_xyzw(q)[:, 2] for q in quats]),
            dtype=torch.float64, device=self.device)
        centers = pos.torch.to(torch.float64).clone()
        expected_center, expected_axis = optical_from_capsule(current_capsule_pose.to(torch.float64),
            self.camera.cfg.offset.pos, self.camera.cfg.offset.rot)
        position_error = torch.linalg.vector_norm(centers - expected_center, dim=1)
        axis_error = torch.linalg.vector_norm(axes - expected_axis, dim=1)
        self.records.append({'capsule_pose_world_xyzw': current_capsule_pose.detach().cpu().tolist(),
            'optical_center_world_m': centers.cpu().tolist(), 'optical_axis_world': axes.cpu().tolist(),
            'mount_position_error_m': position_error.cpu().tolist(), 'mount_axis_error': axis_error.cpu().tolist()})
        if (position_error > 1e-6).any() or (axis_error > 1e-5).any():
            raise RuntimeError('live mounted optical transform is stale or mount differs from physical capsule: '
                + repr(self.records[-1]))
        return centers, axes
