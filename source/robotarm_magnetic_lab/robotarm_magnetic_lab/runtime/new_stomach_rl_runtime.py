"""Single-environment action/coverage/RGB/reward composition for the new task.

Actor sees only frozen visual features and committed instruction history.
Capsule truth is confined to geometric coverage, old reward and explicit Critic.
"""
import hashlib
import numpy as np
import torch

from robotarm_magnetic_lab.coverage.new_stomach_rl_coverage import frozen_visibility, DualRateCoverage
from .new_stomach_rl_optics import LiveMountedOptics
from .new_stomach_rl_rgb import PolicyRGBBoundary
from .new_stomach_rl_reward import NewStomachReward
from .quaternion_conventions import rotation_matrix_from_xyzw


class NewStomachRLRuntime:
    def __init__(self, env, mask_path):
        if env.num_envs != 1:
            raise ValueError('new runtime requires per-environment action/bridge isolation before vectorization')
        self.env, self.ready = env, False
        self.term = env.action_manager.get_term('magnet')
        self.term._geometry()
        self.camera, self.capsule = env.scene['capsule_camera'], env.scene['capsule']
        self.visibility = frozen_visibility(self.term._new_stomach_runtime.reference, mask_path, env.device)
        self.optics = LiveMountedOptics(self.camera, env.device, env.sim.physics_manager.forward)
        self.coverage = DualRateCoverage(self.visibility, 1, geometry_optics=self.optics)
        self.reward = NewStomachReward(1, env.device)
        self.policy_second = 0
        self.visual_features = torch.zeros((1,512), device=env.device)
        self.encoder = None
        self.ten_hz_records, self.one_hz_records = [], []
        self.term.geometry_tick_callback = self.on_physics_boundary
        env._new_stomach_rl_runtime = self

    def capsule_pose(self):
        return self.capsule.data.root_link_pose_w.torch

    def camera_optics(self):
        data = self.camera.data
        positions = getattr(data.pos_w, 'torch', data.pos_w).to(torch.float64)
        quats = getattr(data.quat_w_ros, 'torch', data.quat_w_ros).cpu().numpy()
        axes = torch.tensor(np.stack([rotation_matrix_from_xyzw(q)[:,2] for q in quats]),
            dtype=torch.float64, device=self.env.device)
        return positions, axes

    def initialize(self):
        self.coverage.reset([0]); self.optics.records.clear()
        self.policy_second = 0
        self.rgb_clock = PolicyRGBBoundary(1, self.env.device)
        self.rgb, self.rgb_frames = self.rgb_clock.sample(0, 0, self.camera)
        self.coverage.initialize(self.optics(self.capsule_pose()))
        if not torch.equal(self.visibility(self.camera_optics()), self.coverage._boundary_visible):
            raise RuntimeError('reset C0 optical/RGB mismatch')
        self.reward.reset_reward([0], self.coverage.c10_records[0]['coverage'])
        self.ten_hz_records = [dict(self.coverage.c10_records[0], **self.optics.records[0], initialization=True)]
        self.one_hz_records = [self.rgb_record(0, self.coverage.c1_records[0]['coverage'])]
        self.visual_features.zero_()
        if self.encoder is not None:
            self.encoder.reset()
        self._encode_visual()
        self.ready = True

    def _encode_visual(self):
        if self.env.cfg.group in ('B','D'):
            if self.encoder is None:
                raise RuntimeError('Vision group requires its frozen ResNet18 encoder')
            from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.vision import _sample_capsule_image
            image = _sample_capsule_image(self.rgb, self.camera.data.intrinsic_matrices.torch,
                120., .98, is_depth=False)
            self.visual_features = self.encoder(image, self.rgb_frames)
        else:
            self.visual_features.zero_()

    def rgb_record(self, second, c1):
        return {'policy_second': second, 'sim_time_s': float(second), 'rgb_frame': int(self.rgb_frames[0]),
            'rgb_sha256': hashlib.sha256(self.rgb.detach().cpu().numpy().tobytes()).hexdigest(),
            'coverage': list(c1), 'forced_capture': self.rgb_clock.sync.last_forced_capture}

    def on_physics_boundary(self, local_tick):
        if not self.ready:
            return
        tick = self.policy_second*240 + local_tick
        delta = self.coverage.update_geometry_at_physics_tick(tick, self.capsule_pose())
        if delta is not None:
            result = self.reward.update_reward_10hz(self.capsule_pose(), delta.coverage)
            row = dict(self.coverage.c10_records[-1], **self.optics.records[-1])
            row['reward_terms'] = [float(getattr(result, name)[0]) for name in (
                'coverage_reward', 'escape_reward', 'no_progress_reward', 'coverage_resumed_reward')]
            row['total_reward'] = float(result.total_reward[0])
            row['recovery_phase'] = result.phase_one_hot_4[0].cpu().tolist()
            self.ten_hz_records.append(row)

    def finish_second(self):
        self.on_physics_boundary(240)
        self.policy_second += 1
        self.rgb, self.rgb_frames = self.rgb_clock.sample(self.policy_second, self.policy_second*240, self.camera)
        c1 = self.coverage.observe_camera_frame(self.policy_second, self.policy_second, self.camera_optics())
        total = self.reward.finish_policy_second()
        self._encode_visual()
        self.one_hz_records.append(dict(self.rgb_record(self.policy_second, c1.coverage.cpu().tolist()),
            c10=self.coverage.c10_records[-1]['coverage'],
            reward_terms=self.reward.last_second_terms[0].cpu().tolist(), total_reward=float(total[0])))
        return total

    def actor_observation(self):
        return torch.cat((self.visual_features, self.term.executed_history.reshape(1,36)), dim=1)

    def critic_observation(self):
        pose = self.capsule_pose()
        velocity = self.capsule.data.root_com_vel_w.torch
        joints = self.term.robot.data.joint_pos.torch[:,self.term.ids]
        # last_step is a completed sample, not reset state. Reading it here
        # leaked the previous episode's phase into the privileged C0 Critic.
        phase = torch.nn.functional.one_hot(torch.cat([tracker.phase for tracker in self.reward.trackers]),4).to(pose.dtype)
        coverage = pose.new_tensor([[self.coverage.c10_records[-1]['coverage'][0],
            self.coverage.c1_records[-1]['coverage'][0]]])
        return torch.cat((pose, velocity, joints, phase, coverage), dim=1)
