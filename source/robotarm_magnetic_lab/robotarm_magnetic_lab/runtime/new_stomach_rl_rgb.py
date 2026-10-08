"""1Hz policy-boundary RGB capture; reuse the validated frame synchronizer.

This runtime is intended for the new environment observation path, not only
the validator. The existing synchronizer force-captures only a missing frame
via Camera._update_buffers_impl; compatibility must be rechecked on SDK update.
"""
import torch
from .task009d0_coverage_runtime import Task009D0RgbSynchronizer


class PolicyRGBBoundary:
    def __init__(self, num_envs, device):
        with torch.inference_mode(False):
            self.sync = Task009D0RgbSynchronizer(num_envs, device)
        self.last_second = -1

    def sample(self, policy_second, physics_tick, camera):
        second = int(policy_second)
        if physics_tick != second*240 or second != self.last_second+1 or camera.cfg.update_period != 1.:
            raise ValueError('RGB acquisition requires the next actual 1Hz physical boundary')
        rgb = camera.data.output['rgb']
        with torch.inference_mode(False):
            frames = self.sync.observe(second, camera)
        rgb = getattr(camera.data.output['rgb'], 'torch', camera.data.output['rgb'])
        if not torch.isfinite(rgb).all():
            raise RuntimeError('nonfinite policy RGB')
        self.last_second = second
        return rgb, frames
