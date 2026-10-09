"""One authoritative full-camera capture per vector policy boundary.

SDK sensor periods use per-row floating timestamps which are re-based on a
partial reset. They must not decide the global policy clock. Keep the archived
synchronizer untouched; only this new task schedules camera acquisition.
"""
from types import SimpleNamespace
import torch


class VectorPolicyRGBBoundary:
    def __init__(self,num_envs,device):
        with torch.inference_mode(False):
            self._last_frames=torch.full((num_envs,),-1,dtype=torch.int64,device=device)
        self.last_second=-1
        self.sync=SimpleNamespace(last_forced_capture=False,reset_rows=self.reset_rows)

    def reset_rows(self,ids):
        # No capture or physical step on reset, no change to other frame IDs.
        self._last_frames[ids]=-1

    def sample(self,second,tick,camera):
        if tick!=second*240 or second!=self.last_second+1 or camera.cfg.update_period!=1.:
            raise ValueError('next actual one-second vector boundary required')
        before=getattr(camera.frame,'torch',camera.frame).to(torch.int64).clone()
        initialized=self._last_frames>=0
        if before.shape!=self._last_frames.shape or torch.any(before[initialized]!=self._last_frames[initialized]):
            raise RuntimeError('camera captured outside the vector policy boundary')
        if torch.any(before[~initialized]!=0):
            raise RuntimeError('reset camera must start at sensor frame zero')
        # Do not read Camera.data first: its lazy getter can advance only some
        # rows. Mark all rows due and use SDK's timestamp/flag bookkeeping once.
        # Protected SDK API: re-audit against sensor_base.py on SDK upgrades.
        import warp as wp
        if not hasattr(camera,'_update_outdated_buffers') or not hasattr(camera,'_is_outdated'):
            raise RuntimeError('camera SDK boundary API changed; explicit audit required')
        wp.to_torch(camera._is_outdated).fill_(True)
        camera._update_outdated_buffers(force_recompute=True)
        after=getattr(camera.frame,'torch',camera.frame).to(torch.int64).clone()
        if not torch.equal(after,before+1):
            raise RuntimeError('each camera row must capture exactly one new frame')
        rgb=getattr(camera.data.output['rgb'],'torch',camera.data.output['rgb'])
        final=getattr(camera.frame,'torch',camera.frame).to(torch.int64)
        if not torch.equal(final,after):
            raise RuntimeError('camera lazy getter caused a second policy capture')
        if not torch.isfinite(rgb).all(): raise RuntimeError('nonfinite vector policy RGB')
        self._last_frames.copy_(after);self.last_second=second
        self.sync.last_forced_capture=True
        return rgb,after
