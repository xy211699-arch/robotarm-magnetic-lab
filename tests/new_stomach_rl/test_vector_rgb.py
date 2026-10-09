from types import SimpleNamespace
import sys
import torch
import pytest
from robotarm_magnetic_lab.runtime.new_stomach_rl_vector_rgb import VectorPolicyRGBBoundary


class Camera:
    def __init__(self):
        self.cfg=SimpleNamespace(update_period=1.)
        self.frame=torch.zeros(2,dtype=torch.int64)
        self._is_outdated=torch.zeros(2,dtype=torch.bool)
        self.calls=0
        self._data=SimpleNamespace(output={'rgb':torch.zeros(2,2,2,3,dtype=torch.uint8)})
    def _update_outdated_buffers(self,force_recompute=False):
        assert force_recompute and self._is_outdated.all()
        self.frame+=1;self.calls+=1;self._is_outdated.zero_()
    @property
    def data(self):
        # If this is called before acquisition, the SDK can advance just row0.
        if self._is_outdated.any(): self.frame[0]+=1
        return self._data


def test_partial_reset_then_one_full_capture_without_advancing_other_clock(monkeypatch):
    monkeypatch.setitem(sys.modules,'warp',SimpleNamespace(to_torch=lambda x:x))
    camera=Camera();clock=VectorPolicyRGBBoundary(2,'cpu')
    assert clock.sample(0,0,camera)[1].tolist()==[1,1]
    assert clock.sample(1,240,camera)[1].tolist()==[2,2]
    camera.frame[0]=0;clock.reset_rows(torch.tensor([0]))
    assert clock._last_frames.tolist()==[-1,2] and camera.calls==2
    # Simulate the period mask divergence after per-row timestamp rebase.
    camera._is_outdated[0]=True
    assert clock.sample(2,480,camera)[1].tolist()==[1,3]
    assert camera.calls==3
    assert clock.sample(3,720,camera)[1].tolist()==[2,4]
    with pytest.raises(ValueError): clock.sample(3,720,camera)
    assert camera.calls==4


def test_outside_capture_missing_or_multiple_frames_are_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules,'warp',SimpleNamespace(to_torch=lambda x:x))
    camera=Camera();clock=VectorPolicyRGBBoundary(2,'cpu');clock.sample(0,0,camera)
    camera.frame[0]+=1
    with pytest.raises(RuntimeError,match='outside'): clock.sample(1,240,camera)
    camera=Camera();clock=VectorPolicyRGBBoundary(2,'cpu')
    camera._update_outdated_buffers=lambda **kw:None
    with pytest.raises(RuntimeError,match='exactly one'): clock.sample(0,0,camera)
    camera._update_outdated_buffers=lambda **kw:camera.frame.add_(2)
    with pytest.raises(RuntimeError,match='exactly one'): clock.sample(0,0,camera)
