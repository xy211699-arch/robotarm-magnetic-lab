from types import SimpleNamespace

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import (
    CapsuleState,
    LatchReason,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.eleven_action_latch import CapsuleLatchRuntime


class _Attr:
    def __init__(self, value=0):
        self.value = value

    def __bool__(self):
        return True

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value
        return True


class _Api:
    def __init__(self):
        self.pos = _Attr(0)
        self.rot = _Attr(0)
        self.kinematic = _Attr(False)

    def GetLockedPosAxisAttr(self): return self.pos
    def CreateLockedPosAxisAttr(self): return self.pos
    def GetLockedRotAxisAttr(self): return self.rot
    def CreateLockedRotAxisAttr(self): return self.rot
    def GetKinematicEnabledAttr(self): return self.kinematic
    def CreateKinematicEnabledAttr(self): return self.kinematic


class _Composer:
    def __init__(self): self.reset_count = 0
    def reset(self): self.reset_count += 1


class _RootView:
    def __init__(self):
        self.disabled = np.zeros((1, 1), dtype=np.uint8)
        self.wake_count = 0

    @staticmethod
    def _numpy(value):
        if hasattr(value, "numpy"):
            return value.numpy()
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def set_disable_simulations(self, data, indices):
        data_np = self._numpy(data).reshape(-1)
        indices_np = self._numpy(indices).reshape(-1)
        self.disabled[indices_np.astype(np.int64), 0] = data_np[: len(indices_np)]

    def get_disable_simulations(self):
        return self.disabled.copy()

    def wake_up(self, indices):
        self.wake_count += len(self._numpy(indices).reshape(-1))


class _Capsule:
    def __init__(self):
        self.pose = np.asarray([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
        self.velocity = np.ones(6)
        self.last_written_velocity = None
        self.permanent_wrench_composer = _Composer()
        self.root_view = _RootView()

    def state(self):
        return CapsuleState(self.pose[:3], self.pose[3:], self.velocity[:3], self.velocity[3:])

    def write_root_velocity_to_sim_index(self, root_velocity):
        value = np.asarray(root_velocity).reshape(6)
        self.last_written_velocity = value.copy()
        self.velocity = value.copy()


def test_dynamic_flags_lock_all_six_axes_and_zero_velocity():
    capsule, api = _Capsule(), _Api()
    runtime = CapsuleLatchRuntime.dynamic_lock_flags(capsule, api)
    result = runtime.lock_current(capsule.state(), LatchReason.INITIAL)
    assert api.pos.Get() == api.rot.Get() == 0b111
    np.testing.assert_allclose(capsule.last_written_velocity, 0.0)
    assert result.latched
    assert capsule.permanent_wrench_composer.reset_count == 1


def test_unlock_clears_flags_but_does_not_change_pose():
    capsule, api = _Capsule(), _Api()
    runtime = CapsuleLatchRuntime.dynamic_lock_flags(capsule, api)
    runtime.lock_current(capsule.state(), LatchReason.INITIAL)
    pose_before = capsule.pose.copy()
    result = runtime.unlock_zeroed(capsule.state())
    assert api.pos.Get() == api.rot.Get() == 0
    np.testing.assert_allclose(capsule.pose, pose_before)
    assert not result.latched


def test_kinematic_fallback_is_not_selected_silently():
    with pytest.raises(RuntimeError, match="fallback requires tracked profile selection"):
        CapsuleLatchRuntime.auto_fallback(_Capsule(), _Api())


def test_tensor_disable_simulation_lock_and_release_are_direct_and_woken():
    capsule, api = _Capsule(), _Api()
    runtime = CapsuleLatchRuntime.tensor_disable_simulation(capsule, api)
    locked = runtime.lock_current(capsule.state(), LatchReason.INITIAL)
    assert locked.latched
    assert locked.simulation_disabled
    assert capsule.root_view.disabled[0, 0] == 1
    released = runtime.unlock_zeroed(capsule.state())
    assert not released.latched
    assert not released.simulation_disabled
    assert capsule.root_view.disabled[0, 0] == 0
    assert capsule.root_view.wake_count == 1
    np.testing.assert_allclose(capsule.last_written_velocity, 0.0)
