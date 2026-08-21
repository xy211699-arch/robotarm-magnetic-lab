import numpy as np
import pytest

from robotarm_magnetic_lab.runtime.dynamic_force_macro_runner import SynchronousMacroRunner


class FakeTerm:
    def __init__(self):
        self.lifecycle = "idle"
        self.substeps = 0
        self.trace_digest = "digest"
        self.events = []

    def process(self):
        if self.lifecycle == "idle": self.lifecycle = "running"
    def physics(self):
        self.substeps += 1
        if self.substeps == 240: self.lifecycle = "boundary_ready"
    def release_after_boundary_capture(self):
        self.events.append("release_active_wrench")
        self.lifecycle = "idle"


class FakeEnv:
    device = "cpu"
    camera_frame_id = 0
    common_step_counter = 0
    def __init__(self):
        self.unwrapped = self
        self.dynamic_force_macro = FakeTerm()
        self.events = self.dynamic_force_macro.events
    def step(self, action):
        self.dynamic_force_macro.process()
        for _ in range(4): self.dynamic_force_macro.physics()
        self.common_step_counter += 1
        if self.common_step_counter % 2 == 0: self.camera_frame_id += 1
    def capture_boundary_rgb(self):
        self.events.append("capture_boundary_rgb")
        return np.zeros((2, 2, 3), dtype=np.uint8)


def test_one_macro_has_exactly_240_physics_substeps():
    env = FakeEnv()
    transition = SynchronousMacroRunner(env).step(0)
    assert env.dynamic_force_macro.substeps == 240
    assert transition.simulated_duration_s == pytest.approx(1.0)


def test_up_captures_before_force_release():
    env = FakeEnv()
    SynchronousMacroRunner(env).step(5)
    assert env.events[-2:] == ["capture_boundary_rgb", "release_active_wrench"]
