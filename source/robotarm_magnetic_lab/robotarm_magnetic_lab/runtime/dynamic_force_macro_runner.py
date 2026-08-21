"""Synchronous 1 Hz boundary runner for TASK-008."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MacroTransition:
    action_id: int
    start_rgb_frame_id: int
    boundary_rgb_frame_id: int
    simulated_start_s: float
    simulated_end_s: float
    trace_digest: str
    boundary_rgb: Any
    catastrophic_fault: bool

    @property
    def simulated_duration_s(self) -> float:
        return self.simulated_end_s - self.simulated_start_s


class SynchronousMacroRunner:
    """Own exactly sixty 60 Hz environment steps per Actor transition."""

    def __init__(self, env, coverage_evaluator=None):
        self.env = env
        self.base = getattr(env, "unwrapped", env)
        self.coverage_evaluator = coverage_evaluator
        self.term = (
            self.base.action_manager.get_term("dynamic_force_macro")
            if hasattr(self.base, "action_manager")
            else self.base.dynamic_force_macro
        )

    def _frame_id(self) -> int:
        if hasattr(self.base, "scene"):
            return int(self.base.scene["capsule_camera"].frame.torch[0].item())
        return int(getattr(self.base, "camera_frame_id", 0))

    def _capture(self):
        if hasattr(self.base, "capture_boundary_rgb"):
            return self.base.capture_boundary_rgb()
        image = self.base.scene["capsule_camera"].data.output["rgb"]
        tensor = getattr(image, "torch", image)
        return tensor[0].clone()

    def step(self, action_id) -> MacroTransition:
        import torch
        import numpy as np

        action = int(action_id)
        start_frame = self._frame_id()
        start_time = float(getattr(self.base, "sim_time", getattr(self.base, "common_step_counter", 0) / 60.0))
        interrupted = False
        for _ in range(60):
            value = torch.tensor([[float(action)]], device=getattr(self.base, "device", "cpu"))
            result = self.env.step(value)
            if isinstance(result, tuple) and len(result) >= 5:
                terminated, truncated = result[2], result[3]
                interrupted |= bool(torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any())
            if self.coverage_evaluator is not None:
                self.coverage_evaluator.maybe_update()
        if self.term.lifecycle != "boundary_ready":
            raise RuntimeError(f"macro ended in unexpected lifecycle {self.term.lifecycle}")
        boundary_frame = self._frame_id()
        boundary_rgb = self._capture()
        finite_trace = all(
            np.isfinite(np.asarray(item.com_world)).all()
            and np.isfinite(np.asarray(item.applied_force_world)).all()
            and np.isfinite(np.asarray(item.applied_torque_world)).all()
            for item in getattr(self.term, "trace", ())
        )
        self.term.release_after_boundary_capture()
        end_time = float(
            getattr(self.base, "sim_time", getattr(self.base, "common_step_counter", 60) / 60.0)
        )
        return MacroTransition(
            action, start_frame, boundary_frame, start_time, end_time,
            self.term.trace_digest, boundary_rgb, interrupted or not finite_trace,
        )
