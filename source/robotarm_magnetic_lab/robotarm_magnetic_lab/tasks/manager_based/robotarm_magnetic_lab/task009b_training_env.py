"""Formal TASK-009B environment semantics for training and acceptance tests."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from .controllers.parameterized_force import ParameterizedForceMode


RESET_HOLD_CYCLES = 10


def _stable_rgb_digest(rgb: torch.Tensor) -> str:
    """Return a deterministic lightweight content digest without copying a full frame."""
    flat = rgb.detach().reshape(-1)
    if flat.numel() == 0:
        raise RuntimeError("policy RGB tensor is empty")
    count = min(1024, flat.numel())
    indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
    sample = flat.index_select(0, indices).float().cpu().numpy()
    metadata = np.asarray(
        [float(flat.min().item()), float(flat.max().item()), float(flat.mean().item())],
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(rgb.shape)).encode("ascii"))
    digest.update(str(rgb.dtype).encode("ascii"))
    digest.update(sample.tobytes())
    digest.update(metadata.tobytes())
    return digest.hexdigest()


class Task009BTrainingEnv(ManagerBasedRLEnv):
    """Add the contracted one-second HOLD stabilization to every explicit reset.

    The ten 10 Hz HOLD cycles advance physics and acquire ten fresh strategy
    images, but their episode-length cost is cleared before the first Actor
    action.  ``common_step_counter`` remains monotonic for frame auditing.
    """

    def reset(self, seed=None, env_ids=None, options=None):
        # Camera.reset() restarts its episode-local frame counter.  Discard the
        # prior boundary association before the base reset computes its first
        # observation.
        if hasattr(self, "_task009b_policy_rgb_sync"):
            del self._task009b_policy_rgb_sync
        if hasattr(self, "_task009b_policy_rgb_sync_latest"):
            del self._task009b_policy_rgb_sync_latest
        observation, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        hold = torch.full(
            (self.num_envs, 2),
            0.5,
            device=self.device,
            dtype=torch.float32,
        )
        hold[:, 0] = float(ParameterizedForceMode.HOLD)
        trace = []
        for cycle in range(RESET_HOLD_CYCLES):
            start_time = float(self.common_step_counter) * float(self.step_dt)
            observation, _, terminated, truncated, extras = super().step(hold)
            if bool(torch.any(terminated).item()) or bool(torch.any(truncated).item()):
                raise RuntimeError("formal reset HOLD stabilization unexpectedly terminated")
            rgb = observation["policy"]["rgb"]
            sync = dict(getattr(self, "_task009b_policy_rgb_sync_latest", {}))
            trace.append(
                {
                    "cycle": cycle,
                    "physics_substeps": int(self.cfg.decimation),
                    "start_sim_time_s": start_time,
                    "end_sim_time_s": float(self.common_step_counter) * float(self.step_dt),
                    "actor_rgb_frame": int(sync.get("frame", -1)),
                    "rgb_content_sha256": _stable_rgb_digest(rgb),
                    "rgb_finite": bool(torch.isfinite(rgb).all().item()),
                    "forced_capture": bool(sync.get("forced_capture", False)),
                }
            )
        self.episode_length_buf.zero_()
        extras["task009b_reset_stabilization"] = trace
        self._task009b_last_reset_stabilization = trace
        return observation, extras
