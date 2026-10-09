"""Shared-clock WARMUP/ACTIVE lifecycle; this module never advances physics."""
import torch
from .new_stomach_rl_vector_rows import row_ids


class VectorLifecycle:
    def __init__(self, count, device):
        if int(count) != count or count < 1:
            raise ValueError('positive environment count required')
        self.count, self.device = int(count), device
        with torch.inference_mode(False):
            self.active = torch.zeros(count, dtype=torch.bool, device=device)
            self.episode_seconds = torch.zeros(count, dtype=torch.int64, device=device)
            self.episode_id = torch.zeros_like(self.episode_seconds)
        self.global_tick = 0
        self._in_batch = False

    def reset_rows(self, ids):
        if self._in_batch:
            raise RuntimeError('reset is only allowed at a shared boundary')
        rows = row_ids(ids, self.count)
        with torch.inference_mode(False), torch.no_grad():
            self.active[rows] = False
            self.episode_seconds[rows] = 0
            self.episode_id[rows] += 1
        return rows

    def begin(self, actions):
        if self._in_batch:
            raise RuntimeError('previous boundary is unfinished')
        if actions.ndim != 2 or actions.shape[0] != self.count or actions.shape[1] not in (9,36):
            raise ValueError('expected [N,9] or [N,36] actions')
        if not torch.isfinite(actions).all():
            raise ValueError('nonfinite actions')
        self._valid = self.active.clone()
        issued = actions.clone()
        issued[~self._valid] = 0
        self._in_batch = True
        return issued

    def finish(self, physics_steps):
        if not self._in_batch or physics_steps != 240:
            raise RuntimeError('one boundary requires exactly 240 physics steps')
        with torch.inference_mode(False), torch.no_grad():
            valid = self._valid.clone()
            start = ~valid
            self.episode_seconds[valid] += 1
            self.active[:] = True
            self.global_tick += 240
        self._in_batch = False
        return dict(valid_transition=valid, episode_start=start,
            episode_seconds=self.episode_seconds.clone(), global_physics_tick=self.global_tick)


def prime_zero_compatible_reward(reward, coverage):
    """New adapter accepts C0=0 without changing legacy positive-C0 checks."""
    c0 = torch.as_tensor(coverage, dtype=torch.float64, device=reward.device).reshape(-1)
    if reward.num_envs != 1 or c0.numel() != 1 or not torch.isfinite(c0).all() or (c0 < 0).any() or (c0 > 1).any():
        raise ValueError('finite C0 in [0,1] required')
    if bool(c0[0] > 0):
        reward.reset_reward([0], c0)
        return
    # The original reset only primes these baselines, never update(C0).
    with torch.inference_mode(False), torch.no_grad():
        tracker = reward.trackers[0]
        tracker.reset()
        tracker.previous_coverage.zero_()
        tracker._initialized.fill_(True)
        reward._pending.zero_(); reward._counts.zero_()
        reward.last_second_terms.zero_()
        reward._previous.zero_(); reward._initialized.fill_(True)
