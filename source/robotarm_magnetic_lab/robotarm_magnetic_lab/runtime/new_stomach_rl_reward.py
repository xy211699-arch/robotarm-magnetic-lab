"""Unchanged TASK-010 four-term 0.1s rewards, summed (not averaged) per 1s.

One old tracker per row avoids its synchronous-history-reset limitation without
changing any old thresholds/phase transitions. C0 primes previous coverage only:
it does not add a rewarded sample or shorten the five-second window.
"""
from dataclasses import fields
import torch
from .task010_recovery import Task010RecoveryTracker, RecoveryStep


class NewStomachReward:
    def __init__(self, num_envs, device):
        self.num_envs, self.device = int(num_envs), str(device)
        with torch.inference_mode(False):
            self.trackers = [Task010RecoveryTracker(1, device, .1) for _ in range(num_envs)]
            self._pending = torch.zeros((num_envs,4), device=device)
            self._counts = torch.zeros(num_envs, dtype=torch.int64, device=device)
            self._previous = torch.zeros(num_envs, dtype=torch.float64, device=device)
            self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.last_second_terms = self._pending.clone()
        self.last_step = None

    def reset_reward(self, env_ids, c0):
        rows = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
        c0 = torch.as_tensor(c0, device=self.device, dtype=torch.float64).reshape(-1)
        if c0.shape != rows.shape or not torch.isfinite(c0).all() or (c0 <= 0).any() or (c0 > 1).any():
            raise ValueError('finite positive reset C0 for each row required')
        with torch.inference_mode(False), torch.no_grad():
            for index, row in enumerate(rows.tolist()):
                tracker = self.trackers[row]
                tracker.reset()
                # Same baseline initialization as old TASK-010 training; do
                # not call update(C0), which would reward C0 and seed history.
                tracker.previous_coverage[:] = c0[index].float()
                tracker._initialized.fill_(True)
            self._pending[rows] = 0; self._counts[rows] = 0
            self.last_second_terms[rows] = 0
            self._previous[rows] = c0; self._initialized[rows] = True

    def update_reward_10hz(self, capsule_pose, c10):
        capsule_pose = torch.as_tensor(capsule_pose, device=self.device)
        coverage = torch.as_tensor(c10, device=self.device, dtype=torch.float64).reshape(-1)
        if capsule_pose.shape != (self.num_envs,7) or coverage.shape != (self.num_envs,):
            raise ValueError('reward pose/C10 shape mismatch')
        if not torch.isfinite(capsule_pose).all() or not torch.isfinite(coverage).all():
            raise RuntimeError('nonfinite reward inputs')
        if not self._initialized.all() or (self._counts >= 10).any():
            raise RuntimeError('reset/finish required before the next ten physical rewards')
        if (coverage < self._previous).any():
            raise RuntimeError('C10 decreased')
        if (torch.linalg.vector_norm(capsule_pose[:,3:], dim=1) <= 0).any():
            raise ValueError('nonzero xyzw capsule quaternion required')
        with torch.inference_mode(False), torch.no_grad():
            steps = [tracker.update(capsule_pose[i:i+1,:3], capsule_pose[i:i+1,3:], coverage[i:i+1], .1)
                for i, tracker in enumerate(self.trackers)]
            result = RecoveryStep(*(torch.cat([getattr(step, field.name) for step in steps], dim=0)
                for field in fields(RecoveryStep)))
            contributions = torch.stack([result.coverage_reward, result.escape_reward,
                result.no_progress_reward, result.coverage_resumed_reward], dim=1)
            if not torch.isfinite(contributions).all():
                raise RuntimeError('nonfinite reward contributions')
            self._pending += contributions
            self._counts += 1; self._previous.copy_(coverage)
            self.last_step = result
            return result

    def finish_policy_second(self):
        if not (self._counts == 10).all():
            raise RuntimeError('exactly ten 0.1s rewards required for a policy second')
        with torch.inference_mode(False), torch.no_grad():
            self.last_second_terms.copy_(self._pending)
            total = self._pending.sum(dim=1).clone()
            self._pending.zero_(); self._counts.zero_()
            return total
