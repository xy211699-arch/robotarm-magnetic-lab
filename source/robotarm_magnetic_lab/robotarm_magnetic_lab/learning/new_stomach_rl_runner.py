"""Finite-update recurrent PPO preflight, not a formal training supervisor.

Smoke constants are provisional: gamma=.999**10, lambda=.95, clip=.2,
Adam 3e-4, value coefficient .5, one full recurrent minibatch / two epochs.
Entropy and KL diagnostics are per action dimension; PPO density is the joint
9/36-dimensional density. No reward/observation normalization is introduced.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .new_stomach_rl_actor import ContinuousGRUActor
from .new_stomach_rl_distribution import BoundedNormal


@dataclass
class Rollout:
    observations: torch.Tensor
    critic_inputs: torch.Tensor
    actions: torch.Tensor
    latents: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    initial_hidden: torch.Tensor
    old_means: torch.Tensor
    final_critic_inputs: torch.Tensor
    old_log_std: torch.Tensor | None = None


class SmokePPO:
    gamma, gae_lambda = .999**10, .95

    def __init__(self, dimension, device):
        self.dimension, self.device, self.update_count = dimension, str(device), 0
        self.actor = ContinuousGRUActor(dimension).to(device)
        self.critic = nn.Sequential(nn.Linear(28,256), nn.SiLU(), nn.Linear(256,256),
                                   nn.SiLU(), nn.Linear(256,1)).to(device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=3e-4)

    def update(self, data):
        tensors = [data.observations, data.critic_inputs, data.actions, data.latents,
                   data.old_log_probs, data.rewards, data.initial_hidden, data.old_means,
                   data.final_critic_inputs]
        if not all(torch.isfinite(t).all() for t in tensors):
            raise RuntimeError('nonfinite rollout')
        with torch.no_grad():
            values = self.critic(data.critic_inputs).squeeze(-1)
            last = self.critic(data.final_critic_inputs).squeeze(-1)
            advantages = torch.zeros_like(values)
            accumulator = torch.zeros_like(last)
            for index in reversed(range(len(values))):
                active = (~data.dones[index]).float()
                next_value = last if index == len(values)-1 else values[index+1]
                delta = data.rewards[index] + self.gamma * active * next_value - values[index]
                accumulator = delta + self.gamma*self.gae_lambda*active*accumulator
                advantages[index] = accumulator
            returns = values+advantages
            advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)
        reset = torch.cat((torch.zeros_like(data.dones[:1]),data.dones[:-1]))
        old_std = self.actor.log_std.detach().clamp(-5.,2.).clone() if data.old_log_std is None else data.old_log_std
        old = BoundedNormal(data.old_means, old_std.expand_as(data.old_means))
        parameters = list(self.actor.parameters())+list(self.critic.parameters())
        for epoch in range(2):
            dist,_ = self.actor.distribution(data.observations, data.initial_hidden.detach(), reset)
            logp = dist.log_prob(data.actions, data.latents)
            ratio = (logp-data.old_log_probs).exp()
            policy = -torch.minimum(ratio*advantages, ratio.clamp(.8,1.2)*advantages).mean()
            value_loss = (self.critic(data.critic_inputs).squeeze(-1)-returns).square().mean()
            loss = policy+.5*value_loss
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite PPO objective')
            self.optimizer.zero_grad(); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError('nonfinite PPO gradient')
            grad = torch.nn.utils.clip_grad_norm_(parameters,1.)
            self.optimizer.step()
        self.update_count += 1
        with torch.no_grad():
            new,_ = self.actor.distribution(data.observations, data.initial_hidden.detach(), reset)
            metric = dict(policy_loss=float(policy),value_loss=float(value_loss),gradient_norm=float(grad),
                kl_per_dimension=float(old.kl_per_dimension(new).mean()),
                entropy_per_dimension=float(new.entropy_mc().mean()/self.dimension),
                reward_mean=float(data.rewards.mean()))
        if not all(torch.isfinite(torch.tensor(v)) for v in metric.values()):
            raise RuntimeError('nonfinite PPO metrics')
        return metric

    def save(self, path):
        torch.save(dict(dimension=self.dimension,gamma=self.gamma,gae_lambda=self.gae_lambda,
            actor=self.actor.state_dict(),critic=self.critic.state_dict(),optimizer=self.optimizer.state_dict(),
            update_count=self.update_count,rng=torch.get_rng_state()), path)

    def load(self, path):
        data = torch.load(path,map_location=self.device,weights_only=True)
        if data['dimension'] != self.dimension or data['gamma'] != self.gamma or data['gae_lambda'] != self.gae_lambda:
            raise ValueError('checkpoint dimension/timebase mismatch')
        self.actor.load_state_dict(data['actor']); self.critic.load_state_dict(data['critic'])
        self.optimizer.load_state_dict(data['optimizer']); self.update_count = data['update_count']
        torch.set_rng_state(data['rng'].cpu())
