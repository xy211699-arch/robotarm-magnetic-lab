"""Project-local recurrent PPO for the TASK-010 hybrid action policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from .task010_actor import Task010Actor
from .task010_critic import Task010Critic
from .task010_distribution import Task010ModeBetaDistribution


def compute_task010_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    sampler_interrupted: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != values.shape or rewards.shape != terminated.shape:
        raise ValueError("TASK-010 GAE tensors must share [T,N] shape")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for step in reversed(range(rewards.shape[0])):
        continuation = (~terminated[step]).to(dtype=rewards.dtype)
        if step == rewards.shape[0] - 1:
            next_value = last_values if sampler_interrupted else torch.zeros_like(last_values)
        else:
            next_value = values[step + 1]
        delta = rewards[step] + gamma * continuation * next_value - values[step]
        gae = delta + gamma * lam * continuation * gae
        advantages[step] = gae
    returns = advantages + values
    return returns, advantages


class Task010RolloutStorage:
    def __init__(self, rollout_steps: int, num_envs: int, device: str | torch.device) -> None:
        self.rollout_steps, self.num_envs = int(rollout_steps), int(num_envs)
        self.device = torch.device(device)
        shape = (self.rollout_steps, self.num_envs)
        self.actor_observation = torch.zeros((*shape, 519), device=self.device)
        self.critic_observation = torch.zeros((*shape, 65), device=self.device)
        self.action = torch.zeros((*shape, 2), device=self.device)
        self.reward = torch.zeros(shape, device=self.device)
        self.terminated = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.reset_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.value = torch.zeros(shape, device=self.device)
        self.log_prob = torch.zeros(shape, device=self.device)
        self.old_logits = torch.zeros((*shape, 6), device=self.device)
        self.old_concentration_raw = torch.zeros((*shape, 5, 2), device=self.device)
        self.hidden_state = torch.zeros((self.rollout_steps, 1, self.num_envs, 256), device=self.device)
        self.returns = torch.zeros(shape, device=self.device)
        self.advantages = torch.zeros(shape, device=self.device)
        self.step = 0

    def add(self, **transition) -> None:
        if self.step >= self.rollout_steps:
            raise RuntimeError("TASK-010 rollout storage is full")
        index = self.step
        for name in ("actor_observation", "critic_observation", "action", "reward", "terminated", "reset_mask", "value", "log_prob"):
            source = transition[name]
            if isinstance(source, torch.Tensor):
                source = source.detach()
            getattr(self, name)[index].copy_(source)
        parameters = transition["distribution_parameters"]
        self.old_logits[index].copy_(parameters["logits"])
        self.old_concentration_raw[index].copy_(parameters["concentration_raw"])
        hidden = transition["hidden_state"]
        if hidden.shape != (1, self.num_envs, 256):
            hidden = torch.zeros((1, self.num_envs, 256), device=self.device)
        self.hidden_state[index].copy_(hidden.detach())
        self.step += 1

    @property
    def full(self) -> bool:
        return self.step == self.rollout_steps

    def compute_returns(self, last_values: torch.Tensor, *, gamma: float, lam: float, sampler_interrupted: bool) -> None:
        if not self.full:
            raise RuntimeError("TASK-010 rollout must be full before GAE")
        self.returns, self.advantages = compute_task010_gae(
            self.reward, self.value, self.terminated, last_values,
            gamma=gamma, lam=lam, sampler_interrupted=sampler_interrupted,
        )

    def recurrent_batches(self, *, num_mini_batches: int, num_epochs: int) -> Iterator[dict[str, torch.Tensor]]:
        if self.num_envs % num_mini_batches:
            raise ValueError("TASK-010 num_envs must divide evenly into recurrent mini-batches")
        envs_per_batch = self.num_envs // num_mini_batches
        for _ in range(num_epochs):
            for batch_index in range(num_mini_batches):
                env_ids = torch.arange(
                    batch_index * envs_per_batch, (batch_index + 1) * envs_per_batch,
                    device=self.device, dtype=torch.int64,
                )
                yield {
                    "env_ids": env_ids,
                    "actor_observation": self.actor_observation[:, env_ids],
                    "critic_observation": self.critic_observation[:, env_ids],
                    "action": self.action[:, env_ids],
                    "reset_mask": self.reset_mask[:, env_ids],
                    "value": self.value[:, env_ids],
                    "log_prob": self.log_prob[:, env_ids],
                    "old_logits": self.old_logits[:, env_ids],
                    "old_concentration_raw": self.old_concentration_raw[:, env_ids],
                    "returns": self.returns[:, env_ids],
                    "advantages": self.advantages[:, env_ids],
                    "initial_hidden_state": self.hidden_state[0, :, env_ids],
                }


class Task010PPO:
    """Minimal adapter matching the audited RSL-RL act/update lifecycle."""

    def __init__(
        self,
        actor: Task010Actor,
        critic: Task010Critic,
        *,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        learning_rate: float = 3.0e-4,
        desired_kl: float = 0.01,
        max_grad_norm: float = 1.0,
        entropy_coef: float = 0.005,
    ) -> None:
        self.actor, self.critic = actor, critic
        self.num_learning_epochs, self.num_mini_batches = int(num_learning_epochs), int(num_mini_batches)
        self.clip_param, self.value_loss_coef = float(clip_param), float(value_loss_coef)
        self.desired_kl, self.max_grad_norm, self.entropy_coef = float(desired_kl), float(max_grad_norm), float(entropy_coef)
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=float(learning_rate)
        )

    def act(self, actor_observation: torch.Tensor, critic_observation: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        hidden = self.actor.get_hidden_state().detach().clone()
        action = self.actor(actor_observation, stochastic_output=True)
        return {
            "action": action,
            "value": self.critic(critic_observation).squeeze(1),
            "log_prob": self.actor.get_output_log_prob(action).squeeze(1),
            "distribution_parameters": self.actor.output_distribution_params(),
            "hidden_state": hidden,
        }

    def process_env_step(self, *args, **kwargs) -> None:
        """Compatibility hook; the project runner writes explicit transitions."""
        del args, kwargs

    def update(self, storage: Task010RolloutStorage) -> dict[str, float]:
        if not storage.full:
            raise RuntimeError("TASK-010 PPO update requires a complete rollout")
        valid_advantages = storage.advantages
        normalized_advantages = (valid_advantages - valid_advantages.mean()) / valid_advantages.std(unbiased=False).clamp_min(1.0e-8)
        storage.advantages.copy_(normalized_advantages)
        totals = {
            name: 0.0 for name in (
                "surrogate_loss", "value_loss", "categorical_entropy",
                "conditional_beta_entropy", "joint_entropy", "joint_kl",
                "clip_fraction", "gradient_norm",
            )
        }
        updates = 0
        for batch in storage.recurrent_batches(num_mini_batches=self.num_mini_batches, num_epochs=self.num_learning_epochs):
            logits, raw, _ = self.actor.evaluate_parameters(
                batch["actor_observation"], masks=batch["reset_mask"], hidden_state=batch["initial_hidden_state"]
            )
            shape = logits.shape
            new_distribution = Task010ModeBetaDistribution(logits.reshape(-1, 6), raw.reshape(-1, 5, 2))
            old_distribution = Task010ModeBetaDistribution(
                batch["old_logits"].reshape(-1, 6), batch["old_concentration_raw"].reshape(-1, 5, 2)
            )
            actions = batch["action"].reshape(-1, 2)
            new_log_prob = new_distribution.log_prob(actions).reshape(shape[0], shape[1])
            old_log_prob = batch["log_prob"]
            ratio = torch.exp(new_log_prob - old_log_prob)
            advantages = batch["advantages"]
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            surrogate_loss = -torch.minimum(unclipped, clipped).mean()
            values = self.critic(batch["critic_observation"]).squeeze(-1)
            value_delta = (values - batch["value"]).clamp(-self.clip_param, self.clip_param)
            clipped_values = batch["value"] + value_delta
            value_loss = torch.maximum((values - batch["returns"]).square(), (clipped_values - batch["returns"]).square()).mean()
            entropy = new_distribution.entropy().mean()
            categorical_entropy = new_distribution.categorical.entropy().mean()
            conditional_beta_entropy = (
                new_distribution.categorical.probs[:, 1:] * new_distribution.beta.entropy()
            ).sum(dim=1).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            if not torch.isfinite(loss).item():
                raise RuntimeError("TASK-010 PPO loss is non-finite")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = list(self.actor.parameters()) + list(self.critic.parameters())
            for parameter in parameters:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError("TASK-010 PPO gradient is non-finite")
            gradient_norm = nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            self.optimizer.step()
            if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
                raise RuntimeError("TASK-010 PPO parameter is non-finite")
            joint_kl = old_distribution.kl(new_distribution).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_param).to(torch.float32).mean()
            for name, value in (
                ("surrogate_loss", surrogate_loss), ("value_loss", value_loss),
                ("categorical_entropy", categorical_entropy),
                ("conditional_beta_entropy", conditional_beta_entropy),
                ("joint_entropy", entropy), ("joint_kl", joint_kl),
                ("clip_fraction", clip_fraction), ("gradient_norm", gradient_norm),
            ):
                totals[name] += float(value.detach().item())
            updates += 1
        diagnostics = {name: value / updates for name, value in totals.items()}
        returns_variance = storage.returns.var(unbiased=False)
        diagnostics["value_explained_variance"] = float(
            (1.0 - (storage.returns - storage.value).var(unbiased=False) / returns_variance.clamp_min(1.0e-12)).item()
        )
        if diagnostics["joint_kl"] > 2.0 * self.desired_kl:
            for group in self.optimizer.param_groups:
                group["lr"] = max(1.0e-5, group["lr"] / 1.5)
        elif 0.0 < diagnostics["joint_kl"] < 0.5 * self.desired_kl:
            for group in self.optimizer.param_groups:
                group["lr"] = min(1.0e-3, group["lr"] * 1.5)
        diagnostics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        return diagnostics
