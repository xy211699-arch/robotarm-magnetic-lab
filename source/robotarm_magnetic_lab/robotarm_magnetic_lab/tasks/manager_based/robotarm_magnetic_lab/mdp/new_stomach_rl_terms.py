"""Isolated new-task observations/reward; no inherited bring-up reward terms."""
import torch


def actor_observation(env):
    runtime = getattr(env, '_new_stomach_rl_runtime', None)
    if runtime is None or not runtime.ready:
        return torch.zeros((env.num_envs,548), device=env.device)
    return runtime.actor_observation()


def privileged_critic_observation(env):
    runtime = getattr(env, '_new_stomach_rl_runtime', None)
    if runtime is None or not runtime.ready:
        return torch.zeros((env.num_envs,28), device=env.device)
    return runtime.critic_observation()


def summed_four_term_reward(env):
    runtime = getattr(env, '_new_stomach_rl_runtime', None)
    if runtime is None or not runtime.ready:
        return torch.zeros(env.num_envs, device=env.device)
    return runtime.finish_second()
