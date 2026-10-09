"""Vector observations and the unchanged arm/world safety decision per row."""
import numpy as np
import torch


def actor_observation(env):
    if getattr(env,'runtime',None) is None:
        return torch.zeros((env.num_envs,548),device=env.device)
    return env.runtime.observations()['policy']


def privileged_critic_observation(env):
    if getattr(env,'runtime',None) is None:
        return torch.zeros((env.num_envs,28),device=env.device)
    return env.runtime.observations()['critic']


def audited_vector_collision(env):
    from ..controllers.batched_arm_clearance import minimum_clearance_fast,minimum_world_clearance
    results = []
    for term in env.action_manager.get_term('magnet').rows:
        term._geometry()
        q = term.robot.data.joint_pos.torch[0,term.ids[:6]].detach().cpu().numpy()[None]
        own = float(minimum_clearance_fast(term.kinematics,q)[0])
        wall = float(minimum_world_clearance(term.world_checker,q)[0])
        results.append(not np.isfinite([own,wall]).all() or min(own,wall)<term.cfg.required_clearance_m)
    return torch.tensor(results,device=env.device)
