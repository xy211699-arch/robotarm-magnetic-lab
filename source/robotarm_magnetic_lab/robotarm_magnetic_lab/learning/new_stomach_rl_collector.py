"""Real vector boundary collector, also testable without importing Isaac Lab."""
import torch
from dataclasses import fields
from .new_stomach_rl_distribution import BoundedNormal
from .new_stomach_rl_long_rollout import LongRollout,recurrent_step,replay_sequence


def join_rollouts(pieces):
    if not pieces:raise ValueError('empty rollout pieces')
    if any(not torch.equal(x.old_log_std,pieces[0].old_log_std) for x in pieces):
        raise ValueError('policy changed between rollout pieces')
    return LongRollout(**{f.name:getattr(pieces[0],f.name) if f.name in ('initial_hidden','old_log_std') else
        torch.cat([getattr(x,f.name) for x in pieces]) for f in fields(LongRollout)})


class VectorRolloutCollector:
    def __init__(self,env,runner,observations,on_boundary=lambda *a:None):
        self.env,self.runner,self.observations=env,runner,observations
        self.hidden=torch.zeros((1,env.num_envs,256),device=runner.device)
        self.on_boundary=on_boundary
        self.boundaries=self.samples=0
        self.timeouts=torch.zeros(env.num_envs,dtype=torch.int64)

    def fresh(self,observations):
        self.observations=observations;self.hidden.zero_()

    def collect(self,length):
        if length<1: raise ValueError('positive rollout length required')
        stored={name:[] for name in LongRollout.__dataclass_fields__ if name not in ('initial_hidden','old_log_std')}
        initial=self.hidden.clone();std=self.runner.actor.log_std.detach().clamp(-5.,2.).clone()
        for _ in range(length):
            env,runner=self.env,self.runner
            valid=env.lifecycle.active.clone()
            start=env.lifecycle.episode_seconds.eq(0)
            obs=self.observations
            with torch.no_grad():
                means,next_hidden=recurrent_step(runner.actor,obs['policy'],self.hidden,start,valid)
                distribution=BoundedNormal(means,std.expand_as(means))
                action,latent=distribution.sample()
                value=runner.critic(obs['critic']).squeeze(-1)
            following,reward,terminated,truncated,extras=env.step(action)
            if extras['physical_substeps']!=240 or not torch.equal(extras['valid_transition'],valid):
                raise RuntimeError('physical clock or collector valid mask mismatch')
            # This is the PRE-autoreset final observation, never following[critic].
            final=extras['final_obs']['critic'].detach().clone()
            with torch.no_grad(): bootstrap=runner.critic(final).squeeze(-1)
            history=extras['final_obs']['policy'][:,512:].reshape(env.num_envs,4,9).detach().clone()
            values=dict(observations=obs['policy'],critic_inputs=obs['critic'],actions=action,
                latents=latent,old_log_probs=distribution.log_prob(action,latent),rewards=reward,
                values=value,bootstrap_values=bootstrap,valid=valid,episode_start=start,
                terminated=terminated,truncated=truncated,final_critic_inputs=final,
                executed_history=history,old_means=means)
            for name,x in values.items(): stored[name].append(x.detach().clone())
            self.hidden=next_hidden.detach().clone()
            self.hidden[:,terminated|truncated]=0
            self.observations=following
            self.boundaries+=1;self.samples+=int(valid.sum())
            self.timeouts+=truncated.detach().cpu().to(torch.int64)
            self.on_boundary(self,action,extras)
        data=LongRollout(**{k:torch.stack(v) for k,v in stored.items()},initial_hidden=initial,old_log_std=std)
        data.validate(runner.dimension)
        # Before any optimizer change, collection and recurrent replay agree.
        with torch.no_grad():
            means,_,_=replay_sequence(runner.actor,data.observations,initial,data.episode_start,data.valid)
            logp=BoundedNormal(means,std.expand_as(means)).log_prob(data.actions,data.latents)
            if not torch.allclose(logp[data.valid],data.old_log_probs[data.valid],atol=1e-5,rtol=1e-5):
                raise RuntimeError('collected/replayed joint log probability mismatch')
        return data
