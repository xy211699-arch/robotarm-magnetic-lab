"""R0 vector boundary math; no simulator or formal training launch.

Timeout bootstrap is explicit: callers supply the Critic value of the saved
pre-reset final observation. A reset observation is never a substitute.
"""
from dataclasses import dataclass
import torch
from .new_stomach_rl_distribution import BoundedNormal
from .new_stomach_rl_runner import SmokePPO


def boundary_gae(rewards, values, bootstrap_values, valid, episode_start,
                 terminated, truncated, gamma, lam):
    shape=rewards.shape
    if rewards.ndim!=2 or not shape[0] or not shape[1] or any(x.shape!=shape for x in (values,bootstrap_values,valid,episode_start,terminated,truncated)):
        raise ValueError('all GAE inputs must have shape [T,N]')
    if not rewards.is_floating_point() or any(x.dtype!=rewards.dtype for x in (values,bootstrap_values)):
        raise ValueError('reward/value/next-value dtypes must match')
    if any(x.dtype!=torch.bool for x in (valid,episode_start,terminated,truncated)):
        raise ValueError('boundary masks must be boolean')
    if not 0<=gamma<=1 or not 0<=lam<=1: raise ValueError('invalid discount')
    if any(not torch.isfinite(x[valid]).all() for x in (rewards,values,bootstrap_values)):
        raise ValueError('nonfinite active GAE sample')
    reward=torch.where(valid,rewards,0.)
    value=torch.where(valid,values,0.)
    bootstrap=torch.where(valid & ~terminated,bootstrap_values,0.)
    advantages=torch.zeros_like(values)
    acc=torch.zeros_like(values[0])
    for t in reversed(range(len(values))):
        continuation=valid[t] & ~terminated[t] & ~truncated[t]
        if t+1<len(values): continuation=continuation & valid[t+1] & ~episode_start[t+1]
        else: continuation=torch.zeros_like(continuation)
        delta=reward[t]+gamma*bootstrap[t]-value[t]
        acc=torch.where(valid[t],delta+gamma*lam*torch.where(continuation,acc,0.),0.)
        advantages[t]=acc
    return advantages,torch.where(valid,advantages+value,0.)


def recurrent_step(actor, observation, hidden, episode_start, valid):
    n=observation.shape[0]
    if observation.shape!=(n,548) or hidden.shape!=(1,n,256) or episode_start.shape!=(n,) or valid.shape!=(n,):
        raise ValueError('recurrent row shapes differ')
    if episode_start.dtype!=torch.bool or valid.dtype!=torch.bool: raise ValueError('boolean row masks required')
    cleared=hidden*~episode_start[None,:,None]
    clean=torch.where(valid[:,None],observation,0.)
    means,following=actor.parameters_sequence(clean,cleared)
    # WARMUP/padding does not advance hidden. A reset clears only its row.
    return torch.where(valid[:,None],means,0.),torch.where(valid[None,:,None],following,cleared)


def replay_sequence(actor,observations,initial_hidden,episode_start,valid):
    if observations.ndim!=3 or episode_start.shape!=observations.shape[:2] or valid.shape!=episode_start.shape:
        raise ValueError('sequence masks must match [T,N]')
    hidden=initial_hidden
    means=[];before=[]
    for t in range(len(observations)):
        before.append(hidden)
        mean,hidden=recurrent_step(actor,observations[t],hidden,episode_start[t],valid[t])
        means.append(mean)
    return torch.stack(means),hidden,torch.stack(before)


@dataclass
class LongRollout:
    observations: torch.Tensor
    critic_inputs: torch.Tensor
    actions: torch.Tensor
    latents: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    bootstrap_values: torch.Tensor
    valid: torch.Tensor
    episode_start: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_critic_inputs: torch.Tensor  # [T,N,28], captured BEFORE autoreset
    executed_history: torch.Tensor   # [T,N,4,9], never used as policy sample
    initial_hidden: torch.Tensor
    old_means: torch.Tensor
    old_log_std: torch.Tensor

    def validate(self,dimension):
        t,n=self.valid.shape
        expected={'observations':(t,n,548),'critic_inputs':(t,n,28),'actions':(t,n,dimension),
            'latents':(t,n,dimension),'old_means':(t,n,dimension),'final_critic_inputs':(t,n,28),
            'executed_history':(t,n,4,9),'initial_hidden':(1,n,256),'old_log_std':(dimension,)}
        expected.update({name:(t,n) for name in ('old_log_probs','rewards','values','bootstrap_values',
            'valid','episode_start','terminated','truncated')})
        for name,shape in expected.items():
            x=getattr(self,name)
            if x.shape!=shape or not torch.isfinite(x).all(): raise ValueError(f'invalid rollout {name}')
        if not self.valid.any(): raise ValueError('rollout has no effective samples')
        if any(getattr(self,name).dtype!=torch.bool for name in ('valid','episode_start','terminated','truncated')):
            raise ValueError('boolean rollout masks required')
        if not torch.allclose(self.actions,self.latents.tanh(),atol=1e-6,rtol=1e-6):
            raise ValueError('PPO requires sampled actions/latents, not projected commands')


def sequence_chunks(observations,valid,episode_start,hidden_before,length):
    """Contiguous row sequences with right padding; never shuffle time steps."""
    if length<1: raise ValueError('positive sequence length required')
    t,n,_=observations.shape
    if valid.shape!=(t,n) or episode_start.shape!=(t,n) or hidden_before.shape!=(t,1,n,256):
        raise ValueError('sequence shape mismatch')
    for row in range(n):
        for start in range(0,t,length):
            count=min(length,t-start)
            obs=observations.new_zeros((length,1,548));obs[:count]=observations[start:start+count,row:row+1]
            mask=torch.zeros((length,1),dtype=torch.bool,device=valid.device);mask[:count]=valid[start:start+count,row:row+1]
            reset=torch.zeros_like(mask);reset[:count]=episode_start[start:start+count,row:row+1]
            yield dict(row=row,start=start,count=count,observations=obs,valid=mask,episode_start=reset,
                initial_hidden=hidden_before[start,:,row:row+1].detach().clone())


class LongRolloutPPO(SmokePPO):
    """Provisional R0 full-sequence update; old single smoke is unchanged.

    Uses the original model, optimizer and two-epoch smoke constants. Formal
    epochs/minibatches/sample budget/checkpoint semantics await R1/R2 approval.
    """
    def update(self,data):
        data.validate(self.dimension)
        with torch.no_grad():
            advantages,returns=boundary_gae(data.rewards,data.values,data.bootstrap_values,data.valid,
                data.episode_start,data.terminated,data.truncated,self.gamma,self.gae_lambda)
            active=advantages[data.valid]
            advantages=torch.where(data.valid,(advantages-active.mean())/active.std(unbiased=False).clamp_min(1e-8),0.)
        parameters=list(self.actor.parameters())+list(self.critic.parameters())
        for _ in range(2):
            means,_,_=replay_sequence(self.actor,data.observations,data.initial_hidden.detach(),data.episode_start,data.valid)
            distribution=BoundedNormal(means,self.actor.log_std.clamp(-5.,2.).expand_as(means))
            logp=distribution.log_prob(data.actions,data.latents)
            ratio=(logp[data.valid]-data.old_log_probs[data.valid]).exp()
            a=advantages[data.valid]
            policy=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
            critic=self.critic(data.critic_inputs).squeeze(-1)
            value_loss=(critic[data.valid]-returns[data.valid]).square().mean()
            loss=policy+.5*value_loss
            if not torch.isfinite(loss): raise RuntimeError('nonfinite vector PPO loss')
            self.optimizer.zero_grad();loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError('nonfinite vector PPO gradient')
            grad=torch.nn.utils.clip_grad_norm_(parameters,1.);self.optimizer.step()
        self.update_count+=1
        return dict(update=self.update_count,effective_samples=int(data.valid.sum()),policy_loss=float(policy.detach()),
            value_loss=float(value_loss.detach()),gradient_norm=float(grad))
