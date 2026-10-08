"""At most 2--5 sequential preflight updates; never formal long training."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from validate_preflight import audit_assets, DEFAULT_MANIFEST, DEFAULT_MASK, ROOT, persist_before_shutdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=list('ABCD'),required=True)
    parser.add_argument('--num_envs',type=int,default=1)
    parser.add_argument('--updates',type=int,choices=range(2,6),default=2)
    parser.add_argument('--rollout_steps',type=int,default=8)
    parser.add_argument('--pose_manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--mask',type=Path,default=DEFAULT_MASK)
    parser.add_argument('--output_root',type=Path,default=ROOT/'artifacts/new_stomach_rl/smoke')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 1 or not 2 <= args.rollout_steps <= 64:
        parser.error('single-env isolation currently certified; rollout must be 2..64')
    output = args.output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='running',gate=4,group=args.group,command=sys.argv,assets=audit_assets(args.pose_manifest,args.mask))
    args.enable_cameras = True
    launcher = AppLauncher(args)
    env = None
    try:
        import torch
        import gymnasium as gym
        from threadpoolctl import threadpool_limits
        from isaaclab.app import launch_simulation
        from robotarm_magnetic_lab.learning.new_stomach_rl_runner import SmokePPO,Rollout
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.new_stomach_rl_env import register_preflight
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_rl_env_cfg import RobotarmMagneticNewStomachRLPreflightCfg
        cfg = RobotarmMagneticNewStomachRLPreflightCfg(group=args.group)
        cfg.sim.device = args.device
        cfg.seed = 1008
        manifest = json.loads(args.pose_manifest.read_text())
        pose_id = manifest['fixed_live_reload_pose_ids']['train'][0]
        with launch_simulation(cfg,args),threadpool_limits(limits=1,user_api='blas'):
            wrapped = gym.make(register_preflight(),cfg=cfg,pose_manifest=args.pose_manifest,
                mask_path=args.mask,pose_id=pose_id)
            env = wrapped.unwrapped
            observations,_ = env.reset(seed=1008)
            if not torch.equal(observations['critic'][:,22:26],observations['critic'].new_tensor([[1.,0.,0.,0.]])):
                raise RuntimeError('C0 Critic inherited a recovery phase')
            dimension = 9 if args.group in ('A','B') else 36
            torch.manual_seed(1008)
            runner = SmokePPO(dimension,env.device)
            hidden = torch.zeros((1,1,256),device=env.device)
            report.update(gamma_per_second=runner.gamma,gae_lambda_provisional=runner.gae_lambda,
                task_id=wrapped.spec.id,actor_allowed_inputs=['frozen RGB 512D (or zeros)','last actually issued quarter-second commands 36D'],
                critic_privileged_inputs=['capsule pose 7','velocity 6','joint positions 9','recovery phase 4','C10/C1 2'],
                observation_shapes={k:list(v.shape) for k,v in observations.items()},updates=[],pose_id=pose_id,
                initial_critic_phase=observations['critic'][:,22:26].cpu().tolist(),
                device=env.device,physics_hz=240,policy_hz=1,coverage_hz=10)
            term = env.action_manager.get_term('magnet')
            runtime = env.runtime
            with (output/'policy_1hz.jsonl').open('w') as one,(output/'coverage_reward_10hz.jsonl').open('w') as ten:
                ten.write(json.dumps(runtime.ten_hz_records[0])+'\n')
                one.write(json.dumps(runtime.one_hz_records[0])+'\n')
                for update in range(args.updates):
                    start = time.perf_counter()
                    initial = hidden.detach().clone()
                    old_std = runner.actor.log_std.detach().clamp(-5.,2.).clone()
                    stored = {k:[] for k in ('observations','critic_inputs','actions','latents','old_log_probs','rewards','dones','old_means')}
                    for tick in range(args.rollout_steps):
                        with torch.no_grad():
                            distribution,next_hidden = runner.actor.distribution(observations['policy'],hidden)
                            action,latent = distribution.sample()
                            logp = distribution.log_prob(action,latent)
                        previous_rows = len(runtime.ten_hz_records)
                        following,reward,terminated,truncated,extras = env.step(action)
                        if terminated.any() or truncated.any():
                            raise RuntimeError('short preflight terminated; do not hide/replace failed rollout')
                        if len(runtime.ten_hz_records)-previous_rows != 10:
                            raise RuntimeError('formal step did not emit ten true physical coverage samples')
                        if args.group in ('A','C') and following['policy'][:,:512].count_nonzero():
                            raise RuntimeError('Blind observation leaked visual features')
                        if not torch.equal(following['policy'][:,512:],term.executed_history.reshape(1,36)):
                            raise RuntimeError('Actor history is not actually issued commands')
                        values = dict(observations=observations['policy'],critic_inputs=observations['critic'],
                            actions=action,latents=latent,old_log_probs=logp,rewards=reward,
                            dones=terminated|truncated,old_means=distribution.base.loc)
                        for name,value in values.items():
                            stored[name].append(value.detach().clone())
                        for row in runtime.ten_hz_records[previous_rows:]:
                            ten.write(json.dumps(row,allow_nan=False)+'\n')
                        one.write(json.dumps(dict(runtime.one_hz_records[-1],raw_policy_action=action[0].cpu().tolist(),
                            actual_issued_history=term.executed_history[0].cpu().tolist(),projection=term.telemetry),allow_nan=False)+'\n')
                        one.flush();ten.flush()
                        observations,hidden = following,next_hidden.detach()
                    data = Rollout(**{k:torch.stack(v) for k,v in stored.items()},initial_hidden=initial,
                        final_critic_inputs=observations['critic'].detach().clone(),old_log_std=old_std)
                    metric = runner.update(data)
                    metric.update(update=runner.update_count,wall_s=time.perf_counter()-start)
                    report['updates'].append(metric)
                    runner.save(output/f'update_{runner.update_count:04d}.pt')
                    print('RL_PPO_SMOKE '+json.dumps(metric),flush=True)
            checkpoint = output/f'update_{runner.update_count:04d}.pt'
            restored = SmokePPO(dimension,env.device)
            restored.load(checkpoint)
            with torch.no_grad():
                old_mean,_ = runner.actor.parameters_sequence(observations['policy'],hidden)
                new_mean,_ = restored.actor.parameters_sequence(observations['policy'],hidden)
                if not torch.equal(old_mean,new_mean):
                    raise RuntimeError('checkpoint output differs on identical allowed observations/hidden')
            previous = runtime.coverage.c10_records[-1]['coverage']
            observations,_ = env.reset(seed=1008)
            if not torch.equal(observations['critic'][:,22:26],observations['critic'].new_tensor([[1.,0.,0.,0.]])):
                raise RuntimeError('second reset Critic inherited a recovery phase')
            if (len(runtime.ten_hz_records) != 1 or len(runtime.one_hz_records) != 1
                or term.executed_history.count_nonzero() or env.episode_length_buf.any()
                or runtime.reward._pending.count_nonzero()):
                raise RuntimeError('reset inherited coverage/reward/action history')
            with torch.no_grad():
                distribution,hidden = restored.actor.distribution(observations['policy'],torch.zeros_like(hidden))
                action = distribution.mode()
            returned,reward,terminated,truncated,_ = env.step(action)
            if terminated.any() or truncated.any() or not torch.isfinite(reward).all():
                raise RuntimeError('restored checkpoint/new reset step failed')
            report.update(status='pass',checkpoint={'path':str(checkpoint),'bytes':checkpoint.stat().st_size,
                'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},checkpoint_exact_output=True,
                reset_history_zero=True,previous_c10=previous,new_c0=runtime.ten_hz_records[0]['coverage'],
                restored_step_rgb_frame=runtime.one_hz_records[-1]['rgb_frame'],
                visual_forward_count=runtime.encoder.forward_image_count if runtime.encoder else 0)
    except Exception as error:
        report.update(status='fail',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        persist_before_shutdown(report,output,launcher.app.close)


if __name__ == '__main__':
    main()
