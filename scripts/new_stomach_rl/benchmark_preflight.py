"""Isolated sequential capacity workers; no removal of single-env safety guard."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from validate_preflight import audit_assets,DEFAULT_MANIFEST,DEFAULT_MASK,ROOT,persist_before_shutdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_envs',default='1,2,4,8,12')
    parser.add_argument('--level',type=int,default=None,help=argparse.SUPPRESS)
    parser.add_argument('--pose_manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--mask',type=Path,default=DEFAULT_MASK)
    parser.add_argument('--output_directory',type=Path)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = args.output_directory or ROOT/'artifacts/new_stomach_rl/benchmark'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True,exist_ok=True)
    if args.level is None:
        levels = [int(n) for n in args.num_envs.split(',')]
        if levels != [1,2,4,8,12]:
            parser.error('capacity sequence is fixed: 1,2,4,8,12')
        summary = dict(status='running',gate=5,max_verified_num_envs=0,workers=[],not_run=[])
        for index,level in enumerate(levels):
            folder = output/f'envs_{level}'
            command = [str(ROOT/'run_isaaclab.sh'),'-p',str(Path(__file__).resolve()),'--level',str(level),
                '--device',args.device,'--viz','none','--output_directory',str(folder),
                '--pose_manifest',str(args.pose_manifest),'--mask',str(args.mask)]
            with (output/f'envs_{level}.log').open('w') as log:
                result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
            file = folder/'summary.json'
            data = json.loads(file.read_text()) if file.exists() else dict(status='fail',error='worker produced no persistent summary')
            summary['workers'].append(dict(num_envs=level,process_exit=result.returncode,summary=str(file),status=data['status'],command=command))
            if data['status'] != 'pass':
                summary['status'] = 'limited_pass' if summary['max_verified_num_envs'] and data['status'] == 'unsupported' else 'blocked'
                summary['not_run'] = levels[index+1:]
                break
            summary['max_verified_num_envs'] = level
        else:
            summary['status'] = 'pass'
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('RL_BENCHMARK '+json.dumps(summary),flush=True)
        return
    args.enable_cameras = True
    report = dict(status='running',gate=5,num_envs=args.level,command=sys.argv,
        assets=audit_assets(args.pose_manifest,args.mask),repeat_steps=32,warmup_steps=8,long_rollout_steps=64)
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
        cfg = RobotarmMagneticNewStomachRLPreflightCfg(group='D')
        cfg.sim.device = args.device; cfg.seed=1008;cfg.scene.num_envs=args.level
        manifest=json.loads(args.pose_manifest.read_text())
        pose_id=manifest['fixed_live_reload_pose_ids']['train'][0]
        with launch_simulation(cfg,args),threadpool_limits(limits=1,user_api='blas'):
            env=gym.make(register_preflight(),cfg=cfg,pose_manifest=args.pose_manifest,mask_path=args.mask,pose_id=pose_id).unwrapped
            runtime,term=env.runtime,env.action_manager.get_term('magnet')
            bridge=env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            timers={}; timings=[]
            def timed(obj,name,key):
                original=getattr(obj,name)
                def wrapped(*a,**kw):
                    started=time.perf_counter()
                    try:
                        return original(*a,**kw)
                    finally:
                        timers[key]=timers.get(key,0.)+time.perf_counter()-started
                setattr(obj,name,wrapped)
            timed(term,'clearance','collision_nested_s')
            timed(term,'apply_actions','actuator_total_nested_s')
            timed(bridge,'physics_step','magnetic_s')
            timed(runtime.coverage,'update_geometry_at_physics_tick','coverage_10hz_s')
            timed(runtime.reward,'update_reward_10hz','reward_10hz_s')
            timed(runtime,'_encode_visual','visual_encoder_s')
            timed(env.sim,'render','render_s')
            report['timer_note']='nested timers are not additive; CUDA completed at outer step boundaries'
            driver_memory=[]
            def memory_sample():
                text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                import os
                value=next((int(line.split(',')[1]) for line in text.splitlines() if int(line.split(',')[0]) == os.getpid()),0)
                driver_memory.append(value)
            runner=SmokePPO(36,env.device)
            def execute(observations,hidden,block,index,stream):
                timers.clear()
                with torch.no_grad():
                    distribution,following_hidden=runner.actor.distribution(observations['policy'],hidden)
                    action,latent=distribution.sample();logp=distribution.log_prob(action,latent)
                torch.cuda.synchronize();started=time.perf_counter()
                following,reward,terminated,truncated,_=env.step(action)
                torch.cuda.synchronize();wall=time.perf_counter()-started
                if terminated.any() or truncated.any() or not torch.isfinite(reward).all():
                    raise RuntimeError('benchmark safety/nonfinite failure')
                record=dict(block=block,index=index,step_wall_s=wall,planner_s=term.telemetry['planner_wall_s'],
                    projection_scale=term.telemetry['projection_scale'],**timers)
                timings.append(record);stream.write(json.dumps(record)+'\n');stream.flush();memory_sample()
                values=dict(observations=observations['policy'].clone(),critic_inputs=observations['critic'].clone(),
                    actions=action.clone(),latents=latent.clone(),old_log_probs=logp.clone(),rewards=reward.clone(),
                    dones=(terminated|truncated).clone(),old_means=distribution.base.loc.clone())
                return following,following_hidden.detach(),values
            with (output/'step_timings.jsonl').open('w') as stream:
                for repeat in range(2):
                    observation,_=env.reset(seed=1008+repeat)
                    timed(runtime.rgb_clock,'sample','rgb_sync_s')
                    hidden=torch.zeros((1,1,256),device=env.device)
                    for tick in range(8+32):
                        block=f'warmup_{repeat}' if tick<8 else f'repeat_{repeat}'
                        observation,hidden,_=execute(observation,hidden,block,tick,stream)
                    print('RL_BENCHMARK_REPEAT '+str(repeat+1),flush=True)
                report['ppo_updates']=[]
                for repeat in range(2):
                    observation,_=env.reset(seed=1010+repeat)
                    timed(runtime.rgb_clock,'sample','rgb_sync_s')
                    hidden=torch.zeros((1,1,256),device=env.device);initial=hidden.clone()
                    old_std=runner.actor.log_std.detach().clamp(-5.,2.).clone()
                    data=[]
                    for tick in range(64):
                        observation,hidden,row=execute(observation,hidden,f'rollout_{repeat}',tick,stream)
                        data.append(row)
                    rollout=Rollout(**{key:torch.stack([row[key] for row in data]) for key in data[0]},
                        initial_hidden=initial,final_critic_inputs=observation['critic'].clone(),old_log_std=old_std)
                    torch.cuda.synchronize();start=time.perf_counter();metric=runner.update(rollout);torch.cuda.synchronize()
                    metric['ppo_wall_s']=time.perf_counter()-start
                    report['ppo_updates'].append(metric)
                    runner.save(output/f'long_rollout_update_{repeat+1:04d}.pt')
                    print('RL_BENCHMARK_ROLLOUT '+json.dumps(metric),flush=True)
            measured=[r for r in timings if r['block'].startswith('repeat')]
            seconds=sum(r['step_wall_s'] for r in measured)
            projections=[r['projection_scale'] for r in timings]
            report.update(status='pass',env_steps_per_s=len(measured)/seconds,transitions_per_s=len(measured)*env.num_envs/seconds,
                benchmark_steps=len(timings),two_64_step_rollouts=True,
                committed_fraction=sum(x>0 for x in projections)/len(projections),
                projected_fraction=sum(0<x<1 for x in projections)/len(projections),hold_fraction=sum(x==0 for x in projections)/len(projections),
                peak_driver_memory_sampled_mib=max(driver_memory),peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                timer_means={key:sum(r.get(key,0) for r in measured)/len(measured) for key in measured[0] if key.endswith('_s')},
                equivalent_768000_transition_hours=768000/(len(measured)*env.num_envs/seconds)/3600,
                physics_device=env.sim.device,rgb_device=str(runtime.rgb.device),environment_device=env.device)
            with (output/'step_timings.csv').open('w') as stream:
                columns=sorted(set().union(*(row.keys() for row in timings)))
                writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(timings)
    except Exception as error:
        report.update(status='unsupported' if args.level>1 and 'single environment' in str(error) else 'fail',
            error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        persist_before_shutdown(report,output,launcher.app.close)


if __name__=='__main__':
    main()
