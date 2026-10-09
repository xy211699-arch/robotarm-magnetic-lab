"""Finite P3 vector sampling benchmark; no PPO or training is launched."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
from validate_preflight import ROOT, audit_assets, inventory, persist_before_shutdown

LEVELS = (1, 4, 8, 12, 16, 20)


def schedule():
    # Two 64-boundary rollouts plus 16 extra boundaries ensure row0 reaches
    # its own 120s timeout after the deliberate reset at boundary17.
    return [('timing_0', 8, 32), ('timing_1', 8, 32), ('rollout', 0, 144)]


def check_batch(substeps, calls, valid, terminated):
    if substeps != 240 or any(n != 240 for n in calls):
        raise RuntimeError('240 physical steps and magnetic calls per row required')
    if len(calls) != len(valid) or any(terminated):
        raise RuntimeError('invalid row count or safety termination')
    return sum(bool(v) for v in valid)


def finalize(report, output, env, launcher):
    # Kit can terminate Python during close; emit the event and persist first.
    print('CAPACITY_RESULT '+json.dumps(dict(status=report['status'],output=str(output))),flush=True)
    def shutdown():
        try:
            if env is not None: env.close()
        finally:
            if launcher is not None: launcher.app.close()
    persist_before_shutdown(report,output,shutdown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_envs', type=int, choices=LEVELS, required=True)
    parser.add_argument('--group', choices=list('ABCD'), required=True)
    parser.add_argument('--pose_manifest', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--output_root', type=Path, default=ROOT/'artifacts/new_stomach_rl_capacity/capacity')
    parser.add_argument('--screening', action='store_true', help='8-boundary multi-env smoke only; no capacity qualification')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not str(args.device).startswith('cuda'): parser.error('GPU PhysX required')
    assets = audit_assets(args.pose_manifest, args.mask)
    output = args.output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', stage='P3_capacity', group=args.group, num_envs=args.num_envs,
        physics_hz=240, magnetic_hz=240, rgb_hz=1, coverage_hz=10, model_updates=0,
        assets=assets, screening_only=args.screening, command=sys.argv, head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        blocks=[], partial_reset_checks=[], unverified=['PPO_update_time','formal_training','Chunk_P0_equivalence'])
    launcher = env = None
    rows_log = output/'boundaries.jsonl'
    completed = 0
    build_started = time.perf_counter()
    try:
        args.enable_cameras = True
        launcher = AppLauncher(args)
        import torch
        import gymnasium as gym
        import omni.usd
        from threadpoolctl import threadpool_limits
        from isaaclab.app import launch_simulation
        from robotarm_magnetic_lab.runtime.new_stomach_rl_capacity_reference import canonical_actions
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.new_stomach_rl_vector_env import register_vector
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_rl_vector_env_cfg import RobotarmMagneticNewStomachRLVectorCfg
        cfg = RobotarmMagneticNewStomachRLVectorCfg(group=args.group)
        cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device; cfg.seed = 1008
        with launch_simulation(cfg,args), threadpool_limits(limits=1,user_api='blas'):
            env = gym.make(register_vector(), cfg=cfg, pose_manifest=args.pose_manifest, mask_path=args.mask,
                pose_id='train-0003', pose_split='train').unwrapped
            enabled = omni.usd.get_context().get_stage().GetPrimAtPath('/physicsScene').GetAttribute('physxScene:enableGPUDynamics').Get()
            camera = env.scene['capsule_camera']
            report['devices'] = dict(environment=str(env.device), simulation=str(env.sim.device),
                camera=str(camera.device), gpu_dynamics=enabled,
                coverage=[str(r.visibility.device) for r in env.runtime.rows])
            if not enabled or any(not str(x).startswith('cuda') for x in (env.device,env.sim.device,camera.device)):
                raise RuntimeError('actual GPU simulation/camera required')
            report['startup_build_wall_seconds'] = time.perf_counter()-build_started
            term = env.action_manager.get_term('magnet')
            bridge = env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            calls = [0]*args.num_envs
            for i,row in enumerate(bridge.rows):
                original = row.physics_step
                def counted(*a, _i=i, _original=original, **kw):
                    result = _original(*a,**kw); calls[_i] += 1; return result
                row.physics_step = counted
            boundary = []
            finish = env.runtime.finish_batch
            def captured(valid):
                result = finish(valid); boundary.clear()
                for i,r in enumerate(env.runtime.rows):
                    latest = r.ten_hz_records[-10:] if bool(valid[i]) else r.ten_hz_records[-1:]
                    if len(latest) != (10 if bool(valid[i]) else 1): raise RuntimeError('missing 10Hz reward/coverage')
                    if bool(valid[i]) and any(abs(float(x['sim_time_s'])-(r.policy_second-1+(j+1)/10))>1e-9 for j,x in enumerate(latest)):
                        raise RuntimeError('wrong 10Hz coverage clock')
                    if not torch.isfinite(r.capsule.data.root_link_pose_w.torch).all() or not torch.isfinite(r.capsule.data.root_com_vel_w.torch).all():
                        raise RuntimeError('nonfinite capsule state')
                    if not torch.isfinite(bridge.rows[i]._filtered_wrench).all(): raise RuntimeError('nonfinite magnetic wrench')
                    boundary.append(dict(row=i, valid=bool(valid[i]), c10=latest,
                        c1=dict(r.one_hz_records[-1]), telemetry=dict(term.rows[i].telemetry)))
                return result
            env.runtime.finish_batch = captured
            actions = canonical_actions('single' if args.group in 'AB' else 'chunk')
            torch.cuda.reset_peak_memory_stats(env.device)
            total_valid = steady_valid = 0
            total_seconds = steady_seconds = 0.
            timeouts = [0]*args.num_envs
            projected = rejected = active_commands = 0
            driver_samples = []
            def driver_sample():
                try:
                    text = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i','0'],timeout=3,text=True)
                    driver_samples.append(float(text.strip()))
                except (OSError,ValueError,subprocess.SubprocessError): pass
            def snapshot_other_rows():
                snapshots = []
                for i in range(1,args.num_envs):
                    r=env.runtime.rows[i]; t=term.rows[i]; b=bridge.rows[i]
                    values=[r.capsule.data.root_link_pose_w.torch,r.capsule.data.root_com_vel_w.torch,
                        t.robot.data.joint_pos.torch,t._target,t.executed_history,b._filtered_wrench,b.elapsed,
                        r.coverage.c10.mask,r.coverage.c1.mask,r.visual_features,r.reward._pending,r.reward._counts,
                        env.lifecycle.active[i:i+1],env.lifecycle.episode_seconds[i:i+1],env.lifecycle.episode_id[i:i+1]]
                    snapshots.append(([x.clone() for x in values],r.policy_second))
                return snapshots
            sampling_started=time.perf_counter()
            with rows_log.open('x') as stream:
                for block,warmup,timed in ([('screening',0,8)] if args.screening else schedule()):
                    started=time.perf_counter(); env.reset(seed=1008)
                    torch.cuda.synchronize(env.device); reset_seconds=time.perf_counter()-started
                    total_seconds+=reset_seconds
                    block_valid=0; block_seconds=0.
                    for k in range(warmup+timed):
                        extra_reset_seconds=0.
                        if block=='rollout' and k==17 and args.num_envs>1:
                            before=snapshot_other_rows(); tick=env._sim_step_counter
                            reset_start=time.perf_counter(); env.reset_rows([0]); torch.cuda.synchronize(env.device)
                            extra_reset_seconds=time.perf_counter()-reset_start
                            after=snapshot_other_rows()
                            unchanged=all(ca==cb and all(torch.equal(a,b) for a,b in zip(aa,bb))
                                for (aa,ca),(bb,cb) in zip(before,after)) and tick==env._sim_step_counter
                            report['partial_reset_checks'].append(dict(batch=k,unchanged=unchanged))
                            if not unchanged: raise RuntimeError('partial reset contaminated another row')
                        calls[:]=[0]*args.num_envs
                        action=torch.tensor(np.repeat(actions[k%20][None],args.num_envs,axis=0),dtype=torch.float32,device=env.device)
                        torch.cuda.synchronize(env.device); started=time.perf_counter()
                        obs,reward,terminated,truncated,extras=env.step(action)
                        torch.cuda.synchronize(env.device); elapsed=time.perf_counter()-started
                        valid=extras['valid_transition'].cpu().tolist()
                        n=check_batch(extras['physical_substeps'],calls,valid,terminated.cpu().tolist())
                        for r in boundary:
                            if r['valid']:
                                active_commands+=1
                                projected+=int(float(r['telemetry'].get('projection_scale',1.))<1.)
                                rejected+=int(r['telemetry'].get('status')=='rejected_hold')
                        if any(not torch.isfinite(x).all() for x in obs.values()) or not torch.isfinite(reward).all():
                            raise RuntimeError('nonfinite boundary output')
                        ended=truncated.cpu().tolist(); timeouts=[a+int(b) for a,b in zip(timeouts,ended)]
                        total_valid+=n; block_valid+=n; total_seconds+=elapsed+extra_reset_seconds; block_seconds+=elapsed+extra_reset_seconds
                        if block.startswith('timing') and k>=warmup:
                            steady_valid+=n; steady_seconds+=elapsed
                        completed+=1
                        record=dict(block=block,batch=k,elapsed_s=elapsed,partial_reset_seconds=extra_reset_seconds,
                            physical_substeps=extras['physical_substeps'],magnetic_calls=list(calls),
                            valid_transition=valid,truncated=ended,rows=list(boundary))
                        stream.write(json.dumps(record)+'\n'); stream.flush()
                        if completed%8==0:
                            driver_sample()
                            print('CAPACITY_PROGRESS '+json.dumps(dict(group=args.group,num_envs=args.num_envs,
                                completed=completed,total=8 if args.screening else 224,valid_samples=total_valid,block=block)),flush=True)
                    report['blocks'].append(dict(name=block,warmup_batches=warmup,timed_batches=timed,
                        valid_samples=block_valid,reset_seconds=reset_seconds,step_seconds=block_seconds))
            if completed!=(8 if args.screening else 224) or (not args.screening and any(n<1 for n in timeouts)):
                raise RuntimeError('full budget/120s timeout evidence missing')
            sampling_wall=time.perf_counter()-sampling_started
            report.update(status='screened' if args.screening else 'pass',global_batches=completed,valid_samples=total_valid,
                sampling_wall_seconds=sampling_wall,q_valid=total_valid/sampling_wall,
                measured_reset_and_step_seconds=total_seconds,
                active_commands=active_commands,projected_commands=projected,rejected_hold_commands=rejected,
                projection_ratio=projected/active_commands,rejected_hold_ratio=rejected/active_commands,
                steady_valid_samples=steady_valid,steady_wall_seconds=steady_seconds,
                steady_q_valid=steady_valid/steady_seconds if steady_seconds else None,timeouts_by_row=timeouts,
                cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(env.device),
                cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(env.device),
                process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                driver_memory_samples_mib=driver_samples,driver_sampled_max_mib=max(driver_samples,default=None),
                driver_memory_note='8-boundary samples; not instantaneous hardware peak',
                sampling_note='sampling only, includes reset/HOLD cost; startup/build reported separately; no PPO updates')
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'fail',
            completed_batches=completed,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        report['artifacts']=[inventory(rows_log)] if rows_log.exists() else []
        finalize(report,output,env,launcher)


if __name__=='__main__': main()
