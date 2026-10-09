"""Identity-bound R2 acceptance or user-approved 1000-update full training."""
import argparse
import hashlib
import json
import random
import signal
import subprocess
import time
import traceback
from pathlib import Path
from supervise_training import ROOT,validate_config,write,checkpoint_due
from validate_preflight import inventory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--run_dir',type=Path,required=True)
    p.add_argument('--resume_checkpoint',type=Path);p.add_argument('--resume_sha256')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p);args=p.parse_args()
    config=validate_config(args.config);folder=args.run_dir
    full=config['purpose']=='full_training'
    if bool(args.resume_checkpoint)!=bool(args.resume_sha256):p.error('resume path and SHA-256 required together')
    if args.resume_checkpoint and not full:p.error('only full training supports explicit resume')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=config['implementation_head']:p.error('implementation HEAD changed; prepare inputs again')
    args.device='cuda:0';args.visualizer='none';args.enable_cameras=True
    args.kit_args='--/UJITSO/enabled=false --/UJITSO/geometry=false'
    interrupted=False;launcher=env=None;collector=None;runner=None;checkpoint=None;started=time.time();writer=None
    report=dict(status='running',run_id=folder.name,group=config['group'],num_envs=8,head=head,
        purpose=config['purpose'],update_count=0,effective_samples=0,global_boundaries=0,updates=[],fixed_replays=[],rollouts=[])
    def stop(*unused):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def heartbeat(stage,metrics=None):
        write(folder/'heartbeat.json',dict(run_id=folder.name,heartbeat_epoch_s=time.time(),stage=stage,
            status='running',update_count=runner.update_count if runner else 0,
            effective_samples=collector.samples if collector else 0,latest_checkpoint=checkpoint,
            metrics=metrics or {},gpu_memory_bytes=torch.cuda.memory_allocated() if runner else 0))
        if interrupted: raise InterruptedError('user requested stop')
    try:
        import numpy as np
        import torch
        if full:
            from torch.utils.tensorboard import SummaryWriter
            writer=SummaryWriter(log_dir=str(folder/'tensorboard'))
        heartbeat('startup');launcher=AppLauncher(args)
        import gymnasium as gym
        import omni.usd
        from threadpoolctl import threadpool_limits
        from isaaclab.app import launch_simulation
        from robotarm_magnetic_lab.learning.new_stomach_rl_collector import VectorRolloutCollector,join_rollouts
        from robotarm_magnetic_lab.learning.new_stomach_rl_long_rollout import LongRolloutPPO
        from robotarm_magnetic_lab.learning.new_stomach_rl_checkpoint import save_checkpoint,restore_checkpoint,configuration_sha
        from robotarm_magnetic_lab.runtime.new_stomach_rl_library import FrozenPoseLibrary
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.new_stomach_rl_vector_env import register_vector
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_rl_vector_env_cfg import RobotarmMagneticNewStomachRLVectorCfg
        cfg=RobotarmMagneticNewStomachRLVectorCfg(group=config['group']);cfg.sim.device=args.device;cfg.seed=config['seed']
        random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
        with launch_simulation(cfg,args),threadpool_limits(limits=1,user_api='blas'),(folder/'boundaries.jsonl').open('x') as tape:
            env=gym.make(register_vector(),cfg=cfg,pose_manifest=config['pose_manifest'],mask_path=config['mask']).unwrapped
            enabled=omni.usd.get_context().get_stage().GetPrimAtPath('/physicsScene').GetAttribute('physxScene:enableGPUDynamics').Get()
            report['devices']=dict(environment=str(env.device),simulation=str(env.sim.device),
                camera=str(env.scene['capsule_camera'].device),coverage=[str(r.visibility.device) for r in env.runtime.rows],gpu_dynamics=enabled)
            if not enabled or any(not s.startswith('cuda') for s in [report['devices'][k] for k in ('environment','simulation','camera')]+report['devices']['coverage']):
                raise RuntimeError('actual GPU PhysX/camera/coverage required')
            dimension=9 if config['group'] in 'AB' else 36;runner=LongRolloutPPO(dimension,env.device)
            # Capture before env.step autoreset clears terminal row records.
            captured=[];phase='training';finish=env.runtime.finish_batch
            def record(valid):
                result=finish(valid);captured.clear()
                term=env.action_manager.get_term('magnet')
                for i,row in enumerate(env.runtime.rows):
                    records=row.ten_hz_records[-10:] if bool(valid[i]) else row.ten_hz_records[-1:]
                    if len(records)!=(10 if bool(valid[i]) else 1):raise RuntimeError('missing 10Hz coverage records')
                    if bool(valid[i]) and any(abs(x['sim_time_s']-(row.policy_second-1+(j+1)/10))>1e-9 for j,x in enumerate(records)):
                        raise RuntimeError('10Hz clock mismatch')
                    if config['group'] in 'AC' and row.visual_features.count_nonzero():raise RuntimeError('Blind visual leak')
                    captured.append(dict(row=i,pose_id=env.reset_pose_ids[i],valid=bool(valid[i]),
                        c10=records,c1=dict(row.one_hz_records[-1]),telemetry=dict(term.rows[i].telemetry),
                        actual_issued_history=term.rows[i].executed_history.reshape(4,9).detach().cpu().tolist()))
                return result
            env.runtime.finish_batch=record
            calls=[0]*8;bridge=env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            for i,row in enumerate(bridge.rows):
                original=row.physics_step
                def counted(*a,_i=i,_original=original,**kw):
                    result=_original(*a,**kw);calls[_i]+=1;return result
                row.physics_step=counted
            original_step=env.step
            def audited_step(action):
                calls[:]=[0]*8;began=time.perf_counter();result=original_step(action)
                if calls!=[240]*8 or result[-1]['physical_substeps']!=240:raise RuntimeError('240 magnetic/physical steps required')
                actual=torch.tensor([x['actual_issued_history'] for x in captured],device=env.device).reshape(8,36)
                if not torch.equal(result[-1]['final_obs']['policy'][:,512:],actual):raise RuntimeError('Actor history not actual issued commands')
                tape.write(json.dumps(dict(phase=phase,global_tick=env.lifecycle.global_tick,elapsed_s=time.perf_counter()-began,
                    raw_action=action.detach().cpu().tolist(),valid=result[-1]['valid_transition'].cpu().tolist(),
                    truncated=result[3].cpu().tolist(),final_critic_inputs=result[-1]['final_obs']['critic'].detach().cpu().tolist(),
                    rows=captured),allow_nan=False)+'\n');tape.flush()
                if writer:
                    step=collector.boundaries if collector else 0
                    c0=[x['c10'][-1]['coverage'][0] for x in captured if not x['valid']]
                    c120=[x['c10'][-1]['coverage'][0] for i,x in enumerate(captured) if bool(result[3][i])]
                    tag='episode' if phase in ('training','checkpoint_reset') else phase
                    if c0:writer.add_scalar(tag+'/c0_mean',sum(c0)/len(c0),step)
                    if c120:writer.add_scalar(tag+'/c120_mean',sum(c120)/len(c120),step)
                return result
            env.step=audited_step
            observations,_=env.reset(seed=cfg.seed)
            collector=VectorRolloutCollector(env,runner,observations,lambda *a:heartbeat('sampling'))
            identity=dict(config_sha256=configuration_sha(config),code_sha256=hashlib.sha256(head.encode()).hexdigest(),
                assets_sha256=configuration_sha(config['assets']),weights_sha256=config['weights']['sha256'])
            if args.resume_checkpoint:
                restored=[]
                report['restoration']=restore_checkpoint(args.resume_checkpoint,args.resume_sha256,runner,env.libraries,
                    config,identity,lambda:restored.append(env.reset()[0]),lambda:collector.hidden.zero_())
                collector.fresh(restored[0])
                statistics=report['restoration']['statistics']
                collector.boundaries=int(statistics['global_boundaries'])
                collector.samples=report['restoration']['effective_samples']
                collector.timeouts=torch.tensor(statistics['timeouts'],dtype=torch.int64)
                if collector.boundaries!=runner.update_count*64 or not 0<=runner.update_count<=1000:
                    raise RuntimeError('resume iteration/boundary counters mismatch')
                report['resume_checkpoint']=inventory(args.resume_checkpoint)
                checkpoint=dict(report['resume_checkpoint'],kind='training_boundary')
                report['checkpoint_update']=runner.update_count
                heartbeat('sampling')
            for update in range(runner.update_count,config['max_updates']):
                if not full and update==2:
                    # Explicit full reset ends outstanding episodes; no PhysX continuity claim.
                    phase='checkpoint_reset';observations,_=env.reset();collector.fresh(observations)
                    if env.lifecycle.episode_seconds.any() or env.lifecycle._in_batch:raise RuntimeError('checkpoint not at new episode boundary')
                    checkpoint=save_checkpoint(folder/'boundary_update_0002.pt',runner,env.libraries,config,identity,
                        collector.samples,at_episode_boundary=True,statistics=dict(timeouts=collector.timeouts.tolist()))
                    heartbeat('checkpoint')
                    saved_parameters=[x.detach().clone() for x in list(runner.actor.parameters())+list(runner.critic.parameters())]
                    runner=LongRolloutPPO(dimension,env.device);collector.runner=runner
                    restored=[]
                    result=restore_checkpoint(checkpoint['path'],checkpoint['sha256'],runner,env.libraries,config,identity,
                        lambda:restored.append(env.reset()[0]),lambda:collector.hidden.zero_())
                    collector.fresh(restored[0]);report['restoration']=result
                    report['restoration']['parameters_exact']=all(torch.equal(a,b) for a,b in zip(saved_parameters,
                        list(runner.actor.parameters())+list(runner.critic.parameters())))
                    if runner.update_count!=2 or not report['restoration']['parameters_exact']:
                        raise RuntimeError('new runner restoration mismatch')
                phase='training'
                # Controlled asynchronous reset is a boundary event, never an extra physics step.
                pieces=[]
                for part in ((5,59) if not full and update==0 else (64,)):
                    pieces.append(collector.collect(part))
                    if not full and update==0 and part==5:
                        before=env.lifecycle.global_tick
                        env.reset_rows([0]);collector.observations=env.runtime.observations();collector.hidden[:,0]=0
                        if env.lifecycle.global_tick!=before:raise RuntimeError('partial reset advanced physics')
                data=join_rollouts(pieces)
                from dataclasses import fields
                path=folder/f'rollout_update_{update+1:04d}.pt'
                with path.open('xb') as stream:
                    torch.save({f.name:getattr(data,f.name).detach().cpu() for f in fields(data)},stream)
                report['rollouts'].append(inventory(path))
                heartbeat('update');metric=runner.update(data);report['updates'].append(metric);heartbeat('sampling',metric)
                if writer:
                    for key,value in metric.items():writer.add_scalar('train/'+key,value,runner.update_count)
                    writer.add_scalar('train/reward_mean',float(data.rewards[data.valid].mean()),runner.update_count)
                    writer.add_scalar('train/total_effective_samples',collector.samples,runner.update_count)
                    writer.flush()
                write(folder/'progress_summary.json',dict(run_id=folder.name,update_count=runner.update_count,
                    effective_samples=collector.samples,global_boundaries=collector.boundaries,last_metric=metric))
                if full and checkpoint_due(runner.update_count,config['save_interval']):
                    phase='checkpoint_reset';observations,_=env.reset();collector.fresh(observations)
                    if env.lifecycle.episode_seconds.any() or env.lifecycle._in_batch:raise RuntimeError('checkpoint not at zero-second reset boundary')
                    checkpoint=save_checkpoint(folder/f'update_{runner.update_count:04d}.pt',runner,env.libraries,config,identity,
                        collector.samples,at_episode_boundary=True,
                        statistics=dict(timeouts=collector.timeouts.tolist(),global_boundaries=collector.boundaries))
                    report['checkpoint_update']=runner.update_count;heartbeat('checkpoint')
                print('TRAIN_UPDATE '+json.dumps(dict(group=config['group'],update=runner.update_count,
                    effective_samples=collector.samples,metrics=metric)),flush=True)
            if not collector.timeouts.gt(0).all():raise RuntimeError('120s TIMEOUT not exercised in all rows')
            report.update(update_count=runner.update_count,effective_samples=collector.samples,global_boundaries=collector.boundaries,
                timeouts_per_row=collector.timeouts.tolist(),checkpoint=checkpoint)
            # Frozen train/validation replay is separate from training sample budget.
            for split in ('train','validation'):
                phase=f'fixed_{split}';pose=config[f'fixed_{split}_pose']
                env.libraries=[FrozenPoseLibrary(config['pose_manifest'],split,cfg.seed+i,pose) for i in range(8)]
                obs,_=env.reset();hidden=torch.zeros_like(collector.hidden)
                for _ in range(120):
                    with torch.no_grad():
                        distribution,hidden=runner.actor.distribution(obs['policy'],hidden)
                        action=distribution.mode()
                    obs,reward,term,trunc,extras=env.step(action);heartbeat('sampling')
                if not trunc.all():raise RuntimeError('fixed replay did not reach TIMEOUT')
                report['fixed_replays'].append(dict(split=split,pose_id=pose,seconds=120,status='pass'))
            report.update(status='completed',artifacts=[inventory(folder/'boundaries.jsonl'),inventory(Path(checkpoint['path']))])
    except InterruptedError as error:report.update(status='interrupted',error=str(error))
    except Exception as error:report.update(status='failed',error=str(error),traceback=traceback.format_exc())
    finally:
        if runner and report['status']!='completed':
            try:
                report['interrupted_weights_only']=save_checkpoint(folder/'interrupted_weights_only.pt',runner,
                    env.libraries,config,identity,collector.samples if collector else 0,
                    at_episode_boundary=False,allow_weights_only=True)
            except Exception as error:report['interrupted_checkpoint_error']=str(error)
        if runner:report['update_count']=runner.update_count
        if collector:report.update(effective_samples=collector.samples,global_boundaries=collector.boundaries)
        report.update(started_epoch_s=started,ended_epoch_s=time.time(),elapsed_seconds=time.time()-started)
        write(folder/'summary.json',report)
        if writer:writer.close()
        print('R2_RESULT '+json.dumps(dict(status=report['status'],run_dir=str(folder))),flush=True)
        if env:env.close()
        if launcher:launcher.app.close()
    if report['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
