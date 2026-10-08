"""Gate 6: restored D checkpoint, actual physical tape and native autoreset."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from validate_preflight import audit_assets,DEFAULT_MANIFEST,DEFAULT_MASK,ROOT,persist_before_shutdown


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--pose_split',choices=['train','validation'],default='train')
    parser.add_argument('--seconds',type=int,choices=[4,120],default=120,
        help='4 is only a shortened-timeout lifecycle diagnostic, not final acceptance')
    parser.add_argument('--pose_manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--mask',type=Path,default=DEFAULT_MASK)
    parser.add_argument('--output_root',type=Path,default=ROOT/'artifacts/new_stomach_rl/end_to_end')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    output=args.output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(gate=6,status='running',command=sys.argv,split=args.pose_split,seconds=args.seconds,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        assets=audit_assets(args.pose_manifest,args.mask),checkpoint=dict(path=str(args.checkpoint.resolve()),
            sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()))
    args.enable_cameras=True
    launcher=AppLauncher(args)
    env=None
    try:
        import torch
        import numpy as np
        import gymnasium as gym
        from threadpoolctl import threadpool_limits
        from isaaclab.app import launch_simulation
        from robotarm_magnetic_lab.learning.new_stomach_rl_runner import SmokePPO
        from robotarm_magnetic_lab.runtime.new_stomach_rl_trace import PhysicalStepRecorder
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.new_stomach_rl_env import register_preflight
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_rl_env_cfg import RobotarmMagneticNewStomachRLPreflightCfg
        cfg=RobotarmMagneticNewStomachRLPreflightCfg(group='D')
        cfg.sim.device=args.device;cfg.seed=1008;cfg.episode_length_s=args.seconds
        manifest=json.loads(args.pose_manifest.read_text())
        pose_id=manifest['fixed_live_reload_pose_ids'][args.pose_split][0]
        with launch_simulation(cfg,args),threadpool_limits(limits=1,user_api='blas'):
            env=gym.make(register_preflight(),cfg=cfg,pose_manifest=args.pose_manifest,mask_path=args.mask,
                pose_split=args.pose_split,pose_id=pose_id,record_physics=True).unwrapped
            observation,_=env.reset(seed=1008)
            runtime=env.runtime
            runner=SmokePPO(36,env.device);runner.load(args.checkpoint)
            hidden=torch.zeros((1,1,256),device=env.device)
            ten_rows=[dict(runtime.ten_hz_records[0])];one_rows=[dict(runtime.one_hz_records[0])]
            report.update(pose_id=pose_id,physics_hz=240,policy_hz=1,coverage_hz=10,
                trace_column_groups=PhysicalStepRecorder.columns,rewards=[],safety_projection=[],
                actor_inputs=548,critic_inputs=28,model_updates_during_evaluation=0)
            with (output/'physics_240hz.jsonl').open('w') as physics,(output/'coverage_reward_10hz.jsonl').open('w') as ten,(output/'policy_rgb_1hz.jsonl').open('w') as one:
                ten.write(json.dumps(ten_rows[0])+'\n');one.write(json.dumps(one_rows[0])+'\n')
                for second in range(1,args.seconds+1):
                    before=len(runtime.ten_hz_records)
                    with torch.no_grad():
                        distribution,next_hidden=runner.actor.distribution(observation['policy'],hidden)
                        action=distribution.mode()
                    following,reward,terminated,truncated,extras=env.step(action)
                    if terminated.any() or bool(truncated[0]) != (second==args.seconds):
                        raise RuntimeError('unexpected safety termination or native timeout boundary')
                    terminal=second==args.seconds
                    rows=(env.last_episode_records['ten_hz'][before:] if terminal else runtime.ten_hz_records[before:])
                    rgb_row=(env.last_episode_records['one_hz'][-1] if terminal else runtime.one_hz_records[-1])
                    if len(rows)!=10 or not torch.isfinite(reward).all() or abs(sum(r['total_reward'] for r in rows)-float(reward[0]))>1e-5:
                        raise RuntimeError('formal boundary reward/10Hz mismatch')
                    observed=extras['final_obs']['policy'] if terminal else following['policy']
                    if not torch.equal(observed[:,512:],env.last_policy_history.reshape(1,36)):
                        raise RuntimeError('Actor did not receive actually issued history')
                    ten_rows.extend(rows);one_rows.append(dict(rgb_row))
                    report['rewards'].append(float(reward[0]))
                    report['safety_projection'].append(env.last_policy_telemetry['projection_scale'])
                    for row in rows:
                        ten.write(json.dumps(row,allow_nan=False)+'\n')
                    one.write(json.dumps(dict(rgb_row,raw_policy_action=action[0].cpu().tolist(),
                        actual_issued_history=env.last_policy_history[0].cpu().tolist(),
                        actor_frame_source='final_obs' if terminal else 'step_return',
                        telemetry=env.last_policy_telemetry),allow_nan=False)+'\n')
                    values=env.last_physics_tensor.cpu().tolist()
                    for tick,data in enumerate(values,1):
                        physical_tick=env.last_policy_physics_end_tick-240+tick
                        physics.write(json.dumps(dict(policy_second=second,substep=tick,
                            physics_counter=physical_tick,sim_time_s=(second-1)+tick/240,
                            counted_global_time_s=physical_tick/240,
                            capsule_pose_xyzw=data[:7],capsule_velocity_world=data[7:13],
                            applied_robot_capsule_wrench_si=data[13:25],source_pose_xyzw=data[25:32],
                            joint_actual_rad=data[32:41],joint_issued_reference_rad=data[41:]),allow_nan=False)+'\n')
                    ten.flush();one.flush();physics.flush()
                    observation,hidden=following,next_hidden.detach()
                    if second%10==0 or terminal:
                        print('RL_END_TO_END '+json.dumps(dict(second=second,c10=rgb_row['c10'],c1=rgb_row['coverage'],
                            reward=float(reward[0]),truncated=terminal)),flush=True)
            c10=np.asarray([r['coverage'][0] for r in ten_rows]);c1=np.asarray([r['coverage'][0] for r in one_rows])
            frames=[r['rgb_frame'] for r in one_rows]
            if (len(c10)!=args.seconds*10+1 or len(c1)!=args.seconds+1 or (np.diff(c10)<0).any()
                or (np.diff(c1)<0).any() or (c1>c10[::10]+1e-12).any() or (np.diff(frames)!=1).any()):
                raise RuntimeError('end-to-end clock/coverage/frame invariant failure')
            if (len(runtime.ten_hz_records)!=1 or len(runtime.one_hz_records)!=1
                or env.action_manager.get_term('magnet').executed_history.count_nonzero()
                or runtime.reward._pending.count_nonzero() or env.episode_length_buf.any()
                or not torch.equal(observation['critic'][:,22:26],observation['critic'].new_tensor([[1.,0.,0.,0.]]))):
                raise RuntimeError('native terminal reset inherited old state')
            hidden.zero_()
            with torch.no_grad():
                distribution,_=runner.actor.distribution(observation['policy'],hidden)
                action=distribution.mode()
            next_observation,next_reward,terminated,truncated,_=env.step(action)
            if terminated.any() or truncated.any() or not torch.isfinite(next_reward).all():
                raise RuntimeError('post-terminal reset continuation failed')
            report.update(status='pass',physics_records=args.seconds*240,c10_points=len(c10),c1_points=len(c1),
                c10_final=float(c10[-1]),c1_final=float(c1[-1]),rgb_frame_increments_exact=True,
                cumulative_reward_terms=np.asarray([r['reward_terms'] for r in ten_rows[1:]]).sum(0).tolist(),
                native_same_step_autoreset_passed=True,reset_history_and_c0_passed=True,
                post_reset_first_policy_step_passed=True,reset_c0=runtime.ten_hz_records[0]['coverage'])
    except Exception as error:
        report.update(status='fail',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        persist_before_shutdown(report,output,launcher.app.close)


if __name__=='__main__':
    main()
