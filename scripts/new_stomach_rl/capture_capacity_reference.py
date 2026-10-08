"""P0固定参照采集（用户手动启动，不训练）；保留原单环境执行链路。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

from validate_preflight import ROOT, audit_assets, persist_before_shutdown


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    # AppLauncher pre-parses before registering its flags; defer required
    # checks so --help works without assets and still never constructs Kit.
    parser.add_argument('--mode', choices=['single','chunk'])
    parser.add_argument('--pose_manifest',type=Path)
    parser.add_argument('--mask',type=Path)
    parser.add_argument('--review_bundle',type=Path,default=ROOT/
        'handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/preflight_review_v1')
    parser.add_argument('--output_root',type=Path,default=ROOT/'artifacts/new_stomach_rl_capacity/reference')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.mode is None or args.pose_manifest is None or args.mask is None:
        parser.error('--mode、--pose_manifest和--mask必填')
    from robotarm_magnetic_lab.runtime.new_stomach_rl_review_bundle import (
        check_reference_sources,verify_bundle,INVENTORY_SHA)
    verify_bundle(args.review_bundle,INVENTORY_SHA)
    original_sources=check_reference_sources(ROOT,args.review_bundle)
    # Fixed paths must be supplied; do not depend on a sibling worktree name.
    for file in (args.pose_manifest,args.mask):
        if not file.is_file():
            parser.error(f'needs_input: missing confirmed external asset {file}')
    manifest=json.loads(args.pose_manifest.read_text())
    pose_id=manifest['fixed_live_reload_pose_ids']['train'][0]
    if pose_id!='train-0003':
        parser.error('fixed baseline pose must remain train-0003')
    output=args.output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',stage='P0_reference',mode=args.mode,pose_id=pose_id,
        command=sys.argv,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_preflight_sources_verified=original_sources,
        assets=audit_assets(args.pose_manifest,args.mask),model_updates=0,
        physics_hz=240,policy_hz=1,coverage_hz=10,repeat_runs=2,
        magnetic_input_columns=['source_pose_xyzw_7','capsule_pose_xyzw_7','capsule_velocity_world_6',
            'prior_filtered_wrench_12','prior_coupling_elapsed_1'],
        magnetic_output_columns=['raw_robot_capsule_wrench_12','applied_filtered_wrench_12',
            'coupling_elapsed_1'])
    args.enable_cameras=True
    launcher=AppLauncher(args)
    env=None
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        from threadpoolctl import threadpool_limits
        from isaaclab.app import launch_simulation
        from robotarm_magnetic_lab.runtime.new_stomach_rl_capacity_reference import canonical_actions,MagneticInputTape
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.new_stomach_rl_env import register_preflight
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_rl_env_cfg import RobotarmMagneticNewStomachRLPreflightCfg
        actions=canonical_actions(args.mode)
        (output/'raw_actions.json').write_text(json.dumps(dict(mode=args.mode,
            pose_id=pose_id,seed=1008,actions=actions.tolist()),indent=2)+'\n')
        cfg=RobotarmMagneticNewStomachRLPreflightCfg(group='B' if args.mode=='single' else 'D')
        cfg.sim.device=args.device;cfg.seed=1008
        with launch_simulation(cfg,args),threadpool_limits(limits=1,user_api='blas'):
            env=gym.make(register_preflight(),cfg=cfg,pose_manifest=args.pose_manifest,
                mask_path=args.mask,pose_id=pose_id,pose_split='train',record_physics=True).unwrapped
            import omni.usd
            scene=omni.usd.get_context().get_stage().GetPrimAtPath('/physicsScene')
            gpu_enabled=scene.GetAttribute('physxScene:enableGPUDynamics').Get()
            report['devices']=dict(environment=str(env.device),simulation=str(env.sim.device),
                gpu_dynamics=gpu_enabled)
            if not gpu_enabled or not str(env.sim.device).startswith('cuda'):
                raise RuntimeError('P0 reference requires actual GPU PhysX, not a CLI-only claim')
            bridge=env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            tape=MagneticInputTape(bridge,env.device)
            bridge.physics_step=tape.physics_step
            coverage=env.runtime.coverage
            original_update=coverage._update
            mask_records=[]
            def capture_update(accumulator,visible,sample,tick,records):
                delta=original_update(accumulator,visible,sample,tick,records)
                if tape.active:
                    mask_records.append(dict(branch=10 if accumulator is coverage.c10 else 1,
                        tick=tick,visible=np.packbits(delta.visible_mask[0].cpu().numpy(),bitorder='little'),
                        cumulative=np.packbits(accumulator.mask[0].cpu().numpy(),bitorder='little')))
                return delta
            coverage._update=capture_update
            report['repeats']=[]
            for repeat in range(2):
                observation,_=env.reset(seed=1008)
                report['devices']['camera']=str(env.runtime.rgb.device)
                mask_records.clear()
                vertices=env.runtime.visibility.weights.numel()
                for branch,accumulator in ((10,coverage.c10),(1,coverage.c1)):
                    mask_records.append(dict(branch=branch,tick=0,
                        visible=np.packbits(coverage._boundary_visible[0].cpu().numpy(),bitorder='little'),
                        cumulative=np.packbits(accumulator.mask[0].cpu().numpy(),bitorder='little')))
                folder=output/f'repeat_{repeat}'
                folder.mkdir()
                records=[]
                physics=[]; magnetic_inputs=[]; magnetic_outputs=[]
                for index,raw in enumerate(actions):
                    tape.begin()
                    observation,reward,terminated,truncated,_=env.step(
                        torch.tensor(raw[None],dtype=torch.float32,device=env.device))
                    inputs,outputs=tape.finish()
                    if terminated.any() or truncated.any() or not torch.isfinite(reward).all():
                        raise RuntimeError('reference safety/nonfinite failure; do not hide by resetting')
                    physics.append(env.last_physics_tensor.cpu().numpy())
                    magnetic_inputs.append(inputs.cpu().numpy())
                    magnetic_outputs.append(outputs.cpu().numpy())
                    records.append(dict(policy_second=index+1,raw_action=raw.tolist(),
                        actual_history=env.last_policy_history[0].cpu().tolist(),
                        telemetry=env.last_policy_telemetry,
                        actor_rgb_frame=env.runtime.one_hz_records[-1]['rgb_frame'],reward=float(reward[0])))
                    print('CAPACITY_REFERENCE_PROGRESS '+json.dumps(dict(mode=args.mode,
                        repeat=repeat,seconds=index+1,total_seconds=len(actions))),flush=True)
                if len(mask_records)!=222 or len(env.runtime.ten_hz_records)!=201 or len(env.runtime.one_hz_records)!=21:
                    raise RuntimeError('reference clock/mask capture count mismatch')
                np.savez_compressed(folder/'reference_tape.npz',physics=np.stack(physics),
                    magnetic_inputs=np.stack(magnetic_inputs),magnetic_outputs=np.stack(magnetic_outputs),
                    branch_hz=np.asarray([r['branch'] for r in mask_records]),
                    physics_tick=np.asarray([r['tick'] for r in mask_records]),
                    visible_packed=np.stack([r['visible'] for r in mask_records]),
                    cumulative_packed=np.stack([r['cumulative'] for r in mask_records]),
                    num_vertices=np.asarray(vertices),bitorder=np.asarray('little'))
                for name,rows in (('policy_1hz.jsonl',records),('coverage_reward_10hz.jsonl',
                    env.runtime.ten_hz_records),('rgb_1hz.jsonl',env.runtime.one_hz_records)):
                    with (folder/name).open('w') as stream:
                        for row in rows:
                            stream.write(json.dumps(row,allow_nan=False)+'\n')
                report['repeats'].append(dict(repeat=repeat,policy_seconds=20,physics_steps=4800,
                    magnetic_calls=4800,c10_points=201,c1_points=21,coverage_masks=222))
            bridge.physics_step=tape.original
            coverage._update=original_update
            report.update(status='pass',repeatability_tolerance='not_registered',
                optimization_started=False,model_updates=0)
    except KeyboardInterrupt:
        report.update(status='interrupted',error='user stopped reference capture; not a passing baseline')
    except Exception as error:
        report.update(status='fail',error=dict(type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        # Keep evidence even when Kit exits Python while closing the app.
        files=[dict(path=str(p.resolve()),bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.rglob('*')) if p.is_file()]
        report['artifacts']=files
        persist_before_shutdown(report,output,launcher.app.close)


if __name__=='__main__':
    main()
