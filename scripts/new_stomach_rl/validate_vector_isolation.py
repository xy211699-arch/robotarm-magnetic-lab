"""双环境隔离验收/独立重复性标定；不训练，不改变物理参数。"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from validate_preflight import ROOT,audit_assets,inventory,persist_before_shutdown


def compare_arrays(actual,expected,atol,label):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return dict(label=label,passed=False,reason='shape/nonfinite mismatch')
    error = float(np.max(np.abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.
    return dict(label=label,passed=bool(np.allclose(actual,expected,rtol=0,atol=atol)),
        max_abs_error=error,atol=atol,rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_envs',type=int,default=2)
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--mode',choices=['single','chunk'],default='single')
    parser.add_argument('--pose_manifest',type=Path)
    parser.add_argument('--mask',type=Path)
    parser.add_argument('--tolerance_manifest',type=Path,default=ROOT/'artifacts/new_stomach_rl_capacity/evidence/p1_preregistration_20261009/tolerance_registration.json')
    parser.add_argument('--output_root',type=Path,default=ROOT/'artifacts/new_stomach_rl_capacity/isolation')
    parser.add_argument('--check_inputs',action='store_true',help='只读核对输入，不创建Kit或仿真进程')
    parser.add_argument('--calibrate_repeatability',action='store_true',help='仅采集三次20秒重复性；不冒称120秒隔离通过')
    parser.add_argument('--acceptance_manifest',type=Path,help='用户授权的独立克隆容差登记；不覆盖P0原登记')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 2 or args.seconds != 120:
        parser.error('P2固定两个环境、120秒，不将短测伪装为完整隔离验收')
    if not args.pose_manifest or not args.mask or not str(args.device).startswith('cuda'):
        parser.error('需要已确认的--pose_manifest、--mask和--device cuda:0')
    if any(not p.is_file() for p in (args.pose_manifest,args.mask,args.tolerance_manifest)):
        parser.error('needs_input: 位姿库、掩码或预登记清单缺失')
    if args.calibrate_repeatability and args.acceptance_manifest:
        parser.error('标定与验收必须独立运行')
    from robotarm_magnetic_lab.runtime.new_stomach_rl_reference_audit import audit_reference_pair
    registered = json.loads(args.tolerance_manifest.read_text())
    roots = [Path(registered['references'][mode]['summary']['path']).parent for mode in ('single','chunk')]
    if audit_reference_pair(*roots) != registered:
        parser.error('优化前参照或预登记内容已变，不自动重登记或放宽')
    assets = audit_assets(args.pose_manifest,args.mask)
    from robotarm_magnetic_lab.runtime.new_stomach_rl_acceptance import (
        BOUNDS,measurements,mask_difference,register,load_manifest,compare as relaxed_compare)
    original_sha=inventory(args.tolerance_manifest)['sha256']
    acceptance=(load_manifest(args.acceptance_manifest,args.mode,original_sha)
                if args.acceptance_manifest else None)
    if args.check_inputs:
        print('VECTOR_ISOLATION_INPUTS '+json.dumps(dict(status='inputs_verified',mode=args.mode,
            gpu_simulation='not_run',assets=assets,tolerance_manifest=inventory(args.tolerance_manifest))),flush=True)
        return
    reference_root = roots[0 if args.mode=='single' else 1]
    frozen = np.load(reference_root/'repeat_0/reference_tape.npz',allow_pickle=False)
    output = args.output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='running',stage='P2_isolation',mode=args.mode,num_envs=2,
        task_seconds=120,physics_hz=240,coverage_hz=10,rgb_hz=1,model_updates=0,
        command=sys.argv,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        assets=assets,tolerance_manifest=inventory(args.tolerance_manifest),
        runs=[],comparisons=[],unverified=['capacity_P3','P1_speedup','GRU_training_integration_R0'],
        calibration_only=args.calibrate_repeatability,
        acceptance_manifest=inventory(args.acceptance_manifest) if args.acceptance_manifest else None,
        predefined_calibration_bounds={name:list(value) for name,value in BOUNDS.items()} if args.calibrate_repeatability else None)
    args.enable_cameras = True
    launcher,env = None,None
    try:
        launcher = AppLauncher(args)
        import torch
        import gymnasium as gym
        import omni.usd
        from threadpoolctl import threadpool_limits
        from isaaclab.app import launch_simulation
        from robotarm_magnetic_lab.runtime.new_stomach_rl_capacity_reference import canonical_actions,MagneticInputTape
        from robotarm_magnetic_lab.runtime.new_stomach_rl_trace import PhysicalStepRecorder
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.new_stomach_rl_vector_env import register_vector
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_rl_vector_env_cfg import RobotarmMagneticNewStomachRLVectorCfg
        cfg = RobotarmMagneticNewStomachRLVectorCfg(group='B' if args.mode=='single' else 'D')
        cfg.scene.num_envs = 2; cfg.sim.device = args.device; cfg.seed = 1008
        with launch_simulation(cfg,args),threadpool_limits(limits=1,user_api='blas'):
            env = gym.make(register_vector(),cfg=cfg,pose_manifest=args.pose_manifest,mask_path=args.mask,
                pose_id='train-0003',pose_split='train').unwrapped
            scene = omni.usd.get_context().get_stage().GetPrimAtPath('/physicsScene')
            gpu_enabled = scene.GetAttribute('physxScene:enableGPUDynamics').Get()
            report['devices'] = dict(environment=str(env.device),simulation=str(env.sim.device),gpu_dynamics=gpu_enabled)
            if not gpu_enabled or not str(env.sim.device).startswith('cuda') or not str(env.device).startswith('cuda'):
                raise RuntimeError('必须是实际GPU PhysX')
            term = env.action_manager.get_term('magnet')
            weights=(env.runtime.rows[0].visibility.weights.cpu().numpy()
                if acceptance or args.calibrate_repeatability else None)
            def compare_signal(key,actual,expected,label):
                if acceptance:
                    return relaxed_compare(key,actual,expected,acceptance,label,weights)
                atol=registered['tolerances'][key]['atol'] if key in registered['tolerances'] else 0.
                return compare_arrays(actual,expected,atol,label)
            bridge = env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            tapes = [MagneticInputTape(row,env.device) for row in bridge.rows]
            for row,tape in zip(bridge.rows,tapes): row.physics_step = tape.physics_step
            traces = [PhysicalStepRecorder(row.capsule,row,magnetic,env.device)
                for row,magnetic in zip(term.rows,bridge.rows)]
            original_update = env.scene.update
            recording = False
            def update(dt):
                original_update(dt)
                if recording and abs(dt-env.physics_dt)<1e-12:
                    for trace in traces: trace.append()
            env.scene.update = update
            original_finish = env.runtime.finish_batch
            latest = []
            def finish(valid):
                result = original_finish(valid)
                latest.clear()
                for row in env.runtime.rows:
                    latest.append(dict(c10=list(row.ten_hz_records[-10:] if row.policy_second else row.ten_hz_records),
                        c1=dict(row.one_hz_records[-1]),
                        history=row.term.executed_history[0].cpu().tolist(),
                        visible=np.packbits(row.coverage._boundary_visible[0].cpu().numpy(),bitorder='little'),
                        cumulative=np.packbits(row.coverage.c10.mask[0].cpu().numpy(),bitorder='little')))
                return result
            env.runtime.finish_batch = finish
            mask_records = []
            mask_batch = 0
            for index,runtime in enumerate(env.runtime.rows):
                coverage = runtime.coverage
                original_mask_update = coverage._update
                def capture(accumulator,visible,sample,tick,records,
                            index=index,coverage=coverage,original=original_mask_update):
                    result = original(accumulator,visible,sample,tick,records)
                    mask_records.append(dict(row=index,batch=mask_batch,tick=tick,
                        hz=10 if accumulator is coverage.c10 else 1,
                        visible=np.packbits(result.visible_mask[0].cpu().numpy(),bitorder='little'),
                        cumulative=np.packbits(accumulator.mask[0].cpu().numpy(),bitorder='little')))
                    return result
                coverage._update = capture
            canonical = canonical_actions(args.mode)
            phase_data = []
            phases=('calibration_0','calibration_1','calibration_2') if args.calibrate_repeatability else ('control','reset_row_0')
            batches=20 if args.calibrate_repeatability else 121
            report['reset_invariants']=[]
            def row_snapshot():
                row=env.runtime.rows[1]; magnetic=bridge.rows[1]
                return dict(pose=row.capsule.data.root_link_pose_w.torch.clone(),
                    velocity=row.capsule.data.root_com_vel_w.torch.clone(),
                    joints=term.rows[1].robot.data.joint_pos.torch.clone(),
                    targets=term.rows[1]._target.clone(),history=term.rows[1].executed_history.clone(),
                    filter=magnetic._filtered_wrench.clone(),elapsed=magnetic.elapsed.clone(),
                    c10=row.coverage.c10.mask.clone(),c1=row.coverage.c1.mask.clone(),
                    visual=row.visual_features.clone(),pending_reward=row.reward._pending.clone(),
                    reward_count=row.reward._counts.clone())
            for phase in phases:
                mask_records.clear();mask_batch = 0
                env.reset(seed=1008)
                report['devices']['camera'] = str(env.runtime.rgb.device)
                if not str(env.runtime.rgb.device).startswith('cuda'):
                    raise RuntimeError('相机张量不在CUDA，不能记为GPU双环境验收')
                folder = output/phase; folder.mkdir()
                rows,physical,magnetic_inputs,magnetic_outputs,visible,cumulative = [],[],[],[],[],[]
                for second in range(batches):
                    mask_batch = second+1
                    if phase=='reset_row_0' and second and second%17==0:
                        before = env._sim_step_counter
                        preserved=row_snapshot()
                        preserved_clocks=(env.runtime.rows[1].policy_second,int(env.lifecycle.episode_seconds[1]),
                            bridge.rows[1].frame_count,len(env.runtime.rows[1].ten_hz_records))
                        env.reset_rows([0])
                        if env._sim_step_counter != before:
                            raise RuntimeError('局部reset额外推进物理')
                        after=row_snapshot()
                        checks={name:bool(torch.equal(value,after[name])) for name,value in preserved.items()}
                        checks['clocks']=preserved_clocks==(env.runtime.rows[1].policy_second,int(env.lifecycle.episode_seconds[1]),
                            bridge.rows[1].frame_count,len(env.runtime.rows[1].ten_hz_records))
                        report['reset_invariants'].append(dict(second=second,checks=checks))
                        if not all(checks.values()): raise RuntimeError('局部reset直接污染未重置行状态')
                    raw = canonical[second] if second<20 else np.zeros(canonical.shape[1])
                    actions = torch.tensor(np.stack((raw,raw)),dtype=torch.float32,device=env.device)
                    for trace,tape in zip(traces,tapes): trace.begin();tape.begin()
                    recording = True
                    try:
                        observation,reward,terminated,truncated,extras = env.step(actions)
                    finally:
                        recording = False
                    signals = [tape.finish() for tape in tapes]
                    physical.append(np.stack([trace.finish().cpu().numpy() for trace in traces]))
                    magnetic_inputs.append(np.stack([v[0].cpu().numpy() for v in signals]))
                    magnetic_outputs.append(np.stack([v[1].cpu().numpy() for v in signals]))
                    visible.append(np.stack([v['visible'] for v in latest]))
                    cumulative.append(np.stack([v['cumulative'] for v in latest]))
                    row = dict(second=second+1,global_physics_tick=extras['global_physics_tick'],
                        valid_transition=extras['valid_transition'].cpu().tolist(),
                        episode_start=extras['episode_start'].cpu().tolist(),
                        episode_seconds=extras['episode_seconds'].cpu().tolist(),
                        terminated=terminated.cpu().tolist(),truncated=truncated.cpu().tolist(),
                        final_mask=extras['final_mask'].cpu().tolist(),physical_substeps=extras['physical_substeps'],
                        reward=reward.cpu().tolist(),history=[v['history'] for v in latest],
                        final_critic=(extras['final_obs']['critic'].cpu().tolist() if truncated.any() else None),
                        boundaries=[dict(c10=v['c10'],c1=v['c1']) for v in latest])
                    if extras['physical_substeps'] != 240 or (second<120 and not extras['valid_transition'][1]):
                        raise RuntimeError('未重置行丢步或每秒物理步不等于240')
                    if second==119 and not bool(truncated[1]): raise RuntimeError('120秒TIMEOUT缺失')
                    if second==120 and (bool(extras['valid_transition'][1]) or not bool(extras['episode_start'][1])):
                        raise RuntimeError('TIMEOUT后的正常一秒批次没有执行WARMUP')
                    rows.append(row)
                    with (folder/'boundaries.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                    if phase=='control' and second==19:
                        prefix=[]
                        origin=term.rows[1].env_origin
                        for key,values in (('physics',physical),('magnetic_inputs',magnetic_inputs),('magnetic_outputs',magnetic_outputs)):
                            local=np.stack(values)[:,1].astype(np.float64)
                            if key=='physics': local[:,:,:3]-=origin;local[:,:,25:28]-=origin
                            if key=='magnetic_inputs': local[:,:,:3]-=origin;local[:,:,7:10]-=origin
                            prefix.append(compare_signal(key,local,frozen[key],'prefix-N1-N2-'+key))
                        report['paired_prefix_20s']=prefix
                        if not all(v['passed'] for v in prefix):
                            raise RuntimeError('前20秒N1/N2配对失败，停止剩余GPU批次；不继续放宽已登记标准')
                    if second%10==0 or second==120:
                        print('VECTOR_ISOLATION_PROGRESS '+json.dumps(dict(phase=phase,seconds=second+1,total=batches)),flush=True)
                data = dict(physics=np.stack(physical),magnetic_inputs=np.stack(magnetic_inputs),
                    magnetic_outputs=np.stack(magnetic_outputs),visible_packed=np.stack(visible),cumulative_packed=np.stack(cumulative),
                    mask_rows=np.asarray([v['row'] for v in mask_records]),
                    mask_ticks=np.asarray([v['tick'] for v in mask_records]),
                    mask_hz=np.asarray([v['hz'] for v in mask_records]),
                    mask_global_batches=np.asarray([v['batch'] for v in mask_records]),
                    mask_visible_packed=np.stack([v['visible'] for v in mask_records]),
                    mask_cumulative_packed=np.stack([v['cumulative'] for v in mask_records]))
                np.savez_compressed(folder/'isolation_tape.npz',**data)
                phase_data.append((data,rows))
                report['runs'].append(dict(phase=phase,global_batches=batches,physics_steps=batches*240,
                    row_1_valid_transitions=sum(v['valid_transition'][1] for v in rows),row_1_timeout_count=sum(v['truncated'][1] for v in rows)))
            if args.calibrate_repeatability:
                maxima={name:0. for name in BOUNDS}
                for repeat,(data,_) in enumerate(phase_data):
                    for index in (0,1):
                        for key in ('physics','magnetic_inputs','magnetic_outputs'):
                            local=data[key][:,index].astype(float).copy()
                            origin=term.rows[index].env_origin
                            if key=='physics': local[...,:3]-=origin;local[...,25:28]-=origin
                            if key=='magnetic_inputs': local[...,:3]-=origin;local[...,7:10]-=origin
                            for name,value in measurements(key,local,frozen[key]).items(): maxima[name]=max(maxima[name],value)
                            if repeat:
                                for name,value in measurements(key,data[key][:,index],phase_data[0][0][key][:,index]).items(): maxima[name]=max(maxima[name],value)
                        for hz in (10,1):
                            selected=(data['mask_rows']==index)&(data['mask_hz']==hz)
                            baseline=(phase_data[0][0]['mask_rows']==index)&(phase_data[0][0]['mask_hz']==hz)
                            for key,target in (('mask_visible_packed','visible_packed'),('mask_cumulative_packed','cumulative_packed')):
                                value=mask_difference(data[key][selected],frozen[target][frozen['branch_hz']==hz],weights)
                                maxima['mask_area_fraction']=max(maxima['mask_area_fraction'],value)
                                if repeat:
                                    value=mask_difference(data[key][selected],phase_data[0][0][key][baseline],weights)
                                    maxima['mask_area_fraction']=max(maxima['mask_area_fraction'],value)
                report['observed_calibration_maxima']=maxima
                calibrated=register(maxima,args.mode,original_sha,[inventory(output/phase/'isolation_tape.npz') for phase in phases])
                (output/'acceptance_manifest.json').write_text(json.dumps(calibrated,indent=2)+'\n')
                report.update(status='calibrated',full_isolation='not_run')
                return
            for key in ('physics','magnetic_inputs','magnetic_outputs','visible_packed','cumulative_packed'):
                report['comparisons'].append(compare_signal(key,phase_data[1][0][key][:,1],phase_data[0][0][key][:,1],'reset-isolation-'+key))
            for key in ('mask_ticks','mask_hz','mask_global_batches','mask_visible_packed','mask_cumulative_packed'):
                first = phase_data[0][0];second = phase_data[1][0]
                a,b=second[key][second['mask_rows']==1],first[key][first['mask_rows']==1]
                report['comparisons'].append(compare_signal(key,a,b,'reset-isolation-'+key) if 'packed' in key
                    else compare_arrays(a,b,0.,'reset-isolation-'+key))
            # No changed tolerance for clone float32 translation. A discrepancy
            # is retained as a failed pairing, not hidden by final coverage.
            origin = term.rows[1].env_origin
            for key in ('physics','magnetic_inputs','magnetic_outputs'):
                local = phase_data[0][0][key][:20,1].astype(np.float64)
                if key=='physics': local[:,:,:3]-=origin;local[:,:,25:28]-=origin
                if key=='magnetic_inputs': local[:,:,:3]-=origin;local[:,:,7:10]-=origin
                report['comparisons'].append(compare_signal(key,local,frozen[key],'N1-N2-'+key))
            original_policy = [json.loads(line) for line in (reference_root/'repeat_0/policy_1hz.jsonl').read_text().splitlines()]
            for key in ('actual_history','reward'):
                actual = np.asarray([v['history'][1] if key=='actual_history' else v['reward'][1]
                    for v in phase_data[0][1][:20]])
                expected = np.asarray([v[key] for v in original_policy])
                tolerance=(acceptance['limits']['issued_joints_rad'] if key=='actual_history' else
                    200*acceptance['limits']['mask_area_fraction']+.5) if acceptance else 0.
                report['comparisons'].append(compare_arrays(actual,expected,tolerance,'N1-N2-'+key))
            first = phase_data[0][0]
            for hz in (10,1):
                selected = (first['mask_rows']==1)&(first['mask_hz']==hz)&(first['mask_global_batches']<=20)
                for key,original_key in (('mask_visible_packed','visible_packed'),('mask_cumulative_packed','cumulative_packed')):
                    report['comparisons'].append(compare_signal(key,first[key][selected],frozen[original_key][frozen['branch_hz']==hz],f'N1-N2-{hz}Hz-{key}'))
            left,right = phase_data[0][1],phase_data[1][1]
            for index,(a,b) in enumerate(zip(left,right)):
                if a['boundaries'][1] != b['boundaries'][1] or a['reward'][1] != b['reward'][1]:
                    # RGB digest is renderer-dependent even in P0 repeats.
                    aa=dict(a['boundaries'][1]['c1']);bb=dict(b['boundaries'][1]['c1'])
                    for key in ('rgb_sha256','forced_capture'):
                        aa.pop(key,None);bb.pop(key,None)
                    if acceptance:
                        for x,y in zip(a['boundaries'][1]['c10'],b['boundaries'][1]['c10']):
                            for key in ('physics_tick','sample_id','sim_time_s'):
                                if x[key]!=y[key]: raise RuntimeError('未重置行10Hz时钟受到污染')
                        for key in ('policy_second','sim_time_s','rgb_frame'):
                            if aa[key]!=bb[key]: raise RuntimeError('未重置行RGB时钟受到污染')
                        for v in (a,b):
                            active_records=[x for x in v['boundaries'][1]['c10'] if 'reward_terms' in x]
                            if active_records:
                                if any(abs(sum(x['reward_terms'])-x['total_reward'])>1e-5 for x in active_records): raise RuntimeError('奖励项加总不一致')
                                if abs(sum(x['total_reward'] for x in active_records)-v['reward'][1])>1e-5: raise RuntimeError('10Hz奖励与1Hz加总不一致')
                        if len(a['boundaries'][1]['c10'])!=len(b['boundaries'][1]['c10']): raise RuntimeError('10Hz帧数不一致')
                    elif a['boundaries'][1]['c10']!=b['boundaries'][1]['c10'] or aa!=bb or a['reward'][1]!=b['reward'][1]:
                        raise RuntimeError(f'未重置行覆盖/奖励/RGB时钟受到污染: batch {index}')
            report['status'] = 'pass' if all(v['passed'] for v in report['comparisons']) else 'fail'
            report['artifacts'] = [inventory(p) for p in output.rglob('*') if p.is_file()]
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    except Exception as error:
        report.update(status='fail',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        if 'physical' in locals() and physical:
            try:
                np.savez_compressed(output/'failed_partial_tape.npz',physics=np.stack(physical),
                    magnetic_inputs=np.stack(magnetic_inputs),magnetic_outputs=np.stack(magnetic_outputs),
                    mask_rows=np.asarray([v['row'] for v in mask_records]),
                    mask_ticks=np.asarray([v['tick'] for v in mask_records]),
                    mask_hz=np.asarray([v['hz'] for v in mask_records]),
                    mask_visible_packed=np.stack([v['visible'] for v in mask_records]),
                    mask_cumulative_packed=np.stack([v['cumulative'] for v in mask_records]))
            except Exception as secondary:
                report['partial_tape_error'] = str(secondary)
    finally:
        frozen.close()
        report['artifacts'] = [inventory(p) for p in output.rglob('*') if p.is_file()]
        print('VECTOR_ISOLATION_RESULT '+json.dumps(dict(status=report['status'],output=str(output),
            failed_comparisons=[v['label'] for v in report['comparisons'] if not v['passed']])),flush=True)
        def shutdown():
            try:
                if env is not None: env.close()
            finally:
                if launcher is not None: launcher.app.close()
        persist_before_shutdown(report,output,shutdown)
    if report['status'] != 'pass': raise SystemExit(1)


if __name__=='__main__': main()
