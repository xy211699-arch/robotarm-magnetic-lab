"""P1原链路只读热点采集；audit离线运行，run必须由用户手动启动。"""
import argparse
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

from validate_preflight import ROOT, PKG, inventory


def pure_module(name):
    spec = importlib.util.spec_from_file_location(name, PKG/'runtime'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(argv):
    parser = argparse.ArgumentParser(description='复核原参照，并在任何优化前登记容差；不启动仿真。')
    parser.add_argument('--single_reference', type=Path, required=True)
    parser.add_argument('--chunk_reference', type=Path, required=True)
    parser.add_argument('--output_directory', type=Path, required=True)
    args = parser.parse_args(argv)
    result = pure_module('new_stomach_rl_reference_audit').audit_reference_pair(args.single_reference,args.chunk_reference)
    args.output_directory.mkdir(parents=True,exist_ok=False)
    file = args.output_directory/'tolerance_registration.json'
    file.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('P1_REFERENCE_AUDIT '+json.dumps(dict(status=result['status'],output=str(file.resolve()),
          tolerances=result['tolerances'],rgb_pixels=result['rgb_pixels'])),flush=True)


def forwarded_arguments(argv):
    """Strip only this diagnostic CLI's flags; preserve original Kit arguments."""
    output = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ('--timing','--trace','--save_rgb_pixels'):
            i += 1
        elif token == '--tolerance_manifest':
            i += 2
        elif token.startswith('--tolerance_manifest='):
            i += 1
        else:
            output.append(token)
            i += 1
    return output


def run(argv):
    parser = argparse.ArgumentParser(description='原20动作×2重放，分层CPU计时及外层CUDA完成计时；无模型优化/训练。')
    parser.add_argument('--mode',choices=['single','chunk'])
    parser.add_argument('--pose_manifest',type=Path)
    parser.add_argument('--mask',type=Path)
    parser.add_argument('--review_bundle',type=Path,default=ROOT/'handoffs/attachments/NEW-STOMACH-RL-CAPACITY-20261008/preflight_review_v1')
    parser.add_argument('--output_root',type=Path,default=ROOT/'artifacts/new_stomach_rl_capacity/profile')
    parser.add_argument('--tolerance_manifest',type=Path)
    parser.add_argument('--timing',action='store_true',help='显式启用计时；默认关闭，不在内层插入CUDA同步')
    parser.add_argument('--trace',action='store_true',help='仅首个动作采集torch profiler trace，该步不计稳态吞吐')
    parser.add_argument('--save_rgb_pixels',action='store_true',help='保存原始RGB便于建立像素重复性参照，增加CPU内存/记录开销')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if any(x is None for x in (args.mode,args.pose_manifest,args.mask,args.tolerance_manifest)):
        parser.error('--mode、--pose_manifest、--mask和--tolerance_manifest必填')
    if args.trace and not args.timing:
        parser.error('--trace必须显式配合--timing')
    auditing = pure_module('new_stomach_rl_reference_audit')
    registration = auditing.read_json(args.tolerance_manifest)
    if registration.get('status') != 'pass' or registration.get('optimization_started') is not False:
        parser.error('必须使用尚未读取优化结果的原始参照登记文件')
    original = {mode:Path(r['summary']['path']).parent for mode,r in registration['references'].items()}
    checked = auditing.audit_reference_pair(original['single'],original['chunk'])
    if checked != registration:
        parser.error('原参照/登记文件发生变化，禁止静默重登记或放宽容差')
    if not str(args.device).startswith('cuda'):
        parser.error('P1运行要求GPU PhysX，不能用CPU运行冒称GPU计时')

    # Reuse the complete validated capture implementation, including original
    # task, assets, raw actions, fixed pose, model, clocks and fail-closed tape.
    import capture_capacity_reference as capture
    import gymnasium as gym
    profiling = pure_module('new_stomach_rl_capacity_profile')
    state = dict(profile=None, repeat=-1, active=0, pixels=[[],[]], pixel_frames=[[],[]],
                 sampling_start=None, sampling_wall=None, trace=None)
    hooks = ExitStack()
    requested_command = list(sys.argv)
    old_make, old_persist, old_argv = gym.make, capture.persist_before_shutdown, sys.argv

    def attach(*positional,**keyword):
        made = old_make(*positional,**keyword)
        env = made.unwrapped
        import torch
        from robotarm_magnetic_lab.runtime.new_stomach_rl_rgb import PolicyRGBBoundary
        p = profiling.CapacityProfile(enabled=args.timing,synchronize=lambda:torch.cuda.synchronize(env.device),
                                      annotate=torch.profiler.record_function)
        state['profile'] = p
        bridge = env.event_manager.get_term_cfg('magnetic_collision_bridge').func
        term = env.action_manager.get_term('magnet')
        for obj,method,label in (
            (term,'process_actions','planning'),(term,'clearance','collision_nested'),
            (term,'apply_actions','actuator'),(bridge,'physics_step','magnetic'),
            (bridge.model,'force_torque_on_cube_si','cube_ft_nested'),
            (bridge.model,'force_torque_si','cylinder_ft_nested'),
            (bridge,'_update_field_visualization','guide_visualization_nested'),
            (bridge,'_log','bridge_log_nested'),(env.sim,'step','simulation_step'),
            (env.sim,'render','render_nested'),(env.sim.physics_manager,'forward','pose_forward_nested'),
            (env.runtime.optics,'forward_physics','optics_forward_nested'),
            (env.runtime.visibility.raycaster,'query','first_hit_query_nested'),
            (env.scene,'update','scene_update'),
            (env.runtime.coverage,'update_geometry_at_physics_tick','coverage_10hz'),
            (env.runtime.reward,'update_reward_10hz','reward_10hz'),
            (env.runtime,'_encode_visual','visual_encoder'),
            (PolicyRGBBoundary,'sample','rgb_sample')):
            hooks.enter_context(p.patch(obj,method,label))
        reset, step = env.reset, env.step
        def record_pixels():
            if args.save_rgb_pixels:
                state['pixels'][state['repeat']].append(env.runtime.rgb[0].detach().cpu().numpy().copy())
                state['pixel_frames'][state['repeat']].append(int(env.runtime.rgb_frames[0]))
        def resetting(*a,**kw):
            if state['sampling_start'] is None:
                state['sampling_start'] = time.perf_counter()
            state['repeat'] += 1
            if state['repeat'] >= 2:
                raise RuntimeError('profile must not add a third reset or silently skip a fault')
            with p.boundary(valid_rows=0,phase='reset') as row:
                result = reset(*a,**kw)
                row['repeat'] = state['repeat']
            record_pixels()
            return result
        def stepping(action):
            traced = args.trace and state['active'] == 0
            trace = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False)
                     if traced else nullcontext())
            with p.boundary(valid_rows=1,phase='active',trace=traced) as row:
                with trace as tracing:
                    result = step(action)
                row['repeat'] = state['repeat']
                row['policy_second'] = int(env.runtime.policy_second)
                for label,count in (('actuator',240),('simulation_step',240),('magnetic',240),
                                    ('cube_ft_nested',240),('cylinder_ft_nested',240),
                                    ('coverage_10hz',10),('reward_10hz',10),('rgb_sample',1)):
                    if row['calls'].get(label) != count:
                        raise RuntimeError(f'profile original clock mismatch: {label}={row["calls"].get(label)}, expected {count}')
            if traced:
                state['trace'] = tracing
            state['active'] += 1
            record_pixels()
            return result
        env.reset, env.step = resetting, stepping
        hooks.callback(setattr,env,'reset',reset)
        hooks.callback(setattr,env,'step',step)
        return made

    def prepare_profile(report,output):
        import numpy as np
        report['stage'] = 'P1_original_profile'
        report['profile_command'] = requested_command
        report['tolerance_registration'] = inventory(args.tolerance_manifest)
        report['optimized_backend'] = False
        if state['sampling_start'] is not None:
            state['sampling_wall'] = time.perf_counter()-state['sampling_start']
        p = state['profile']
        if p is not None:
            with (output/'profile_boundaries.jsonl').open('w') as stream:
                for row in p.records:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
            profile = p.summary()
            profile['sampling_including_recording_wall_s'] = state['sampling_wall']
            profile['q_valid_including_recording_reset'] = (profile['valid_transitions']/state['sampling_wall']
                if state['sampling_wall'] else None)
            profile['physical_recording_enabled'] = True
            profile['rgb_pixels_saved'] = args.save_rgb_pixels
            profile['performance_claim'] = 'original diagnostic only, not optimized throughput or P3 capacity'
            report['profile'] = profile
        try:
            if report['status'] == 'pass':
                base = original[args.mode]
                comparisons = []
                for repeat in range(2):
                    folder = output/f'repeat_{repeat}'
                    if args.save_rgb_pixels:
                        if len(state['pixels'][repeat]) != 21:
                            raise RuntimeError('21 actual RGB frames required including C0')
                        np.savez_compressed(folder/'rgb_pixels.npz',pixels=np.stack(state['pixels'][repeat]),
                                            frames=np.array(state['pixel_frames'][repeat]))
                    with np.load(folder/'reference_tape.npz',allow_pickle=False) as current, \
                         np.load(base/f'repeat_{repeat}/reference_tape.npz',allow_pickle=False) as expected:
                        differences = {}
                        for key in ('physics','magnetic_inputs','magnetic_outputs'):
                            differences[key] = float(np.max(np.abs(current[key].astype(float)-expected[key].astype(float))))
                            if differences[key] > registration['tolerances'][key]['atol']:
                                raise RuntimeError(f'original paired {key} differs from preregistered baseline')
                        for key in ('visible_packed','cumulative_packed','branch_hz','physics_tick','num_vertices','bitorder'):
                            if not np.array_equal(current[key],expected[key]):
                                raise RuntimeError(f'original paired {key} must be exact')
                    c10 = (folder/'coverage_reward_10hz.jsonl').read_bytes()
                    if c10 != (base/f'repeat_{repeat}/coverage_reward_10hz.jsonl').read_bytes():
                        raise RuntimeError('original paired C10/reward records must be exact')
                    policy = auditing.read_rows(folder/'policy_1hz.jsonl')
                    old_policy = auditing.read_rows(base/f'repeat_{repeat}/policy_1hz.jsonl')
                    for key in ('actual_history','raw_action','reward'):
                        if [r[key] for r in policy] != [r[key] for r in old_policy]:
                            raise RuntimeError(f'original paired {key} mismatch')
                    rgb = auditing.read_rows(folder/'rgb_1hz.jsonl')
                    old_rgb = auditing.read_rows(base/f'repeat_{repeat}/rgb_1hz.jsonl')
                    for key in ('coverage','rgb_frame','policy_second','sim_time_s'):
                        if [r[key] for r in rgb] != [r[key] for r in old_rgb]:
                            raise RuntimeError(f'original paired RGB metadata {key} mismatch')
                    comparisons.append(dict(repeat=repeat,max_abs_difference=differences,
                                             coverage_masks='exact',c10_rewards='exact',rgb_pixels='not_compared_to_P0_hashes'))
                report['paired_reference_comparison'] = comparisons
        except Exception as error:
            report.update(status='fail',error=dict(type=type(error).__name__,message=str(error)))
        if state['trace'] is not None:
            state['trace'].export_chrome_trace(str(output/'torch_trace.json'))
            events = [dict(key=e.key,count=e.count,cpu_time_total_us=e.cpu_time_total,
                           self_cpu_time_total_us=e.self_cpu_time_total,
                           device_time_total_us=getattr(e,'device_time_total',0.))
                      for e in state['trace'].key_averages()]
            (output/'torch_trace_events.json').write_text(json.dumps(events,indent=2,allow_nan=False)+'\n')
        report['artifacts'] = [inventory(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'summary.json']

    def persist(report,output,shutdown):
        try:
            prepare_profile(report,output)
        except Exception as error:
            # Even diagnostic export/comparison failures must be reported
            # before Kit shutdown, not turn an exit-code 0 into a pass.
            report.update(status='fail',error=dict(type=type(error).__name__,message=str(error)))
        finally:
            hooks.close()
        old_persist(report,output,shutdown)

    forwarded = forwarded_arguments(argv)
    if not any(token == '--output_root' or token.startswith('--output_root=') for token in forwarded):
        forwarded += ['--output_root',str(args.output_root)]
    try:
        gym.make, capture.persist_before_shutdown = attach, persist
        sys.argv = [str(Path(capture.__file__)),*forwarded]
        capture.main()
    finally:
        hooks.close()
        gym.make, capture.persist_before_shutdown, sys.argv = old_make, old_persist, old_argv


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('audit','run'):
        raise SystemExit('usage: profile_capacity.py {audit,run} --help')
    (audit if sys.argv[1] == 'audit' else run)(sys.argv[2:])


if __name__ == '__main__':
    main()
