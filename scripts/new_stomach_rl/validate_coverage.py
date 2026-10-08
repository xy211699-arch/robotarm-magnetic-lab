"""Gate 2 runtime called by validate_preflight --gate coverage.

All ten physical-boundary poses are read from live rigid-body buffers. The
camera is accessed only at real one-second RGB boundaries, never for C10.
"""
import hashlib
import json
import traceback


def run_coverage(args, report, output):
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    launcher = AppLauncher(args)
    env = None
    try:
        import numpy as np
        import torch
        from threadpoolctl import threadpool_limits
        from isaaclab.app import launch_simulation
        from isaaclab.envs import ManagerBasedRLEnv
        from robotarm_magnetic_lab.coverage.new_stomach_rl_coverage import (
            frozen_visibility, DualRateCoverage, optical_from_capsule)
        from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw
        from robotarm_magnetic_lab.runtime.new_stomach_rl_optics import LiveMountedOptics
        from robotarm_magnetic_lab.runtime.new_stomach_rl_rgb import PolicyRGBBoundary
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_vector_env_cfg import RobotarmMagneticNewStomachVectorEnvCfg, NEW_STOMACH_COLLIDER
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.new_stomach_rl_action import NewStomachRLActionCfg
        from validate_preflight import persist_before_shutdown

        cfg = RobotarmMagneticNewStomachVectorEnvCfg()
        cfg.sim.device = args.device
        cfg.actions.magnet = NewStomachRLActionCfg(asset_name='robot', world_mesh_path=NEW_STOMACH_COLLIDER, mode='chunk')
        if cfg.scene.capsule_camera.update_period != 1. or cfg.sim.render_interval != 240:
            raise ValueError('existing RGB1Hz/240Hz physics clock differs')
        with launch_simulation(cfg, args), threadpool_limits(limits=1, user_api='blas'):
            env = ManagerBasedRLEnv(cfg=cfg)
            env.reset(seed=1008)
            term = env.action_manager.get_term('magnet')
            capsule = env.scene['capsule']
            bridge = env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            bridge.reset()
            # Reset-only library load and mandatory unbudgeted 1s HOLD, never
            # a capsule pose write during an action/coverage trajectory.
            manifest = json.loads(args.pose_manifest.read_text())
            pose_id = manifest['fixed_live_reload_pose_ids']['train'][0]
            records = [json.loads(line) for line in open(manifest['data_path'])]
            pose = next(r['pose_world_xyzw'] for r in records if r['pose_id'] == pose_id)
            capsule.permanent_wrench_composer.reset()
            capsule.write_root_pose_to_sim_index(root_pose=torch.tensor([pose], dtype=torch.float32, device=env.device))
            capsule.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6), device=env.device))
            env.sim.forward(); env.scene.update(0.); term.reset()
            _, _, terminated, truncated, _ = env.step(torch.zeros((1,36), device=env.device))
            if terminated.any() or truncated.any():
                raise RuntimeError('library reset HOLD failed')
            env.episode_length_buf.zero_(); term.reset()
            report['reset'] = {'pose_id': pose_id, 'hold_physics_ticks': 240, 'included_in_policy_budget': False}
            term._geometry()
            visibility = frozen_visibility(term._new_stomach_runtime.reference, args.mask, env.device)
            camera = env.scene['capsule_camera']

            live_optics = LiveMountedOptics(camera, env.device, env.sim.physics_manager.forward)
            runtime = DualRateCoverage(visibility, 1, geometry_optics=live_optics)

            def geometry_optical():
                return live_optics(capsule.data.root_link_pose_w.torch)

            def camera_optical():
                center = camera.data.pos_w
                center = getattr(center, 'torch', center).to(torch.float64)
                quat = getattr(camera.data.quat_w_ros, 'torch', camera.data.quat_w_ros)
                axes = np.stack([rotation_matrix_from_xyzw(q)[:, 2] for q in quat.detach().cpu().numpy()])
                return center, torch.as_tensor(axes, dtype=torch.float64, device=env.device)

            runtime.initialize(geometry_optical())
            rgb_boundary = PolicyRGBBoundary(1, env.device)
            rgb, frames = rgb_boundary.sample(0, 0, camera)
            previous_frame = int(frames[0])
            if not torch.equal(visibility(camera_optical()), runtime._boundary_visible):
                raise RuntimeError('initial C0 RGB and live optical transforms differ')
            report['coverage'] = {'rgb_period_s': camera.cfg.update_period, 'physics_hz': 240,
                'policy_hz': 1, 'geometry_hz': 10, 'device': str(env.device),
                'camera_device': str(rgb.device), 'raycast_device': visibility.device,
                'geometry_sha256': visibility.reference.geometry_sha256,
                'target_area_m2': float(visibility.weights.sum()), 'rgb_records': [{
                    'policy_second': 0, 'physics_ticks': 0, 'rgb_frame': previous_frame,
                    'rgb_sha256': hashlib.sha256(rgb.detach().cpu().numpy().tobytes()).hexdigest(),
                    'c10': runtime.c10_records[0]['coverage'], 'c1': runtime.c1_records[0]['coverage']}]}
            elapsed_ticks = 0
            term.geometry_tick_callback = lambda tick: runtime.update_geometry_at_physics_tick(elapsed_ticks + tick, capsule.data.root_link_pose_w.torch)
            with (output / 'coverage_geometry_10hz.jsonl').open('w') as c10file, (output / 'coverage_rgb_1hz.jsonl').open('w') as c1file:
                c10file.write(json.dumps(runtime.c10_records[0])+'\n'); c1file.write(json.dumps(runtime.c1_records[0])+'\n')
                for second in range(1, args.seconds + 1):
                    a = torch.zeros((1,36), device=env.device)
                    a.reshape(1,4,9)[:, :, (second-1)%9] = .1 * (-1 if second%2 else 1)
                    before = len(runtime.c10_records)
                    obs, reward, terminated, truncated, extras = env.step(a)
                    if terminated.any() or truncated.any():
                        raise RuntimeError('coverage validation encountered a safety termination')
                    elapsed_ticks += 240
                    runtime.update_geometry_at_physics_tick(elapsed_ticks, capsule.data.root_link_pose_w.torch)
                    rgb, frames = rgb_boundary.sample(second, elapsed_ticks, camera)
                    current_frame = int(frames[0])
                    if current_frame != previous_frame + 1 or not torch.isfinite(rgb).all():
                        raise RuntimeError('RGB boundary did not produce exactly one finite new frame')
                    latest_optics = live_optics.records[-1]
                    geometric_optical = (torch.tensor(latest_optics['optical_center_world_m'], device=env.device),
                        torch.tensor(latest_optics['optical_axis_world'], device=env.device))
                    actual_optical = camera_optical()
                    camera_visible = visibility(actual_optical)
                    difference = camera_visible != runtime._boundary_visible
                    diagnostic = {'second': second, 'geometry_center': geometric_optical[0].cpu().tolist(),
                        'rgb_center': actual_optical[0].cpu().tolist(),
                        'geometry_axis': geometric_optical[1].cpu().tolist(), 'rgb_axis': actual_optical[1].cpu().tolist(),
                        'different_vertices': int(difference.sum()),
                        'geometry_visible': int(runtime._boundary_visible.sum()), 'rgb_visible': int(camera_visible.sum())}
                    report['coverage'].setdefault('optics_diagnostics', []).append(diagnostic)
                    if difference.any():
                        np.savez(output / 'boundary_visibility_failure.npz',
                            capsule_pose=capsule.data.root_link_pose_w.torch.cpu().numpy(),
                            geometric_center=geometric_optical[0].cpu().numpy(), actual_center=actual_optical[0].cpu().numpy(),
                            geometric_axis=geometric_optical[1].cpu().numpy(), actual_axis=actual_optical[1].cpu().numpy(),
                            geometry_mask=runtime._boundary_visible.cpu().numpy(), actual_mask=camera_visible.cpu().numpy())
                    runtime.observe_camera_frame(second, second, actual_optical)
                    row = {'policy_second': second, 'physics_ticks': elapsed_ticks,
                        'rgb_frame': current_frame, 'rgb_sha256': hashlib.sha256(rgb.detach().cpu().numpy().tobytes()).hexdigest(),
                        'forced_rgb_capture': rgb_boundary.sync.last_forced_capture,
                        'c10': runtime.c10_records[-1]['coverage'], 'c1': runtime.c1_records[-1]['coverage']}
                    report['coverage']['rgb_records'].append(row)
                    for record in runtime.c10_records[before:]:
                        optics = live_optics.records[record['sample_id']]
                        c10file.write(json.dumps(dict(record, **optics))+'\n')
                    c1file.write(json.dumps(runtime.c1_records[-1])+'\n')
                    c10file.flush(); c1file.flush()
                    print('RL_COVERAGE_BOUNDARY '+json.dumps(row), flush=True)
                    previous_frame = current_frame
            if len(runtime.c10_records) != args.seconds*10+1 or len(runtime.c1_records) != args.seconds+1:
                raise RuntimeError('coverage sample counts differ')
            report['coverage'].update(c10_points=len(runtime.c10_records), c1_points=len(runtime.c1_records))
            report['status'] = 'pass'
    except Exception as error:
        report['status'] = 'fail'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    finally:
        if env is not None:
            env.close()
        from validate_preflight import persist_before_shutdown
        persist_before_shutdown(report, output, launcher.app.close)
