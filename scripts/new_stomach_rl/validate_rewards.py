"""Gate 3: independent RewardsCfg, live 10Hz old four terms and 1Hz sums."""
import json
import traceback


def run_rewards(args, report, output):
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
        from robotarm_magnetic_lab.runtime.new_stomach_rl_runtime import NewStomachRLRuntime
        from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_rl_env_cfg import RobotarmMagneticNewStomachRLPreflightCfg
        cfg = RobotarmMagneticNewStomachRLPreflightCfg()
        cfg.sim.device = args.device
        cfg.seed = 1008
        # Validator owns the 120-second stop so terminal data is not replaced
        # by Same-Step autoreset. Does not disable collision/stopping guards.
        cfg.episode_length_s = args.seconds + 1.
        with launch_simulation(cfg, args), threadpool_limits(limits=1, user_api='blas'):
            env = ManagerBasedRLEnv(cfg=cfg)
            env.reset(seed=1008)
            term = env.action_manager.get_term('magnet')
            bridge = env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            bridge.reset()
            manifest = json.loads(args.pose_manifest.read_text())
            pose_id = manifest['fixed_live_reload_pose_ids'][args.pose_split][0]
            records = [json.loads(line) for line in open(manifest['data_path'])]
            pose = next(r['pose_world_xyzw'] for r in records if r['pose_id'] == pose_id)
            capsule = env.scene['capsule']
            capsule.permanent_wrench_composer.reset()
            capsule.write_root_pose_to_sim_index(root_pose=torch.tensor([pose], dtype=torch.float32, device=env.device))
            capsule.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6), device=env.device))
            env.sim.forward(); env.scene.update(0.); term.reset()
            _, reward, terminated, truncated, _ = env.step(torch.zeros((1,36), device=env.device))
            if terminated.any() or truncated.any() or reward.abs().max() != 0:
                raise RuntimeError('reset HOLD failed or generated policy reward')
            env.episode_length_buf.zero_(); term.reset()
            runtime = NewStomachRLRuntime(env, args.mask)
            runtime.initialize()
            if env.reward_manager.active_terms != ['four_terms']:
                raise RuntimeError('unexpected inherited reward terms')
            rng = np.random.default_rng(1008)
            report['reward_validation'] = {'pose_id': pose_id, 'split': args.pose_split,
                'policy_hz': 1, 'reward_hz': 10, 'physics_hz': 240,
                'weights': [100., .1, -.002, .2], 'window_s': 5.,
                'active_reward_terms': env.reward_manager.active_terms,
                'reset_hold_steps': 240, 'reset_hold_reward': 0., 'c0': runtime.coverage.c10_records[0]['coverage'],
                'seconds': [], 'policy': 'seeded random' if args.pose_split == 'validation' else 'axis script'}
            with (output / 'reward_10hz.jsonl').open('w') as ten, (output / 'reward_1hz.jsonl').open('w') as one, (output / 'physics_240hz.jsonl').open('w') as physics:
                ten.write(json.dumps(runtime.ten_hz_records[0])+'\n')
                one.write(json.dumps(runtime.one_hz_records[0])+'\n')
                for second in range(1, args.seconds+1):
                    action = np.zeros((4,9))
                    if args.pose_split == 'validation':
                        action = rng.uniform(-.25, .25, (4,9))
                    else:
                        action[:,(second-1)%9] = .1 * (-1 if second%2 else 1)
                    before = len(runtime.ten_hz_records)
                    observations, actual_reward, terminated, truncated, extras = env.step(
                        torch.tensor(action.reshape(1,36), dtype=torch.float32, device=env.device))
                    if terminated.any() or truncated.any():
                        raise RuntimeError('reward validation safety termination')
                    rows = runtime.ten_hz_records[before:]
                    expected = sum(row['total_reward'] for row in rows)
                    parts = np.asarray([row['reward_terms'] for row in rows]).sum(axis=0)
                    actual = float(actual_reward[0])
                    if len(rows) != 10 or not np.isfinite([expected, actual, *parts]).all() or abs(actual-expected) > 1e-5 or abs(parts.sum()-actual) > 1e-5:
                        raise RuntimeError('four-term/10Hz/1Hz reward sum mismatch')
                    if observations['policy'].shape != (1,548) or observations['critic'].shape != (1,28):
                        raise RuntimeError('observation shape differs')
                    for row in rows:
                        ten.write(json.dumps(row, allow_nan=False)+'\n')
                    one.write(json.dumps(runtime.one_hz_records[-1], allow_nan=False)+'\n')
                    for row in term.physics_records:
                        physics.write(json.dumps(dict(row, policy_second=second), allow_nan=False)+'\n')
                    ten.flush(); one.flush(); physics.flush()
                    record = {'policy_second': second, 'reward': actual, 'terms': parts.tolist(),
                        'c10': runtime.coverage.c10_records[-1]['coverage'], 'c1': runtime.coverage.c1_records[-1]['coverage'],
                        'projection_scale': term.telemetry['projection_scale']}
                    report['reward_validation']['seconds'].append(record)
                    if second%10 == 0 or second == args.seconds:
                        print('RL_REWARD_SECOND '+json.dumps(record), flush=True)
            if len(runtime.ten_hz_records) != args.seconds*10+1 or len(runtime.one_hz_records) != args.seconds+1:
                raise RuntimeError('reward/coverage record count differs')
            report['reward_validation'].update(c10_points=len(runtime.ten_hz_records), c1_points=len(runtime.one_hz_records),
                cumulative_terms=np.asarray([s['terms'] for s in report['reward_validation']['seconds']]).sum(axis=0).tolist())
            report['status'] = 'pass'
    except Exception as error:
        report['status'] = 'fail'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    finally:
        if env is not None:
            env.close()
        from validate_preflight import persist_before_shutdown
        persist_before_shutdown(report, output, launcher.app.close)
