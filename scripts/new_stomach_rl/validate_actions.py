#!/usr/bin/env python3
"""Gate 1 only: real 240Hz joint/magnetic execution; no learner or pose tricks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/new_stomach_rl'))
from validate_preflight import persist_before_shutdown
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=('single', 'chunk'), required=True)
parser.add_argument('--output_root', type=Path, default=ROOT / 'artifacts/new_stomach_rl/action')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_vector_env_cfg import RobotarmMagneticNewStomachVectorEnvCfg, NEW_STOMACH_COLLIDER
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.new_stomach_rl_action import NewStomachRLActionCfg

output = args.output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
output.mkdir(parents=True, exist_ok=False)
report = {'gate': 1, 'mode': args.mode, 'status': 'running', 'command': sys.argv, 'cases': []}
cfg = RobotarmMagneticNewStomachVectorEnvCfg()
cfg.sim.device = args.device
cfg.actions.magnet = NewStomachRLActionCfg(asset_name='robot', world_mesh_path=NEW_STOMACH_COLLIDER, mode=args.mode)
env = None
try:
    with launch_simulation(cfg, args):
        try:
            env = ManagerBasedRLEnv(cfg=cfg)
            term = env.action_manager.get_term('magnet')
            limits = term.robot.data.soft_joint_pos_limits.torch[0, term.ids].detach().cpu().numpy()
            report['sdk_soft_limits_rad'] = [[str(x) for x in row] for row in limits]
            report['joint_names'] = [term.robot.joint_names[i] for i in term.ids]
            bridge = env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            original_physics_step = bridge.physics_step
            magnetic_rows = []

            def traced_physics_step(base, env_ids=None):
                original_physics_step(base, env_ids)
                magnetic_rows.append(bridge.state['wrench'].detach().cpu().numpy().tolist())

            bridge.physics_step = traced_physics_step
            cases = [('hold', np.zeros((4, 9)))]
            for axis in range(9):
                for sign in (-1, 1):
                    a = np.zeros((4, 9)); a[:, axis] = sign * .25
                    cases.append((f'axis-{axis}-{sign:+d}', a))
            cases.append(('combined', np.full((4, 9), .1)))
            reverse = np.zeros((4, 9)); reverse[:, 0] = [.25, -.25, .25, -.25]
            reverse[:, 6] = [.25, -.25, .25, -.25]
            cases.append(('reversal', reverse))
            with (output / 'physics_240hz.jsonl').open('w') as stream, threadpool_limits(limits=1, user_api='blas'):
                for name, chunk in cases:
                    env.reset(seed=1008)
                    bridge.reset()
                    magnetic_rows.clear()
                    action = chunk if args.mode == 'chunk' else chunk[0]
                    obs, reward, terminated, truncated, extras = env.step(torch.as_tensor(action, dtype=torch.float32, device=env.device).reshape(1, -1))
                    result = term.boundary_result()
                    result['case'] = name
                    result['magnetic_physics_calls'] = len(magnetic_rows)
                    result['actual_physics_records'] = len(term.physics_records)
                    result['executed_history'] = term.executed_history[0].detach().cpu().numpy().tolist()
                    result['terminated'] = bool(terminated.any())
                    result['truncated'] = bool(truncated.any())
                    result['finite'] = bool(np.isfinite(np.asarray(magnetic_rows)).all() and np.isfinite(np.asarray([r['joint_actual_rad'] for r in term.physics_records])).all())
                    for row, wrench in zip(term.physics_records, magnetic_rows):
                        stream.write(json.dumps(dict(row, case=name, wrench_world=wrench), allow_nan=False) + '\n')
                    stream.flush()
                    report['cases'].append(result)
                    print('RL_ACTION_CASE ' + json.dumps(result, allow_nan=False), flush=True)
                    if (term.tick != 240 or len(magnetic_rows) != 240 or len(term.physics_records) != 240
                            or not result['finite'] or result['terminated'] or result['truncated']
                            or result['tracking_acceptance'] != 'passed'):
                        raise RuntimeError(f'Gate 1 execution/safety failed: {name}')
            report['status'] = 'pass'
            report['tracking_failures_diagnostic'] = sum(r['tracking_acceptance'] != 'passed' for r in report['cases'])
            if not any(r['projection_scale'] > 0 and r['case'] != 'hold' for r in report['cases']):
                raise RuntimeError('no moving schedule was executable')
        finally:
            if env is not None:
                env.close()
except Exception as error:
    report['status'] = 'fail'
    report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
finally:
    persist_before_shutdown(report, output, launcher.app.close)
