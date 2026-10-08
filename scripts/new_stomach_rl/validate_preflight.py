#!/usr/bin/env python3
"""Evidence-producing, fail-closed preflight. Importing this module starts no Kit."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'source/robotarm_magnetic_lab/robotarm_magnetic_lab'
RUNTIME_MESH_SHA = 'c4028caa5da750d769a6c5e39709bd7caac975339f35eac5d36fc0e48036c72e'
DEFAULT_MANIFEST = ROOT.parent / 'new-stomach-coverage-regenerate/configs/new_stomach_v1/entry/pose_library_manifest_v1.json'
DEFAULT_MASK = Path('/mnt/isaac-linux/robotarm_magnetic_lab/artifacts/new_stomach_coverage/tube_candidate_full_right_appendage_v1/candidate_face_mask.npz')
MASK_SHA = '7261eeec9deeefcd960bb245aae0f495a459f9e3a221abd460017783fdd4d58c'
COUNTS = {'train': 1000, 'validation': 100, 'test': 100}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def audit_pose_manifest(path):
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    if m.get('stomach_geometry_sha256') != RUNTIME_MESH_SHA:
        raise ValueError('pose manifest does not bind the accepted new-stomach geometry')
    if m.get('stomach_model') != 'new_v1' or m.get('split_counts') != COUNTS:
        raise ValueError('wrong stomach model or split counts')
    payload = {k: v for k, v in m.items() if k != 'config_sha256'}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != m['config_sha256']:
        raise ValueError('manifest content hash mismatch')
    data = Path(m['data_path']).resolve()
    if sha(data) != m['data_sha256'] or data.stat().st_size != m['data_bytes']:
        raise ValueError('pose library file hash/size mismatch')
    records = [json.loads(line) for line in data.read_text().splitlines()]
    if Counter(r['split'] for r in records) != COUNTS:
        raise ValueError('pose record split counts mismatch')
    if len({r['pose_id'] for r in records}) != 1200 or len({r['pose_fingerprint_sha256'] for r in records}) != 1200:
        raise ValueError('duplicate pose IDs or final poses')
    for r in records:
        pose = np.asarray(r['pose_world_xyzw'])
        if (r['stomach_geometry_sha256'] != RUNTIME_MESH_SHA or pose.shape != (7,)
                or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 1e-5
                or not r['stable'] or not r['camera_inside_lumen'] or r['unoriented_axis_angle_deg'] < 45
                or r['stable_duration_s'] < .25 or r['settle_elapsed_s'] > 2
                or r['final_linear_speed_m_s'] > .002 or r['final_angular_speed_rad_s'] > np.deg2rad(5)):
            raise ValueError(f'invalid stored pose: {r["pose_id"]}')
    for split, count in COUNTS.items():
        ids = {r['pose_id'] for r in records if r['split'] == split}
        if ids != set(m['split_pose_ids'][split]):
            raise ValueError('split ID membership mismatch')
        fixed = m['fixed_live_reload_pose_ids'][split]
        if len(fixed) != 20 or len(set(fixed)) != 20 or not set(fixed) <= ids:
            raise ValueError('fixed reload sample mismatch')
    return {'manifest': inventory(path), 'data': inventory(data), 'split_counts': COUNTS,
            'unique_poses': 1200, 'stomach_geometry_sha256': RUNTIME_MESH_SHA,
            'stored_candidate_stability': 'pass', 'live_gpu_reload': 'not_run'}


def audit_mask(path, root=ROOT):
    path = Path(path)
    if sha(path) != MASK_SHA:
        raise ValueError('candidate mask file SHA mismatch')
    with np.load(path, allow_pickle=False) as archive:
        labels = archive['face_labels']
        excluded = archive['excluded_face_indices']
        target = archive['reachable_face_indices']
        if labels.shape != (429029,) or not np.array_equal(excluded, np.flatnonzero(labels != 0)) or not np.array_equal(target, np.flatnonzero(labels == 0)):
            raise ValueError('mask membership inconsistent')
        if len(excluded) != 176109 or len(target) != 252920:
            raise ValueError('mask count mismatch')
    summary = json.loads(path.with_name('candidate_summary.json').read_text())
    if abs(summary['reachable_area_m2'] - .04020530122973115) > 1e-10:
        raise ValueError('mask reachable area mismatch')
    orientation = Path(root) / 'configs/new_stomach_v1/orientation_v1.json'
    if sha(orientation) != summary['orientation_config_sha256']:
        raise ValueError('mask orientation identity mismatch')
    return {'file': inventory(path), 'summary': inventory(path.with_name('candidate_summary.json')),
            'total_faces': 429029, 'excluded_faces': 176109, 'target_faces': 252920,
            'target_area_m2': summary['reachable_area_m2'],
            'offline_mesh_sha': summary['offline_reference_geometry_sha256'],
            'runtime_mask_equivalence': 'not_run', 'runtime_mesh_sha_required': RUNTIME_MESH_SHA}


def audit_release_semantics(root=ROOT):
    base = Path(root) / 'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab'
    path = base / 'controllers/actuator_vector.py'
    spec = importlib.util.spec_from_file_location('archived_actuator_vector', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q, velocity, acceleration = module.quintic(np.arange(45).reshape(5, 9) / 1000)
    if q.shape != (961, 9) or not np.isfinite(q).all():
        raise ValueError('archived 4-second preview planner semantics changed')
    tree = ast.parse((base / 'mdp/actuator_vector_action.py').read_text())
    statements = [ast.unparse(n) for n in ast.walk(tree)]
    if "self.execution = result['q'][:241].copy()" not in statements:
        raise ValueError('archived first-second commit changed')
    guard = 'env.num_envs != 1' in statements
    if not guard or "required_clearance_m: float = 0.005" not in statements:
        raise ValueError('archived safety or single-env guard changed')
    return {'preview_seconds': 4, 'committed_ticks': 240, 'single_env_guard': guard,
            'required_clearance_m': .005, 'source': inventory(path),
            'action_source': inventory(base / 'mdp/actuator_vector_action.py')}


def audit_assets(manifest=DEFAULT_MANIFEST, mask=DEFAULT_MASK, root=ROOT):
    root = Path(root)
    orientation = json.loads((root / 'configs/new_stomach_v1/orientation_v1.json').read_text())
    asset = root / 'assets/stomach/new_stomach_v1/new_stomach_physics.usda'
    if sha(asset) != orientation['asset_usd_sha256']:
        raise ValueError('new stomach USD differs from confirmed asset')
    snapshot = json.loads((root / 'dependencies/legacy_magnetic_sim/snapshot_manifest.json').read_text())
    dependencies = []
    for record in snapshot['files']:
        item = inventory(record['original_path'])
        if item['sha256'] != record.get('original_sha256', record['sha256']):
            raise ValueError(f'external dependency changed: {item["path"]}')
        dependencies.append(item)
    stage = inventory('/home/multirobo/Desktop/sim of FF/Stage.usd')
    if stage['sha256'] != '0d5afec84d42989a7910899b4ef78369c454e34185c89052ab25b33fac4a6671':
        raise ValueError('external robot Stage changed')
    return {'asset': inventory(asset), 'orientation': inventory(root / 'configs/new_stomach_v1/orientation_v1.json'),
            'poses': audit_pose_manifest(manifest), 'mask': audit_mask(mask, root),
            'release_semantics': audit_release_semantics(root), 'external_dependencies': dependencies,
            'robot_stage': stage}


def persist_before_shutdown(report, output, shutdown):
    """Kit may terminate Python on close: evidence must precede that call."""
    (output / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('NEW_STOMACH_PREFLIGHT ' + json.dumps({'status': report['status'], 'output': str(output),
                                              'error': report.get('error')}), flush=True)
    shutdown()


def signed_lumen_distance(center, vertices, faces, *, block_size=32768):
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ideal_surface.surface_mesh import _closest_points_on_triangles

    nearest = None
    for start in range(0, len(faces), block_size):
        triangles = vertices[faces[start:start + block_size]]
        points, _ = _closest_points_on_triangles(center, triangles)
        squared = np.sum((points - center) ** 2, axis=1)
        index = int(np.argmin(squared))
        if nearest is None or squared[index] < nearest[0]:
            normal = np.cross(triangles[index, 1] - triangles[index, 0], triangles[index, 2] - triangles[index, 0])
            normal /= np.linalg.norm(normal)
            nearest = (float(squared[index]), points[index], normal)
    if nearest is None:
        raise ValueError('empty lumen mesh')
    return float(np.dot(center - nearest[1], nearest[2]))


def run_runtime(args, report, output):
    # Only this explicit runtime branch starts Kit. Pure audits/tests never do.
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    launcher = AppLauncher(args)
    import omni.usd
    import torch
    import isaaclab
    from pxr import UsdUtils, UsdPhysics, PhysxSchema
    from isaaclab.app import launch_simulation
    from isaaclab.envs import ManagerBasedRLEnv
    from threadpoolctl import threadpool_limits
    from robotarm_magnetic_lab.geometry.new_stomach_runtime import NewStomachRuntimeGeometry
    from robotarm_magnetic_lab.coverage.new_stomach_tubes import candidate_tube_mask, candidate_right_tube_mask, infer_inward_direction
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_env_cfg import NEW_STOMACH_GEOMETRY
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_vector_env_cfg import RobotarmMagneticNewStomachVectorEnvCfg
    from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw
    cfg = RobotarmMagneticNewStomachVectorEnvCfg()
    cfg.sim.device = args.device
    cfg.scene.capsule_camera_preview = None
    env = None
    try:
        with launch_simulation(cfg, args):
            try:
                env = ManagerBasedRLEnv(cfg=cfg)
                env.reset(seed=1008)
                base = env.unwrapped
                report['devices'] = {'env': str(base.device), 'simulation': str(base.sim.device),
                    'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': sys.version,
                    'platform': platform.platform(), 'isaaclab': getattr(isaaclab, '__version__', 'see startup log'),
                    'camera_update_period_s': cfg.scene.capsule_camera.update_period}
                physics_scenes = [p for p in omni.usd.get_context().get_stage().Traverse() if p.IsA(UsdPhysics.Scene)]
                report['devices']['gpu_dynamics_scenes'] = [{
                    'path': str(p.GetPath()),
                    'enabled': PhysxSchema.PhysxSceneAPI(p).GetEnableGPUDynamicsAttr().Get(),
                    'broadphase': PhysxSchema.PhysxSceneAPI(p).GetBroadphaseTypeAttr().Get(),
                } for p in physics_scenes]
                if not str(base.sim.device).startswith('cuda'):
                    raise RuntimeError('Gate 0 runtime requires actual GPU PhysX')
                runtime = NewStomachRuntimeGeometry.from_stage(omni.usd.get_context().get_stage(), NEW_STOMACH_GEOMETRY)
                reference = runtime.reference
                if reference.geometry_sha256 != RUNTIME_MESH_SHA:
                    raise RuntimeError('live mesh hash differs from accepted pose library')
                centers = np.asarray(NEW_STOMACH_GEOMETRY.opening_centers_world_m)
                axes = np.asarray(NEW_STOMACH_GEOMETRY.opening_axes_world)
                face_centers = reference.vertices_world[reference.triangles].mean(axis=1)
                inward = np.array([infer_inward_direction(face_centers, c, a)[0] for c, a in zip(centers, axes)])
                tubes = candidate_tube_mask(reference, centers, inward, tube_length_m=.025, radial_limit_m=.012, mouth_buffer_m=.001)
                right = candidate_right_tube_mask(reference, cut_x_from_min_m=.130, max_y_from_min_m=.110)
                labels = tubes.labels.copy()
                right_mask = np.zeros(len(labels), dtype=bool)
                right_mask[right.face_indices] = True
                labels[right_mask & (labels == 0)] = 3
                with np.load(args.mask, allow_pickle=False) as source:
                    difference = int(np.count_nonzero(labels != source['face_labels']))
                if difference:
                    raise RuntimeError(f'live/offline face mask differs on {difference} faces')
                report['runtime_geometry'] = {'mesh_sha256': reference.geometry_sha256,
                    'mask_different_faces': difference, 'consumer_hashes': runtime.consumer_hashes()}
                layers, assets, unresolved = UsdUtils.ComputeAllDependencies('/home/multirobo/Desktop/sim of FF/Stage.usd')
                report['usd_dependencies'] = {'layer_count': len(layers), 'unresolved': [str(x) for x in unresolved]}
                if unresolved:
                    raise RuntimeError('unresolved external USD dependencies')
                capsule = base.scene['capsule']
                term = base.action_manager.get_term('magnet')
                manifest = json.loads(Path(args.pose_manifest).read_text())
                records = {r['pose_id']: r for r in map(json.loads, Path(manifest['data_path']).read_text().splitlines())}
                chosen = [pose_id for split in COUNTS for pose_id in manifest['fixed_live_reload_pose_ids'][split]]
                rows = []
                report['runtime_reload'] = {'status': 'running', 'requested': len(chosen), 'rows': rows}
                with (output / 'runtime_reload.jsonl').open('w') as stream, threadpool_limits(limits=1, user_api='blas'):
                    for pose_id in chosen:
                        base.reset(seed=1008)
                        bridge = base.event_manager.get_term_cfg('magnetic_collision_bridge').func
                        bridge.reset()
                        pose = np.asarray(records[pose_id]['pose_world_xyzw'])
                        capsule.permanent_wrench_composer.reset()
                        capsule.write_root_pose_to_sim_index(root_pose=torch.as_tensor(pose, dtype=torch.float32, device=base.device)[None])
                        capsule.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))
                        base.sim.forward()
                        base.scene.update(0.)
                        term.reset()
                        obs, reward, terminated, truncated, extras = base.step(torch.zeros((1, 36), device=base.device))
                        final = capsule.data.root_pose_w.torch[0].detach().cpu().numpy()
                        velocity = capsule.data.root_com_vel_w.torch[0].detach().cpu().numpy()
                        rgb = base.scene['capsule_camera'].data.output['rgb']
                        rgb = getattr(rgb, 'torch', rgb)
                        angle = float(np.degrees(np.arccos(np.clip(abs(rotation_matrix_from_xyzw(final[3:])[2, 2]), 0, 1))))
                        link_position = capsule.data.root_link_pos_w.torch[0].detach().cpu().numpy()
                        link_quat = capsule.data.root_link_quat_w.torch[0].detach().cpu().numpy()
                        camera_center = link_position + rotation_matrix_from_xyzw(link_quat) @ np.asarray(base.scene['capsule_camera'].cfg.offset.pos)
                        camera_signed_distance = signed_lumen_distance(camera_center, reference.vertices_world, reference.triangles)
                        row = {'pose_id': pose_id, 'initial_pose_xyzw': pose.tolist(), 'final_pose_xyzw': final.tolist(),
                            'velocity_world': velocity.tolist(), 'axis_angle_deg': angle,
                            'rgb_device': str(rgb.device), 'finite': bool(np.isfinite(final).all() and np.isfinite(velocity).all() and torch.isfinite(rgb).all()),
                            'terminated': bool(terminated.any()), 'truncated': bool(truncated.any()),
                            'hold_steps': term.tick, 'clearance': getattr(base, '_new_stomach_clearance', None),
                            'full_env_and_magnetic_filter_reset': True,
                            'camera_signed_lumen_distance_m': camera_signed_distance}
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        stream.flush()
                        print('PREFLIGHT_GPU_POSE ' + json.dumps(row), flush=True)
                        if not row['finite'] or row['terminated'] or row['truncated'] or angle < 45 or term.tick != 240 or camera_signed_distance <= 0:
                            raise RuntimeError(f'GPU magnetic reset validity failed for {pose_id}')
                report['runtime_reload']['status'] = 'pass'
            finally:
                if env is not None:
                    env.close()
    except Exception as error:
        report['status'] = 'fail'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        raise
    else:
        report['status'] = 'pass'
    finally:
        persist_before_shutdown(report, output, launcher.app.close)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', choices=['assets'], default='assets')
    parser.add_argument('--pose_manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--mask', type=Path, default=DEFAULT_MASK)
    parser.add_argument('--runtime', action='store_true')
    parser.add_argument('--output_root', type=Path, default=ROOT / 'artifacts/new_stomach_rl/preflight')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = args.output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True, exist_ok=False)
    report = {'gate': 0, 'status': 'running', 'command': sys.argv,
              'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
              'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    code = 0
    try:
        report['assets'] = audit_assets(args.pose_manifest, args.mask)
        if args.runtime:
            run_runtime(args, report, output)
        report['status'] = 'pass' if args.runtime else 'assets_pass_runtime_not_run'
    except Exception as error:
        report['status'] = 'fail'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        code = 1
    (output / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('NEW_STOMACH_PREFLIGHT ' + json.dumps({'status': report['status'], 'output': str(output), 'error': report.get('error')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
