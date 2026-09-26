"""五套随机策略同一初态筛查：1Hz动作、10Hz覆盖和双相机录像。"""
import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
import signal
from pathlib import Path
from types import MethodType

ROOT=Path(__file__).resolve().parents[2]
def _terminated(signum,frame):
    raise RuntimeError(f'experiment interrupted by signal {signum}')

signal.signal(signal.SIGTERM,_terminated)
sys.path.insert(0,str(ROOT/'source/robotarm_magnetic_lab'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--policy',choices=['MR1','MR2','MR3','MR4','MR5'],required=True)
parser.add_argument('--seconds',type=int,default=300)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--stimulus',choices=['random','hold','pull_x','pull_minus_x','field_sweep','surface_roll'],default='random')
parser.add_argument('--roll_direction_sign',type=int,choices=(-1,1),default=1)
parser.add_argument('--resume_state',type=Path)
parser.add_argument('--escape_force_mn','--rolling_force_mn',dest='escape_force_mn',type=float,choices=(10.,15.,20.,30.))
parser.add_argument('--motion_pattern',choices=('straight','area_scan'),default='straight')
parser.add_argument('--clearance_mm',type=float,choices=(5.,4.,3.),default=5.)
parser.add_argument('--source_height_mm',type=float,choices=(120.,110.,100.),default=120.)
parser.add_argument('--stomach-model',choices=('legacy','new_v1'),default='legacy')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if not 1<=args.seconds<=300: parser.error('seconds must be 1..300')
if (args.escape_force_mn is not None or args.resume_state) and args.stimulus!='surface_roll':
    parser.error('escape/resume require surface_roll')
if args.motion_pattern=='area_scan' and (args.resume_state or args.stimulus!='surface_roll' or args.escape_force_mn is None):
    parser.error('area_scan requires original initialization, surface_roll and rolling_force_mn; no resume_state')
if (args.clearance_mm!=5. or args.source_height_mm!=120.) and (args.escape_force_mn is None or not (args.resume_state or args.motion_pattern=='area_scan')):
    parser.error('relaxed geometry requires an escape experiment and frozen resume state')
args.enable_cameras=True
launcher=AppLauncher(args)
import cv2
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.configclass import configclass
from robotarm_magnetic_lab.baselines.magnetic_vector_random import sequence
from robotarm_magnetic_lab.coverage.simulator_runtime import P0CoverageRuntime
from robotarm_magnetic_lab.ui.magnetic_observer import configure_observers,DualObserver
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_stomach_env_cfg import RobotarmMagneticStomachLabEnvCfg
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_new_stomach_env_cfg import (
    NEW_STOMACH_ASSET_USD_PATH,NEW_STOMACH_GEOMETRY,RobotarmMagneticNewStomachLabEnvCfg)
from robotarm_magnetic_lab.geometry.new_stomach_runtime import (
    NewStomachRuntimeGeometry,validate_model_specific_inputs)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.actuator_vector_action import ActuatorVectorActionCfg
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.magnetic_action import MagneticPhysicsActionCfg
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.ball_envelope import stomach_collision_mesh
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_parameterized_force_stomach_env_cfg import (
    TASK009B_STOMACH_FLIP_POS,TASK009B_STOMACH_FLIP_ROT_XYZW,
    TASK009B_INTEGRATION_CAPSULE_POS,TASK009B_INTEGRATION_CAPSULE_ROT_XYZW)

@configclass
class Actions:
    magnet=ActuatorVectorActionCfg(asset_name='robot')
    magnetic_physics=MagneticPhysicsActionCfg(asset_name='capsule')

class ScreenCoverage(P0CoverageRuntime):
    @property
    def sim_time_s(self): return getattr(self,'screen_time_s',0.)
    @property
    def total_sim_time_s(self): return self.sim_time_s

class DistanceExceeded(RuntimeError): pass

def audited_collision(env):
    """Use the planner's audited meshes for current-state termination as well."""
    from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.batched_arm_clearance import minimum_clearance_fast,minimum_world_clearance
    term=env.action_manager.get_term('magnet');term._geometry()
    q=term.robot.data.joint_pos.torch[0,term.ids[:6]].detach().cpu().numpy()[None]
    own=float(minimum_clearance_fast(term.kinematics,q)[0])
    world=float(minimum_world_clearance(term.world_checker,q)[0])
    gap=min(own,world)
    env._screen_clearance=dict(self_m=own,world_m=world,
        legacy_collision=bool(env._legacy_bridge_state['collision'][0,0]),
        required_m=term.cfg.required_clearance_m)
    return torch.tensor([not np.isfinite([own,world]).all() or gap<term.cfg.required_clearance_m],device=env.device)

def fresh(camera):
    # Isaac Lab 3 sensor-private buffer synchronization; no extra physics.
    wp.to_torch(camera._is_outdated).fill_(True)
    camera._update_outdated_buffers(force_recompute=True)
    return int(camera.frame.torch[0])

def main():
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    policy_seed=91500+int(args.policy[-1])
    actions=sequence(args.policy,policy_seed,args.seconds+4)
    if args.stimulus!='random':
        actions[:]=0
        if args.stimulus=='pull_x': actions[:,0]=1.
        if args.stimulus=='pull_minus_x': actions[:,0]=-1.
    if args.stimulus not in ('field_sweep','surface_roll'): np.save(output/'requested_sequence.npy',actions)
    elif args.stimulus=='field_sweep': (output/'field_probe_spec.json').write_text(json.dumps(dict(field_step_deg=2.,anchor='initial_magnet_center',
        command_log='boundaries.jsonl/raw_chunk',source_feedback_only_after_initialization=True),indent=2))
    summary=dict(status='running',policy=args.policy,seed=policy_seed,environment_seed=914,
        requested_seconds=args.seconds,completed_cycles=0,frames=0,tracking_failures=0,
        projected_cycles=0,training_started=False,stomach_model=args.stomach_model,
        initialization='frozen_flipped_stomach_quarter_pose_plus_1s_zero_increment')
    summary['stimulus']=args.stimulus
    summary['motion_pattern']=args.motion_pattern
    legacy_unreachable=ROOT/'configs/task009b/unreachable_region_v1.json' if args.stomach_model=='legacy' else None
    validate_model_specific_inputs(
        args.stomach_model,unreachable_region_path=legacy_unreachable,
        legacy_initial_pose=args.stomach_model=='legacy')
    inputs=[Path(__file__).resolve()]
    if legacy_unreachable is not None:inputs.append(legacy_unreachable)
    else:inputs.extend([ROOT/'configs/new_stomach_v1/orientation_v1.json',Path(NEW_STOMACH_ASSET_USD_PATH)])
    inputs+=list((ROOT/'source/robotarm_magnetic_lab/robotarm_magnetic_lab').glob('**/magnetic_vector_random.py'))
    inputs+=list((ROOT/'source/robotarm_magnetic_lab/robotarm_magnetic_lab').glob('**/magnetic_observer.py'))
    if args.stimulus=='field_sweep': inputs+=list((ROOT/'source/robotarm_magnetic_lab/robotarm_magnetic_lab').glob('**/magnetic_field_probe.py'))
    if args.stimulus=='surface_roll': inputs+=list((ROOT/'source/robotarm_magnetic_lab/robotarm_magnetic_lab').glob('**/magnetic_surface_roll.py'))
    if args.escape_force_mn is not None: inputs+=list((ROOT/'source/robotarm_magnetic_lab/robotarm_magnetic_lab').glob('**/magnetic_roll_escape.py'))
    for name in ('actuator_vector_action.py','actuator_vector.py','mesh_cover_audit.py','native_mesh_clearance.py','batched_arm_clearance.py','legacy_bridge.py','ball_envelope.py'):
        inputs+=list((ROOT/'source/robotarm_magnetic_lab/robotarm_magnetic_lab').glob('**/'+name))
    summary['input_files']=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs]
    snapshot_dir=output/'source_snapshot';snapshot_dir.mkdir()
    for p in inputs:
        if p.suffix=='.py': (snapshot_dir/p.name).write_bytes(p.read_bytes())
    cfg=RobotarmMagneticStomachLabEnvCfg() if args.stomach_model=='legacy' else RobotarmMagneticNewStomachLabEnvCfg()
    cfg.actions=Actions();cfg.decimation=240
    if args.stomach_model=='legacy':
        cfg.scene.stomach.init_state.pos=TASK009B_STOMACH_FLIP_POS
        cfg.scene.stomach.init_state.rot=TASK009B_STOMACH_FLIP_ROT_XYZW
        cfg.scene.capsule.init_state.pos=TASK009B_INTEGRATION_CAPSULE_POS
        cfg.scene.capsule.init_state.rot=TASK009B_INTEGRATION_CAPSULE_ROT_XYZW
    cfg.sim.dt=1/240;cfg.sim.device=args.device;cfg.sim.render_interval=1000000
    cfg.observations.vision=None;cfg.episode_length_s=args.seconds+20
    cfg.scene.capsule_camera.update_period=.1
    cfg.scene.capsule_camera_preview=None
    if args.motion_pattern=='area_scan':cfg.terminations.collision.func=audited_collision
    configure_observers(cfg)
    env=None;runtime=None;observer=None;old_update=None;handles=[]
    start=time.monotonic();physics_steps=0;last_frame=None;bad_distance=0;distances=[]
    try:
        with launch_simulation(cfg,args):
            import omni.usd
            env=ManagerBasedRLEnv(cfg=cfg)
            term=env.action_manager.get_term('magnet')
            stage=omni.usd.get_context().get_stage()
            runtime_geometry=None
            if args.stomach_model=='new_v1':
                runtime_geometry=NewStomachRuntimeGeometry.from_stage(stage,NEW_STOMACH_GEOMETRY)
                collision_path=stomach_collision_mesh(stage,configured_path=runtime_geometry.collision_mesh_path)
                summary['geometry_consumers']=runtime_geometry.consumer_hashes()
                summary['orientation_config_sha256']=runtime_geometry.orientation_config_sha256
                summary['asset_usd_sha256']=runtime_geometry.asset_usd_sha256
                summary['initialization']='new_stomach_unconfirmed_entry_pending_task5'
            else:collision_path=stomach_collision_mesh(stage)
            term.cfg.world_mesh_path=collision_path
            env.reset(seed=914)
            with threadpool_limits(limits=1,user_api='blas'):
                env.step(torch.zeros((1,36),device=env.device))
            narrow=term.kinematics._native_narrow
            (output/'runtime_mesh_audit.json').write_text(json.dumps(dict(
                native_cover=term.mesh_cover_audit,closed_native_meshes=narrow.records,
                tight_asm_cover=narrow.asm_certificate,tight_asm_ball_spheres=len(narrow.radii)),indent=2))
            runtime=ScreenCoverage(env,output/'coverage',task_id='magnetic_vector_screen',seed=914,
                commit='df2fd70b6de682c7763dc3c8fa7a499675aaac3a',branch='feature/magnetic-position-following',
                print_updates=False,surface_prim_path=collision_path,
                unreachable_region_path=legacy_unreachable)
            if runtime_geometry is not None and runtime.reference.geometry_sha256!=runtime_geometry.geometry_sha256:
                raise RuntimeError('new stomach coverage geometry disagrees with audited runtime mesh')
            observer=DualObserver(env,output/'video',runtime.raycaster)
            camera=env.scene['capsule_camera'];capsule=env.scene['capsule']
            bridge=env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            if args.resume_state or args.clearance_mm<5.:
                from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.mesh_cover_audit import audit_stage_static_meshes
                term._geometry()
                audit=audit_stage_static_meshes(omni.usd.get_context().get_stage(),term.kinematics,bridge.config['planning']['asm_mount_l6_from_asm'])
                (output/'mesh_cover_audit.json').write_text(json.dumps(audit,indent=2))
                print('STATIC_MESH_COVER_AUDIT',json.dumps(dict(certified=audit['certified'],missing=audit['missing_mesh_mappings'])),flush=True)
                if args.clearance_mm<5. and not audit['certified']:
                    raise RuntimeError('mesh coverage audit failed: reduced clearance prohibited')
            term.cfg.required_clearance_m=args.clearance_mm/1000
            summary['required_clearance_m']=term.cfg.required_clearance_m
            summary['source_minimum_height_m']=args.source_height_mm/1000
            saved=None
            if args.resume_state:
                with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,type(np.dtype('float64'))]):
                    saved=torch.load(args.resume_state,map_location=env.device,weights_only=True)
                config_hash=hashlib.sha256(json.dumps(bridge.config,sort_keys=True).encode()).hexdigest()
                if saved['config_hash']!=config_hash or saved['geometry_hash']!=runtime.reference.geometry_sha256:
                    raise ValueError('resume physical configuration or stomach geometry mismatch')
                env.scene.reset_to(saved['scene'],is_relative=False)
                env.sim.forward()  # propagate restored link poses without a physics step
                env.scene.update(0.)
                term.reset();term._target[:]=saved['servo_reference']
                bridge._filtered_wrench[:]=saved['filtered_wrench'];bridge.elapsed[:]=saved['bridge_elapsed']
                summary['initialization']='restored_dynamic_snapshot_no_extra_hold'
                summary['resume_sha256']=hashlib.sha256(args.resume_state.read_bytes()).hexdigest()
            (output/'physical_config.json').write_text(json.dumps(bridge.config,indent=2))
            summary['initial_capsule_pose_xyzw']=torch.cat([capsule.data.root_link_pos_w.torch[0],capsule.data.root_link_quat_w.torch[0]]).cpu().tolist()
            summary['initial_joint_rad']=term.robot.data.joint_pos.torch[0,term.ids].cpu().tolist()
            summary['geometry_sha256']=runtime.reference.geometry_sha256
            np.savez_compressed(output/'reference_mesh.npz',vertices_world=runtime.reference.vertices_world,
                triangles=runtime.reference.triangles,geometry_sha256=runtime.reference.geometry_sha256)
            summary['coverage_total_area_m2']=runtime.accumulator.total_area_m2
            def separation():
                p=capsule.data.root_link_pos_w.torch[0].cpu().numpy()
                q=capsule.data.root_link_quat_w.torch[0].cpu().numpy()
                center=p+Rotation.from_quat(q).apply([0,0,float(bridge.config['magnets']['target_cylinder']['center_offset_axis_m'])])
                delta=term.pose(term.source)[0]-center
                return float(np.linalg.norm(delta)),delta
            d0,_=separation();summary['initial_magnet_distance_m']=d0
            probe=None
            if args.stimulus=='field_sweep':
                from robotarm_magnetic_lab.baselines.magnetic_field_probe import FieldSweepProbe
                _,delta=separation()
                probe=FieldSweepProbe(term,term.pose(term.source)[0]-delta)
            if args.stimulus=='surface_roll':
                from robotarm_magnetic_lab.baselines.magnetic_surface_roll import SurfaceRollProbe
                probe=SurfaceRollProbe(term,bridge,runtime.reference,args.roll_direction_sign)
                if args.escape_force_mn is not None:
                    from robotarm_magnetic_lab.baselines.magnetic_roll_escape import RollEscapeProbe
                    probe=RollEscapeProbe(term,bridge,runtime.reference,args.roll_direction_sign,args.escape_force_mn/1000,args.source_height_mm/1000,args.motion_pattern)
                    summary['escape_force_mn']=args.escape_force_mn
                    summary['distance_monitor']='farther_than_initial_1.2x_for_2s; intentional_approach_allowed'
                if saved is not None:
                    for name,value in saved['probe'].items():
                        setattr(probe,name,np.asarray(value) if isinstance(value,list) else value)
                    if 'distance_reference_m' in saved: probe.distance=float(saved['distance_reference_m'])
                summary['roll_direction_sign']=args.roll_direction_sign
                summary['runtime_physics_config']=bridge.config['simulation'].copy()
            roll_stream=(output/'surface_roll_10hz.jsonl').open('w') if args.stimulus=='surface_roll' else None
            if roll_stream is not None: handles.append(roll_stream)
            cf=(output/'coverage_10hz.csv').open('w',newline='');handles.append(cf)
            cw=csv.DictWriter(cf,fieldnames=['time_s','frame','coverage','visible_area_m2','distance_m','distance_warning','rgb_sha256']);cw.writeheader()
            stream=(output/'boundaries.jsonl').open('w');handles.append(stream)
            state=(output/'physics_240hz.jsonl').open('w');handles.append(state)
            def capture():
                nonlocal last_frame,bad_distance
                now=physics_steps/240
                observer.pose();env.sim.render()
                frame=fresh(camera)
                if last_frame is not None and frame!=last_frame+1: raise RuntimeError(f'RGB boundary mismatch {last_frame}->{frame}')
                last_frame=frame
                rgb=camera.data.output['rgb'].torch[0,...,:3]
                if not bool(torch.isfinite(rgb.float()).all()): raise RuntimeError('nonfinite RGB')
                digest=hashlib.sha256(rgb.cpu().numpy().tobytes()).hexdigest()
                runtime.screen_time_s=now
                update=runtime.maybe_update(expected_camera_frame=frame,rgb_content_sha256=digest,write_record=False)
                if update is None: raise RuntimeError('missing coverage update')
                runtime.latest_record['timestamp_s']=now
                runtime.writer.append_frame(runtime.latest_record)
                d,delta=separation();distances.append(d)
                warning=(d/d0>1.1) if args.escape_force_mn is not None else abs(d/d0-1)>.1
                outside=(d/d0>1.2) if args.escape_force_mn is not None else abs(d/d0-1)>.2
                bad_distance=bad_distance+1 if outside else 0
                cw.writerow(dict(time_s=now,frame=frame,coverage=float(update.coverage_fraction),
                    visible_area_m2=float(update.visible_area_m2),distance_m=d,distance_warning=int(warning),rgb_sha256=digest));cf.flush()
                fresh(observer.external);fresh(observer.internal)
                observer.capture(now,float(update.coverage_fraction))
                if roll_stream is not None:
                    roll_stream.write(json.dumps(dict(time_s=now,**probe.sample()),allow_nan=False)+'\n');roll_stream.flush()
                summary['frames']+=1;summary['coverage']=float(update.coverage_fraction)
                summary['elapsed_simulation_s']=now
                if physics_steps%3600==0:
                    runtime.snapshot(f't{now:06.1f}');observer.snapshot(f't{now:06.1f}')
                    cv2.imwrite(str(output/'video'/f't{now:06.1f}_capsule.png'),cv2.cvtColor(rgb.cpu().numpy(),cv2.COLOR_RGB2BGR))
                    print('VECTOR_SCREEN_PROGRESS',json.dumps(dict(policy=args.policy,time_s=now,coverage=summary['coverage'],distance_m=d)),flush=True)
                if bad_distance>=20: raise DistanceExceeded('magnet center distance outside +/-20% initial for 2 seconds')
            capture()
            summary['C0']=summary['coverage']
            old_update=env.scene.update
            def update_scene(scene,dt,*a,**kw):
                nonlocal physics_steps
                old_update(dt,*a,**kw)
                if abs(float(dt)-1/240)>1e-9: raise RuntimeError('unexpected physics update interval')
                physics_steps+=1
                if physics_steps%24==0: capture()
            env.scene.update=MethodType(update_scene,env.scene)
            for k in range(args.seconds):
                chunk=probe.chunk() if probe else actions[k:k+4]
                with threadpool_limits(limits=1,user_api='blas'):
                    result=env.step(torch.tensor(chunk.reshape(1,36),device=env.device,dtype=torch.float32))
                if bool(result[2].any()) or bool(result[3].any()):
                    summary['termination_diagnostic']=getattr(env,'_screen_clearance',{})
                    summary['termination_diagnostic'].update(terminated=result[2].cpu().tolist(),truncated=result[3].cpu().tolist())
                    raise RuntimeError('unexpected autoreset')
                record=term.boundary_result();record.update(cycle=k,time_s=k+1,
                    coverage=summary['coverage'],applied_wrench=bridge.state['wrench'][0].cpu().tolist())
                if hasattr(probe,'diagnostics'): record['surface_controller']=probe.diagnostics
                if hasattr(env,'_screen_clearance'):record['actual_clearance']=env._screen_clearance.copy()
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                for item in term.physics_records: state.write(json.dumps(dict(item,cycle=k),allow_nan=False)+'\n')
                state.flush()
                summary['completed_cycles']=k+1
                summary['tracking_failures']+=int(record['tracking_acceptance']!='passed')
                summary['projected_cycles']+=int(record['projection_scale']!=1.)
                (output/'status.json').write_text(json.dumps(summary,indent=2))
            summary['status']='completed'
            if args.stimulus=='surface_roll':
                torch.save(dict(scene=env.scene.get_state(is_relative=False),
                    servo_reference=term._target.clone(),filtered_wrench=bridge._filtered_wrench.clone(),
                    bridge_elapsed=bridge.elapsed.clone(),
                    config_hash=hashlib.sha256(json.dumps(bridge.config,sort_keys=True).encode()).hexdigest(),
                    geometry_hash=runtime.reference.geometry_sha256,
                    probe={name:(getattr(probe,name).tolist() if isinstance(getattr(probe,name),np.ndarray) else getattr(probe,name))
                        for name in ('axis','normal','direction','goal','stage','ready','cycle')},
                    distance_reference_m=float(probe.distance)),
                    output/'final_dynamic_state.pt')
    except BaseException as exc:
        summary['status']='distance_limit' if isinstance(exc,DistanceExceeded) else 'error'
        summary['error']=repr(exc);summary['traceback']=traceback.format_exc();traceback.print_exc()
        if env is not None:
            term=env.action_manager.get_term('magnet')
            (output/'interrupted_action.json').write_text(json.dumps(dict(telemetry=term.telemetry,
                physics=getattr(term,'physics_records',[])),allow_nan=False))
    finally:
        if env is not None and old_update is not None: env.scene.update=old_update
        if observer is not None: observer.snapshot('final');observer.close()
        if runtime is not None:
            try: summary['coverage_directory']=str(runtime.finalize(summary['status']))
            except Exception as exc: summary['finalize_error']=repr(exc)
        for f in handles: f.close()
        summary['wall_s']=time.monotonic()-start
        if distances: summary['distance_range_m']=[min(distances),max(distances)]
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        (output/'status.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        print('VECTOR_SCREEN_SUMMARY',json.dumps(summary),flush=True)
        if env is not None: env.close()
    if summary['status']=='error': raise SystemExit(1)

if __name__=='__main__': main()
