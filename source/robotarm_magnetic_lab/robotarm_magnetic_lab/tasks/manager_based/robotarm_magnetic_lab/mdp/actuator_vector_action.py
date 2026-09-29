"""Independent 1 Hz actuator-delta backend; old magnetic controllers unchanged."""
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass
from robotarm_magnetic_lab.runtime.quaternion_conventions import rotation_matrix_from_xyzw
from ..controllers.action_layer.kinematics import UrdfXrdfSafetyModel
from ..controllers.actuator_vector import decode, goals, plan, SCALES, ACCELERATION, Infeasible, clearance_lower_bound
from ..controllers.source_kinematics import source_pose
from ..controllers.ball_feedback import revolute_branch_shift, stopping_path
from ..controllers.ball_envelope import audit_stage, install
from ..controllers.batched_arm_clearance import minimum_clearance_fast, minimum_world_clearance, link_transforms


def array(x):
    return x.detach().cpu().numpy().astype(float)


class ActuatorVectorAction(ActionTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if not np.isfinite(cfg.required_clearance_m) or cfg.required_clearance_m<.003:
            raise ValueError('clearance must be finite and at least 3mm')
        if env.num_envs!=1 or abs(env.step_dt-1)>1e-9 or abs(env.cfg.sim.dt-1/240)>1e-9:
            raise ValueError('single environment, 1Hz boundary, 240Hz physics required')
        self.robot,self.capsule=env.scene['robot'],env.scene['capsule']
        self.ids=[self.robot.data.joint_names.index(n) for n in
                  ('j1','j2','j3','j4','j5','j6','ballxj','ballyj','ballzj')]
        self.ee=self.robot.data.body_names.index('l6')
        self.source=self.robot.data.body_names.index('magl')
        self.base=self.robot.data.body_names.index('base_link')
        self._raw=torch.zeros((1,36),device=env.device)
        self._processed=self._raw.clone()
        self._target=self.robot.data.joint_pos.torch[:,self.ids].clone()
        self.kinematics=UrdfXrdfSafetyModel(cfg.urdf_path,cfg.xrdf_path)
        self.world_checker=None
        self.envelope=None
        self.execution=None
        self.telemetry={}
        self.records=[]
        self.tick=0
        self.clearance_anchor=None

    @property
    def action_dim(self): return 36  # serialized 4 x 9 chunk, not 36 actuator DOFs
    @property
    def raw_actions(self): return self._raw
    @property
    def processed_actions(self): return self._processed

    def reset(self,env_ids=None):
        self._target[:]=self.robot.data.joint_pos.torch[:,self.ids]
        self.execution=None
        self.records=[]
        self.tick=0
        self.clearance_anchor=None

    def pose(self,body):
        return (array(self.robot.data.body_pos_w.torch[0,body]),
                rotation_matrix_from_xyzw(array(self.robot.data.body_quat_w.torch[0,body])))

    def clearance(self,path):
        result=float('inf')
        path=np.unique(np.asarray(path)[:,:6],axis=0)
        # Conservative distance lower bound, not coarse time subsampling.
        transforms=link_transforms(self.kinematics,path)
        if self.clearance_anchor is None:
            initial={name:t[0].copy() for name,t in transforms.items()}
            arm=float(minimum_clearance_fast(self.kinematics,path[:1])[0])
            world=float(minimum_world_clearance(self.world_checker,path[:1])[0]) if self.world_checker else float('inf')
            self.clearance_anchor=(initial,arm,world)
        initial,arm,world=self.clearance_anchor
        lower=clearance_lower_bound(transforms,initial,
            (self.kinematics.spheres,self.kinematics.environment_spheres),arm,world)
        if lower>=self.cfg.required_clearance_m: return lower
        # All physics-rate samples, memory bounded (no coarse stride).
        for start in range(0,len(path),16):
            q=path[start:start+16,:6]
            value=minimum_clearance_fast(self.kinematics,q)
            if self.world_checker is not None:
                value=np.minimum(value,minimum_world_clearance(self.world_checker,q))
            result=min(result,float(np.min(value)))
        return result

    def _geometry(self):
        if self.envelope is None:
            import omni.usd
            from ..controllers.mesh_cover_audit import audit_stage_static_meshes
            self.mesh_cover_audit=audit_stage_static_meshes(
                omni.usd.get_context().get_stage(),self.kinematics,repair_native=True)
            if not self.mesh_cover_audit['certified']:
                raise RuntimeError(f'native mesh collision cover failed: {self.mesh_cover_audit}')
            self.clearance_anchor=None
            audit=audit_stage(omni.usd.get_context().get_stage())
            p,r=self.pose(self.ee)
            center=r.T@(self.pose(self.source)[0]-p)
            install(self.kinematics,center,audit['radius_m'])
            from ..controllers.native_mesh_clearance import NativeMeshClearance
            bridge=self._env.event_manager.get_term_cfg('magnetic_collision_bridge').func
            self.kinematics._native_narrow=NativeMeshClearance(
                omni.usd.get_context().get_stage(),self.kinematics,str(self._env.device),
                asm_mount=bridge.config['planning']['asm_mount_l6_from_asm'],
                ball_envelope=(center,audit['radius_m']))
            self.envelope=dict(audit,center_l6_m=center.tolist())
            if str(self._env.device).startswith('cuda'):
                self.kinematics._clearance_device=str(self._env.device)
        if self.cfg.world_mesh_path and self.world_checker is None:
            from .world_collision import StomachMeshCollisionChecker
            p,r=self.pose(self.base)
            self.world_checker=StomachMeshCollisionChecker(self.kinematics,
                mesh_prim_path=self.cfg.world_mesh_path,robot_base_position_world_m=p,
                robot_base_rotation_world=r,device=str(self._env.device),
                required_clearance_m=self.cfg.required_clearance_m,trajectory_samples=25)

    def _rebase(self):
        q=array(self.robot.data.joint_pos.torch[0,self.ids])
        ref=array(self._target[0])
        shift=revolute_branch_shift(ref[6:],q[6:])
        if np.any(shift):
            self._target[0,6:]=self._target.new_tensor(ref[6:]+shift)
            if self.execution is not None: self.execution[:,6:]+=shift
            if hasattr(self,'goal_ball'): self.goal_ball+=shift
        return q

    def process_actions(self,actions):
        started=time.monotonic()
        if tuple(actions.shape)!=(1,36): raise ValueError('one flattened 4x9 chunk required')
        raw=array(actions)[0].reshape(4,9)
        clipped,physical=decode(raw,np.asarray(self.cfg.scales))
        self._raw[:]=actions
        q=self._rebase()
        ref=array(self._target[0])
        self._geometry()
        p,r=self.pose(self.ee)
        jac=array(self.robot.data.body_link_jacobian_w.torch[0,self.ee-int(self.robot.is_fixed_base)])
        jac=jac[:,np.asarray(self.ids[:6])+int(self.robot.num_base_dofs)]
        def fk(angles):
            pp,rr=source_pose(p,r,jac,(angles-q[:6])[None])
            return pp[0],rr[0]
        self.fk_audit=fk
        # Integrate relative to continuous servo reference, not a fresh
        # gravity-deflected measurement on every HOLD boundary.
        rp,rr=fk(ref[:6])
        base_rotation=self.pose(self.base)[1]
        limits=array(self.robot.data.soft_joint_pos_limits.torch[0,self.ids])
        error=q-ref
        def certified(path):
            return min(self.clearance(path),self.clearance(path+error))
        reasons=[]
        result=None
        for scale in (1.,.5,.25,0.):
            desired=goals(rp,rr,ref[6:],physical*scale,base_rotation)
            try:
                result=plan(ref,fk,desired,limits,certified,margin=self.cfg.required_clearance_m)
                break
            except Infeasible as exc:
                reasons.append(str(exc))
        if result is None: raise RuntimeError('no certified vector trajectory or HOLD: '+repr(reasons))
        self.execution=result['q'][:241].copy()
        self.velocities=result['velocity'][:241].copy()
        self.goal_position,self.goal_rotation,self.goal_ball=desired[0]
        self._processed[:]=self._processed.new_tensor((clipped*scale).reshape(1,36))
        self.tick=0
        self.records=[]
        self.physics_records=[]
        self.telemetry=dict(raw_chunk=raw.tolist(),bounded_chunk=clipped.tolist(),
            physical_scales=list(self.cfg.scales),applied_chunk=(physical*scale).tolist(),
            projection_scale=scale,reasons=reasons,status='committed' if scale else 'rejected_hold',
            preview_seconds=4,committed_seconds=1,physics_steps=240,
            predicted_min_clearance_m=result['clearance_m'],
            goal_position_m=self.goal_position.tolist(),goal_quat_xyzw=Rotation.from_matrix(self.goal_rotation).as_quat().tolist(),
            goal_ball_rad=self.goal_ball.tolist(),planner_wall_s=time.monotonic()-started,
            initial_servo_error_rad=error.tolist(),envelope=self.envelope,
            native_narrow_query_count=self.kinematics._native_narrow.query_count,
            clearance_geometry='signed native triangle BVH/tight ASM-Ball + conservative world spheres',
            predicted_joint_knots_rad=result['knots'].tolist())

    def apply_actions(self):
        if self.execution is None: return
        q=self._rebase()
        if not np.isfinite(q).all(): raise RuntimeError('nonfinite actuator state')
        measured_velocity=array(self.robot.data.joint_vel.torch[0,self.ids])
        if not np.isfinite(measured_velocity).all(): raise RuntimeError('nonfinite actuator velocity')
        self.physics_records.append(dict(substep=self.tick,joint_actual_rad=q.tolist(),
            joint_reference_rad=array(self._target[0]).tolist(),joint_velocity_rad_s=measured_velocity.tolist()))
        if self.tick>=240: raise RuntimeError('more than 240 steps for one command')
        if self.tick%12==0:
            velocity=array(self.robot.data.joint_vel.torch[0,self.ids])
            actual_stop,_=stopping_path(q,velocity,ACCELERATION)
            reference_stop,_=stopping_path(array(self._target[0]),self.velocities[self.tick],ACCELERATION)
            limits=array(self.robot.data.soft_joint_pos_limits.torch[0,self.ids])
            for stop in (actual_stop,reference_stop):
                if np.any(stop<limits[:,0]) or np.any(stop>limits[:,1]):
                    raise RuntimeError('uncertifiable stop joint limits')
            if np.max(np.abs(velocity)/ACCELERATION)>2.:
                raise RuntimeError('measured stop exceeds certified two-second horizon')
            margin=min(self.clearance(actual_stop),self.clearance(reference_stop))
            if margin<self.cfg.required_clearance_m:
                # No unchecked emergency teleport or unverified force boost.
                raise RuntimeError(f'uncertifiable measured/reference stop clearance {margin}')
            self.records.append(dict(substep=self.tick,joint_actual_rad=q.tolist(),
                joint_velocity_rad_s=velocity.tolist(),stop_clearance_m=margin,
                source_position_m=self.pose(self.source)[0].tolist(),
                capsule_position_m=array(self.capsule.data.root_link_pos_w.torch[0]).tolist()))
        self.tick+=1
        self._target[0]=self._target.new_tensor(self.execution[self.tick])
        self.robot.set_joint_position_target_index(target=self._target,joint_ids=self.ids)
        self.robot.set_joint_velocity_target_index(target=self._target.new_tensor(self.velocities[self.tick][None]),joint_ids=self.ids)
        if self.cfg.gravity_compensation:
            columns=np.asarray(self.ids)+int(self.robot.num_base_dofs)
            gravity=self.robot.data.gravity_compensation_forces.torch[:,columns].clone()
            if not bool(torch.isfinite(gravity).all()): raise RuntimeError('nonfinite gravity compensation')
            self.robot.set_joint_effort_target_index(target=gravity,joint_ids=self.ids)

    def boundary_result(self):
        q=self._rebase()
        p,r=self.pose(self.ee)
        source=self.pose(self.source)[0]
        capsule=array(self.capsule.data.root_link_pos_w.torch[0])
        ep=float(np.linalg.norm(p-self.goal_position))
        er=float(Rotation.from_matrix(self.goal_rotation.T@r).magnitude())
        eb=float(np.max(np.abs(q[6:]-self.goal_ball)))
        ap,ar=self.fk_audit(q[:6])
        return dict(self.telemetry,executed_substeps=self.tick,
            position_error_m=float(np.linalg.norm(p-self.goal_position)),
            rotation_error_rad=float(Rotation.from_matrix(self.goal_rotation.T@r).magnitude()),
            ball_error_rad=(q[6:]-self.goal_ball).tolist(),
            actual_ee_position_m=p.tolist(),actual_ee_quat_xyzw=Rotation.from_matrix(r).as_quat().tolist(),
            actual_joint_rad=q.tolist(),source_capsule_distance_m=float(np.linalg.norm(source-capsule)),
            fk_position_error_m=float(np.linalg.norm(ap-p)),
            fk_rotation_error_rad=float(Rotation.from_matrix(ar.T@r).magnitude()),
            tracking_tolerances=[self.cfg.position_tolerance_m,self.cfg.rotation_tolerance_rad,self.cfg.ball_tolerance_rad],
            tracking_acceptance='passed' if ep<=self.cfg.position_tolerance_m and er<=self.cfg.rotation_tolerance_rad and eb<=self.cfg.ball_tolerance_rad else 'failed')


@configclass
class ActuatorVectorActionCfg(ActionTermCfg):
    class_type:type=ActuatorVectorAction
    urdf_path:str='/home/multirobo/Desktop/robotarm/urdf/robotarm.urdf'
    xrdf_path:str='/mnt/isaac-linux/isaacsim/extsUser/robotarm.magnetic_sim/data/planning/robot.xrdf'
    world_mesh_path:str|None=None
    scales:tuple=tuple(float(x) for x in SCALES)
    gravity_compensation:bool=True
    required_clearance_m:float=.005
    position_tolerance_m:float=.0001
    rotation_tolerance_rad:float=float(np.deg2rad(.02))
    ball_tolerance_rad:float=float(np.deg2rad(.02))
