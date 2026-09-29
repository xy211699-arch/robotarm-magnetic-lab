"""滚动增力候选：有限磁体力目标与执行器/几何联合优化，旧控制器不变。

胶囊真值仅供已授权的仿真低层控制。所有输出仍是4×9相对增量。
没有胶囊力执行器；真正外力仍由现有磁场桥计算与施加。
"""
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .magnetic_surface_roll import SurfaceRollProbe,unit,tangent_axis,bounded_jacobian


def force_target(tangent,normal,weight,tangential_n):
    if tangential_n not in (.010,.015,.020,.030):
        raise ValueError('approved tangential force levels: 10/15/20/30mN')
    gravity=np.array([0.,0.,-weight])
    return tangential_n*tangent+.012*normal-(gravity-normal*(gravity@normal))


def source_bounds_residual(source,magnet,minimum_height=.120):
    offset=np.asarray(source)-magnet
    return np.r_[np.maximum(0,np.abs(offset[:2])-.15),
                 max(0,minimum_height-offset[2]),max(0,offset[2]-.30)]/.001


class AreaScanHeading:
    """Fixed, reproducible multi-heading excitation; elapsed time is physical steps."""
    targets_deg=(0.,60.,120.,60.,-60.,-120.)

    def __init__(self,axis,normal):
        self.reference=tangent_axis(np.asarray(axis,float),np.asarray(normal,float))
        self.angle_deg=0.
        self.steps=0

    def advance(self,normal):
        self.reference=tangent_axis(self.reference,normal)
        segment=min(self.steps//45,len(self.targets_deg)-1)
        goal=self.targets_deg[segment]
        self.angle_deg+=float(np.clip(goal-self.angle_deg,-2.,2.))
        self.steps+=1
        axis=Rotation.from_rotvec(np.asarray(normal)*np.deg2rad(self.angle_deg)).apply(self.reference)
        return unit(axis),dict(pattern='area_scan',rolling_time_s=self.steps,
            segment=int(segment),target_heading_deg=goal,command_heading_deg=self.angle_deg,
            heading_rate_limit_deg_s=2.)


def batched_clearance_jacobian(residual,clearance_many,x,bounds,start,step=1e-4):
    """Same bounded finite differences, grouping only geometry queries."""
    x=np.asarray(x,float);bounds=np.asarray(bounds,float)
    points=[];denominators=[]
    for i in range(len(x)):
        lo=x.copy();hi=x.copy()
        lo[i]=max(-bounds[i],x[i]-step);hi[i]=min(bounds[i],x[i]+step)
        points.extend((hi,lo));denominators.append(hi[i]-lo[i])
    points=np.asarray(points)
    gaps=clearance_many(np.asarray(start)+points)
    values=[residual(p,clearance_value=float(gap)) for p,gap in zip(points,gaps)]
    return np.stack([(values[2*i]-values[2*i+1])/d for i,d in enumerate(denominators)],axis=1)


class RollEscapeProbe(SurfaceRollProbe):
    def __init__(self,term,bridge,reference,direction_sign=-1,tangential_n=.010,minimum_height=.120,motion_pattern='straight'):
        super().__init__(term,bridge,reference,direction_sign)
        force_target(self.direction,self.normal,self.weight,tangential_n)
        self.tangential_n=tangential_n
        if minimum_height not in (.120,.110,.100):raise ValueError('unapproved source height')
        self.minimum_height=minimum_height
        self.warm=np.zeros(9)
        self.rng=np.random.default_rng(917)
        if motion_pattern not in ('straight','area_scan'):raise ValueError('unknown motion pattern')
        self.scan=AreaScanHeading(self.axis,self.normal) if motion_pattern=='area_scan' else None

    def chunk(self):
        if self.stage=='tip':
            return super().chunk()
        started=time.perf_counter()
        from ..tasks.manager_based.robotarm_magnetic_lab.controllers.source_kinematics import source_pose
        from ..tasks.manager_based.robotarm_magnetic_lab.controllers.actuator_vector import SCALES,decode
        from ..tasks.manager_based.robotarm_magnetic_lab.controllers.batched_arm_clearance import minimum_clearance_fast,minimum_world_clearance
        t=self.term;s=self.observe();self.cycle+=1
        self.normal=unit(.8*self.normal+.2*s['normal'])
        self.axis=tangent_axis(self.axis,self.normal)
        scan_record=None
        if self.scan is not None:
            self.axis,scan_record=self.scan.advance(self.normal)
        # During turns keep traction transverse to the actual capsule long axis.
        traction_axis=tangent_axis(s['axis'],self.normal) if self.scan is not None else self.axis
        self.direction=self.direction_sign*unit(np.cross(self.normal,traction_axis))
        goal_force=force_target(self.direction,self.normal,self.weight,self.tangential_n)
        sp,sr=t.pose(t.source);ep,er=t.pose(t.ee);base=t.pose(t.base)[1]
        q=t.robot.data.joint_pos.torch[0,t.ids].detach().cpu().numpy().astype(float)
        jac=t.robot.data.body_link_jacobian_w.torch[0].detach().cpu().numpy()
        J=jac[t.source-int(t.robot.is_fixed_base)][:,np.asarray(t.ids)+int(t.robot.num_base_dofs)]
        JE=jac[t.ee-int(t.robot.is_fixed_base)][:,np.asarray(t.ids[:6])+int(t.robot.num_base_dofs)]
        cap_rotation=Rotation.from_quat(s['q']).as_matrix()
        t._geometry()
        # Geometry is static during this one solve. Ball-only perturbations
        # share the same arm geometry (the Ball envelope covers all rotations).
        clearance_cache={}
        def clearance_many(candidates):
            path=(q+np.asarray(candidates))[:,:6]
            unique,inverse=np.unique(path,axis=0,return_inverse=True)
            keys=[tuple(row) for row in unique]
            missing=[i for i,key in enumerate(keys) if key not in clearance_cache]
            if missing:
                query=unique[missing]
                value=minimum_clearance_fast(t.kinematics,query)
                if t.world_checker is not None:
                    value=np.minimum(value,minimum_world_clearance(t.world_checker,query))
                clearance_cache.update((keys[i],float(v)) for i,v in zip(missing,value))
            return np.asarray([clearance_cache[key] for key in keys])[inverse]
        def field_force(p,r):
            B=self.bridge.model.field_tesla([s['magnet']],p,r).reshape(3)
            F,_=self.bridge.model.force_torque_si(p,r,s['magnet'],cap_rotation)
            return unit(B),np.asarray(F)
        initial_B,initial_F=field_force(sp,sr)
        cumulative=np.zeros(9);previous_p=ep.copy();previous_r=er.copy()
        result=[];solvers=[];first_prediction=None
        bounds=np.r_[[.006]*6,[np.deg2rad(5)]*3]
        for k in range(4):
            start=cumulative.copy()
            def residual(dq,clearance_value=None):
                candidate=start+dq
                if clearance_value is None:clearance_value=float(clearance_many(candidate[None])[0])
                ps,rs=source_pose(sp,sr,J,candidate[None])
                pe,re=source_pose(ep,er,JE,candidate[:6][None])
                B,F=field_force(ps[0],rs[0])
                translation=(pe[0]-previous_p)/.001
                rot=Rotation.from_matrix(previous_r.T@re[0]).as_rotvec()/np.deg2rad(.5)
                normal_F=float(F@self.normal)
                support=self.weight*abs(self.normal[2])
                return np.r_[(F-goal_force)/.003,(B-self.axis)/.04,
                    .05*translation,.05*rot,
                    10*max(0,np.linalg.norm(translation)-1),
                    10*max(0,np.linalg.norm(rot)-1),
                    10*max(0,np.linalg.norm(dq[6:])/np.deg2rad(5)-1),
                    10*max(0,t.cfg.required_clearance_m+.001-clearance_value)/.001,
                    10*source_bounds_residual(ps[0],s['magnet'],self.minimum_height),
                    10*max(0,normal_F-.8*support)/.003,
                    10*max(0,-.005-normal_F)/.003,.001*dq]
            # Multi-start at the committed horizon, warm-start later previews.
            # All four seconds still receive independent force/geometry solves.
            seeds=([np.zeros(9),np.clip(self.warm,-bounds*.95,bounds*.95),
                   self.rng.uniform(-1,1,9)*bounds*.15] if k==0 else
                   [np.clip(self.warm,-bounds*.95,bounds*.95)])
            seeds=[seed for i,seed in enumerate(seeds)
                   if not any(np.array_equal(seed,previous) for previous in seeds[:i])]
            fits=[least_squares(residual,seed,bounds=(-bounds,bounds),
                jac=lambda x:batched_clearance_jacobian(residual,clearance_many,x,bounds,start),max_nfev=60,ftol=1e-6)
                for seed in seeds]
            fit=min(fits,key=lambda f:f.cost)
            solvers.append(dict(status=int(fit.status),message=fit.message,nfev=int(fit.nfev),
                initial_cost=float(.5*np.sum(residual(np.zeros(9))**2)),cost=float(fit.cost),
                increment_norm_rad=float(np.linalg.norm(fit.x)),
                multistart_costs=[float(f.cost) for f in fits]))
            cumulative=start+fit.x
            pe,re=source_pose(ep,er,JE,cumulative[:6][None])
            row=np.r_[base.T@(pe[0]-previous_p),Rotation.from_matrix(previous_r.T@re[0]).as_rotvec(),fit.x[6:]]/SCALES
            result.append(row);previous_p,previous_r=pe[0],re[0]
            if k==0:
                self.warm=fit.x.copy()
                ps,rs=source_pose(sp,sr,J,cumulative[None]);_,F=field_force(ps[0],rs[0])
                self.goal=ps[0].copy();first_prediction=F.tolist()
        self.diagnostics=dict(stage=self.stage,controller='finite_force_escape_v1',
            tangential_target_N=self.tangential_n,normal_target_N=.012,
            field_error_deg=float(np.rad2deg(np.arccos(np.clip(initial_B@self.axis,-1,1)))),
            capsule_axis_error_deg=float(np.rad2deg(np.arccos(np.clip(s['axis']@self.axis,-1,1)))),
            tilt_from_tangent_deg=float(np.rad2deg(np.arcsin(np.clip(abs(s['axis']@self.normal),0,1)))),
            goal_source_world_m=self.goal.tolist(),desired_B_world=self.axis.tolist(),
            desired_force_world_N=goal_force.tolist(),raw_finite_force_N=initial_F.tolist(),
            direction_world=self.direction.tolist(),source_world_m=sp.tolist(),solver=solvers,
            predicted_first_force_N=first_prediction,net_lateral_force_N=float((initial_F+np.array([0,0,-self.weight]))@self.direction),
            planning_wall_s=time.perf_counter()-started)
        self.diagnostics['asm_clearance_by_frame_m']=t.kinematics.asm_clearance_by_frame(q[:6])
        self.diagnostics['stomach_clearance_m']=float(minimum_world_clearance(t.world_checker,q[None,:6])[0]) if t.world_checker else None
        if t.world_checker is not None:
            witness=t.world_checker.check_configuration(q[:6])
            self.diagnostics['stomach_clearance_witness']=dict(frame=witness.frame,
                sphere_index=witness.sphere_index,face_index=witness.face_index,clearance_m=witness.clearance_m)
            centers,radii=t.kinematics.environment_spheres[witness.frame]
            self.diagnostics['stomach_clearance_witness'].update(
                local_center_m=centers[witness.sphere_index].tolist(),
                sphere_radius_m=float(radii[witness.sphere_index]),
                rotating_ball_envelope=bool(witness.frame==t.kinematics.asm_frame and witness.sphere_index==len(radii)-1))
        self.diagnostics['source_height_offset_m']=float(sp[2]-s['magnet'][2])
        self.diagnostics['asm_narrow_clearance_m']=float(minimum_clearance_fast(t.kinematics,q[None,:6])[0])
        self.diagnostics['asm_clearance_by_frame_note']='legacy sphere proxy only; not the signed triangle/tight ASM acceptance value'
        self.diagnostics['normal_force_N']=float(initial_F@self.normal)
        if scan_record is not None:self.diagnostics['area_scan']=scan_record
        return decode(np.asarray(result))[0]
