"""实验用任务空间磁滚动编译器；只输出原4x9增量，不给胶囊直接施力。"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from trimesh.triangles import closest_point


def unit(v):
    v=np.asarray(v,float)
    if np.linalg.norm(v)<1e-10: raise ValueError('undefined direction')
    return v/np.linalg.norm(v)


def magnetic_heading(source, capsule, field):
    n=unit(capsule-source)
    return unit((-np.eye(3)+1.5*np.outer(n,n))@unit(field))


def dipole_force(source,capsule,M,m):
    r=capsule-source;d=np.linalg.norm(r);n=r/d
    return 3e-7/d**4*((m@n)*M+(M@n)*m+(m@M)*n-5*(m@n)*(M@n)*n)


def tangent_axis(axis,normal):
    projected=axis-normal*np.dot(axis,normal)
    if np.linalg.norm(projected)<.05:
        basis=np.eye(3)[np.argmin(np.abs(normal))]
        projected=basis-normal*np.dot(normal,basis)
    return unit(projected)


def bounded_jacobian(function, x, bounds, step=1e-4):
    """Absolute angular perturbations survive float32 collision-query rounding."""
    x=np.asarray(x,float);bounds=np.asarray(bounds,float)
    columns=[]
    for i in range(len(x)):
        lo=x.copy();hi=x.copy()
        lo[i]=max(-bounds[i],x[i]-step)
        hi[i]=min(bounds[i],x[i]+step)
        columns.append((function(hi)-function(lo))/(hi[i]-lo[i]))
    return np.stack(columns,axis=1)


class SurfaceRollProbe:
    def __init__(self,term,bridge,reference,direction_sign=1):
        self.term,self.bridge=term,bridge
        if direction_sign not in (-1,1): raise ValueError('roll direction sign must be +/-1')
        self.direction_sign=direction_sign
        self.triangles=reference.vertices_world[reference.triangles]
        self.centers=self.triangles.mean(axis=1)
        vectors=np.cross(self.triangles[:,1]-self.triangles[:,0],self.triangles[:,2]-self.triangles[:,0])
        self.face_normals=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-15)
        self.stage='tip';self.ready=0;self.cycle=0;self.diagnostics={}
        state=self.observe();self.normal=state['normal'];self.axis=tangent_axis(state['axis'],self.normal)
        self.direction=direction_sign*unit(np.cross(self.normal,self.axis));self.origin=state['position'].copy()
        self.roll_origin=None;self.roll_integral=0.;self.roll_start_integral=0.
        self.contact_samples=0;self.samples=0;self.last_q=None;self.last_position=None
        self.signed_path=0.;self.roll_path_start=0.
        source,_=term.pose(term.source)
        self.distance=np.linalg.norm(source-state['magnet'])
        cfg=bridge.config;mu0=4*np.pi*1e-7
        self.M=cfg['magnets']['main_cube']['remanence_t']*np.prod(cfg['magnets']['main_cube']['dimensions_m'])/mu0
        c=cfg['magnets']['target_cylinder']
        self.m=c['remanence_t']*np.pi*(c['diameter_m']/2)**2*c['height_m']/mu0
        self.weight=cfg['external_magnet']['capsule']['total_mass_kg']*9.81
        self.goal=source.copy()

    def observe(self):
        cap=self.term.capsule
        p=cap.data.root_link_pos_w.torch[0].detach().cpu().numpy().astype(float)
        q=cap.data.root_link_quat_w.torch[0].detach().cpu().numpy().astype(float)
        R=Rotation.from_quat(q).as_matrix();axis=R[:,2]
        points=closest_point(self.triangles,np.broadcast_to(p,(len(self.triangles),3)))
        face=int(np.argmin(np.linalg.norm(points-p,axis=1)));surface=points[face]
        normal=self.face_normals[face].copy()
        if normal@(p-surface)<0: normal=-normal
        # Orient neighboring normals consistently; smooth over one capsule radius.
        nearby=np.linalg.norm(self.centers-surface,axis=1)<.009
        normals=self.face_normals[nearby].copy()
        if len(normals):
            normals*=np.where(normals@normal>=0,1.,-1.)[:,None]
            normal=unit(normals.sum(axis=0))
        contact=self.term._env.scene['capsule_contact'].data.net_forces_w.torch[0,0].detach().cpu().numpy()
        return dict(position=p,q=q,axis=axis,normal=normal,face=face,surface=surface,
            magnet=p+R@np.array([0,0,self.bridge.config['magnets']['target_cylinder']['center_offset_axis_m']]),
            contact_N=float(np.linalg.norm(contact)))

    def sample(self):
        # Called at10Hz for roll/contact evidence; angular increment is world-frame.
        s=self.observe();q=Rotation.from_quat(s['q'])
        if self.last_q is not None:
            delta=(q*self.last_q.inv()).as_rotvec()
            self.roll_integral+=float(delta@unit(s['axis']+self.last_axis))
            self.signed_path+=float((s['position']-self.last_position)@self.direction)
        self.last_q=q;self.last_axis=s['axis'];self.last_position=s['position']
        self.samples+=1;self.contact_samples+=int(s['contact_N']>=1e-4)
        roll=self.roll_integral-self.roll_start_integral
        travel=self.signed_path-self.roll_path_start
        return dict(stage=self.stage,axis_world=s['axis'].tolist(),normal_world=s['normal'].tolist(),
            contact_force_N=s['contact_N'],surface_face=s['face'],
            tilt_from_tangent_deg=float(np.rad2deg(np.arcsin(np.clip(abs(s['axis']@s['normal']),0,1)))),
            integrated_long_axis_roll_rad=roll,signed_lateral_path_m=travel,
            signed_no_slip_residual_m=travel+self.direction_sign*.0065*roll,
            rolling_slip_ratio=abs(abs(travel)-.0065*abs(roll))/max(abs(travel),.0065*abs(roll),1e-6),
            position_world_m=s['position'].tolist(),capsule_quaternion_xyzw=s['q'].tolist())

    def chunk(self):
        from ..tasks.manager_based.robotarm_magnetic_lab.controllers.source_kinematics import source_pose
        from ..tasks.manager_based.robotarm_magnetic_lab.controllers.actuator_vector import SCALES, decode
        from ..tasks.manager_based.robotarm_magnetic_lab.controllers.batched_arm_clearance import minimum_clearance_fast, minimum_world_clearance
        t=self.term;s=self.observe();self.cycle+=1
        normal=unit(.8*self.normal+.2*s['normal']);self.normal=normal
        self.axis=tangent_axis(self.axis,normal)
        self.direction=self.direction_sign*unit(np.cross(normal,self.axis))
        sp,sr=t.pose(t.source);ep,er=t.pose(t.ee)
        B=unit(self.bridge.model.field_tesla([s['magnet']],sp,sr).reshape(3))
        tangent_error=np.rad2deg(np.arcsin(np.clip(abs(s['axis']@normal),0,1)))
        heading_error=np.rad2deg(np.arccos(np.clip(B@self.axis,-1,1)))
        axis_error=np.rad2deg(np.arccos(np.clip(s['axis']@self.axis,-1,1)))
        ready=tangent_error<15 and heading_error<10 and axis_error<20 and s['contact_N']>=1e-4
        self.ready=self.ready+1 if ready else 0
        if self.stage=='tip' and self.ready>=3:
            self.stage='roll';self.roll_origin=s['position'].copy()
            self.roll_start_integral=self.roll_integral;self.roll_path_start=self.signed_path
        # Choose a field/gradient goal; gravity tangent compensation is explicit.
        force_goal=.010*self.direction+.012*normal
        gravity=np.array([0.,0.,-self.weight])
        force_goal-=gravity-normal*(gravity@normal)
        if self.stage=='roll' and (self.cycle%5==0 or np.linalg.norm(self.goal-sp)<.002):
            anchor=s['magnet'];low=anchor+np.array([-.15,-.15,.12]);high=anchor+np.array([.15,.15,.30])
            def residual(p):
                M=self.M*magnetic_heading(p,anchor,self.axis)
                f=dipole_force(p,anchor,M,self.m*self.axis)
                d=np.linalg.norm(p-anchor)
                return np.r_[(f-force_goal)/.010,.1*(p-sp)/.05,
                    max(0,.85*self.distance-d)/.01,max(0,d-1.1*self.distance)/.01]
            fit=least_squares(residual,np.clip(self.goal,low+1e-7,high-1e-7),bounds=(low,high),max_nfev=40)
            self.goal=fit.x
        q=t.robot.data.joint_pos.torch[0,t.ids].detach().cpu().numpy().astype(float)
        J=t.robot.data.body_link_jacobian_w.torch[0,t.source-int(t.robot.is_fixed_base)].detach().cpu().numpy()[:,np.asarray(t.ids)+int(t.robot.num_base_dofs)]
        JE=t.robot.data.body_link_jacobian_w.torch[0,t.ee-int(t.robot.is_fixed_base)].detach().cpu().numpy()[:,np.asarray(t.ids[:6])+int(t.robot.num_base_dofs)]
        cumulative=np.zeros(9);previous_p=ep.copy();previous_r=er.copy();result=[]
        base=t.pose(t.base)[1];t._geometry();solver=[]
        def candidate_clearance(candidate):
            # The backend's cached displacement lower bound is a safety
            # certificate, not a geometric distance field for optimization.
            path=(q+candidate)[None,:6]
            value=float(minimum_clearance_fast(t.kinematics,path)[0])
            if t.world_checker is not None:
                value=min(value,float(minimum_world_clearance(t.world_checker,path)[0]))
            return value
        for k in range(4):
            start=cumulative.copy()
            prev_sp,_=source_pose(sp,sr,J,start[None]);desired=prev_sp[0].copy()
            if self.stage=='roll':
                diff=self.goal-desired;desired+=diff*min(1.,.001/max(np.linalg.norm(diff),1e-12))
            def residual(dq):
                candidate=start+dq
                ps,rs=source_pose(sp,sr,J,candidate[None]);pe,re=source_pose(ep,er,JE,candidate[:6][None])
                heading=magnetic_heading(ps[0],s['magnet'],self.axis)
                rp=Rotation.from_matrix(previous_r.T@re[0]).as_rotvec()
                translation=(pe[0]-previous_p)/.001
                rot=rp/np.deg2rad(.5)
                clearance=candidate_clearance(candidate) if self.stage=='roll' else .01
                return np.r_[(ps[0]-desired)/.001,(rs[0,:,2]-heading)/.03,
                    .15*translation,.15*rot,
                    10*max(0,np.linalg.norm(translation)-1),10*max(0,np.linalg.norm(rot)-1),
                    10*max(0,np.linalg.norm(dq[6:])/np.deg2rad(5)-1),
                    10*max(0,.006-clearance)/.001,.001*dq]
            bounds=np.r_[[.006]*6,[np.deg2rad(5)]*3]
            if self.stage=='tip': bounds[:6]=1e-12
            initial_cost=float(.5*np.sum(residual(np.zeros(9))**2))
            fit=least_squares(residual,np.zeros(9),bounds=(-bounds,bounds),
                jac=lambda x:bounded_jacobian(residual,x,bounds),max_nfev=20,ftol=1e-5)
            solver.append(dict(status=int(fit.status),message=fit.message,nfev=int(fit.nfev),
                initial_cost=initial_cost,cost=float(fit.cost),
                increment_norm_rad=float(np.linalg.norm(fit.x))))
            cumulative=start+fit.x
            pe,re=source_pose(ep,er,JE,cumulative[:6][None])
            row=np.r_[base.T@(pe[0]-previous_p),Rotation.from_matrix(previous_r.T@re[0]).as_rotvec(),fit.x[6:]]/SCALES
            result.append(row);previous_p,previous_r=pe[0],re[0]
        finite_force,_=self.bridge.model.force_torque_si(sp,sr,s['magnet'],Rotation.from_quat(s['q']).as_matrix())
        self.diagnostics=dict(stage=self.stage,ready_streak=self.ready,field_error_deg=float(heading_error),
            capsule_axis_error_deg=float(axis_error),tilt_from_tangent_deg=float(tangent_error),
            goal_source_world_m=self.goal.tolist(),desired_B_world=self.axis.tolist(),
            desired_force_world_N=force_goal.tolist(),raw_finite_force_N=np.asarray(finite_force).tolist(),
            direction_world=self.direction.tolist(),source_world_m=sp.tolist(),solver=solver)
        return decode(np.asarray(result))[0]
