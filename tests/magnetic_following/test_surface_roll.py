import numpy as np
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/baselines/magnetic_surface_roll.py'
spec=importlib.util.spec_from_file_location('surface_roll',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
magnetic_heading,dipole_force,tangent_axis=module.magnetic_heading,module.dipole_force,module.tangent_axis


def test_field_inverse_and_transverse_direction():
    n=np.array([.2,-.3,.9]);n/=np.linalg.norm(n)
    axis=tangent_axis(np.array([1.,0,0]),n)
    travel=np.cross(n,axis)
    assert abs(axis@n)<1e-12
    assert abs(travel@axis)<1e-12 and abs(travel@n)<1e-12
    source=np.array([.03,-.04,.2]);caps=np.zeros(3)
    m=magnetic_heading(source,caps,axis)
    r=(caps-source)/np.linalg.norm(caps-source)
    field=(3*np.outer(r,r)-np.eye(3))@m
    assert np.allclose(field/np.linalg.norm(field),axis)


def test_axial_magnet_requires_contact_for_long_axis_roll():
    axis=np.array([1.,0,0]);normal=np.array([0.,0,1.])
    field=np.array([.2,.3,.4])
    assert np.dot(np.cross(axis,field),axis)==0
    for sign in (-1,1):
        travel=sign*np.cross(normal,axis)
        contact_offset=-.0065*normal
        friction=-.01*travel
        torque=np.cross(contact_offset,friction)
        assert np.dot(torque,axis)*sign<0


def test_force_symmetry_and_tangent_degeneracy():
    z=np.array([0.,0,1.])
    a=dipole_force(z*.2,np.zeros(3),z*100,z*.5)
    b=dipole_force(z*.4,np.zeros(3),z*100,z*.5)
    assert a[2]>0 and np.allclose(a,16*b)
    assert abs(tangent_axis(z,z)@z)<1e-12


def test_absolute_jacobian_survives_float32_geometry_and_bounds():
    def residual(x):
        return np.array([float(np.float32(1.+x[0]))-1.001,2*x[1]-.2])
    x=np.zeros(2);bounds=np.array([.006,.1])
    # Default ~1e-8 perturbations disappear at meter-scale float32 positions.
    assert residual(np.array([1e-8,0]))[0]==residual(x)[0]
    jac=module.bounded_jacobian(residual,x,bounds)
    assert np.allclose(jac,np.diag([1.,2.]),atol=.001)
    fit=module.least_squares(residual,x,bounds=(-bounds,bounds),
        jac=lambda v:module.bounded_jacobian(residual,v,bounds))
    assert np.allclose(fit.x,[.001,.1],atol=1e-4)
    assert np.all(np.isfinite(module.bounded_jacobian(residual,bounds,bounds)))


def test_sample_xyzw_and_signed_rolling_relation():
    import torch
    from types import SimpleNamespace as NS
    from scipy.spatial.transform import Rotation
    p=torch.tensor([[0.,0.,.0065]],dtype=torch.float64)
    R=Rotation.from_euler('y',90,degrees=True)
    q=torch.tensor(R.as_quat()[None],dtype=torch.float64)
    data=NS(root_link_pos_w=NS(torch=p),root_link_quat_w=NS(torch=q))
    contact=NS(data=NS(net_forces_w=NS(torch=torch.tensor([[[0.,0.,.05]]]))))
    term=NS(capsule=NS(data=data),_env=NS(scene={'capsule_contact':contact}),source=0,
            pose=lambda body:(np.array([0.,0.,.2]),np.eye(3)))
    cfg={'magnets':{'main_cube':{'remanence_t':1.45,'dimensions_m':[.05]*3},
        'target_cylinder':{'remanence_t':1.45,'diameter_m':.009,'height_m':.008,'center_offset_axis_m':-.004}},
        'external_magnet':{'capsule':{'total_mass_kg':.005735}}}
    ref=NS(vertices_world=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0.]]),
           triangles=np.array([[0,1,2],[0,2,3]]))
    probe=module.SurfaceRollProbe(term,NS(config=cfg),ref)
    first=probe.sample()
    assert first['tilt_from_tangent_deg']<1e-6
    p[0,1]=-.0065*.1
    q[:]=torch.tensor((Rotation.from_rotvec([.1,0,0])*R).as_quat()[None])
    second=probe.sample()
    assert abs(second['integrated_long_axis_roll_rad']-.1)<1e-10
    assert abs(second['signed_no_slip_residual_m'])<1e-10
    assert second['contact_force_N']>0
