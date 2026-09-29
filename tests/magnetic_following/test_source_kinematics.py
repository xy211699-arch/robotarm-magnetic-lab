import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def test_serial_composition_matches_two_link_forward_kinematics():
    path = Path(__file__).resolve().parents[2]/'source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/source_kinematics.py'
    spec = importlib.util.spec_from_file_location('source_fk',path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    q = np.array([.6,-.3])
    def fk(q):
        return np.array([np.cos(q[0])+.7*np.cos(sum(q)),np.sin(q[0])+.7*np.sin(sum(q)),0.])
    position = fk(q)
    rotation = Rotation.from_euler('z',sum(q)).as_matrix()
    z = np.array([0.,0.,1.])
    second = np.array([np.cos(q[0]),np.sin(q[0]),0])
    jac = np.column_stack([np.r_[np.cross(z,position),z],np.r_[np.cross(z,position-second),z]])
    delta = np.random.default_rng(913).uniform(-1,1,(50,2))
    p,r = m.source_pose(position,rotation,jac,delta)
    for i,d in enumerate(delta):
        np.testing.assert_allclose(p[i],fk(q+d),atol=1e-12)
        np.testing.assert_allclose(r[i],Rotation.from_euler('z',sum(q+d)).as_matrix(),atol=1e-12)
