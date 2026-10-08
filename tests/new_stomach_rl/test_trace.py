import pytest
import torch
from types import SimpleNamespace as S
from robotarm_magnetic_lab.runtime.new_stomach_rl_trace import PhysicalStepRecorder


def test_trace_reads_every_completed_step_not_endpoint_interpolation():
    capsule = S(data=S(root_link_pose_w=torch.zeros((1,7)),root_com_vel_w=torch.zeros((1,6))))
    robot = S(data=S(body_pos_w=torch.zeros((1,1,3)),body_quat_w=torch.zeros((1,1,4)),joint_pos=torch.zeros((1,9))))
    term = S(robot=robot,source=0,ids=list(range(9)),_target=torch.zeros((1,9)))
    bridge = S(state={'wrench':torch.zeros((1,12))})
    trace = PhysicalStepRecorder(capsule,term,bridge,'cpu')
    trace.begin()
    for tick in range(240):
        capsule.data.root_link_pose_w[0,0] = (tick+1)**2*.000001
        bridge.state['wrench'][0,0] = tick
        trace.append()
    data = trace.finish()
    assert data.shape == (240,50)
    assert data[0,0] == pytest.approx(.000001)
    assert data[100,0] == pytest.approx(101**2*.000001)
    assert data[-1,13] == 239
    trace.begin()
    with pytest.raises(RuntimeError,match='240'):
        trace.finish()
    assert data[-1,13] == 239  # last episode record not aliased to reusable buffer


def test_trace_counter_and_nonfinite_fail_closed():
    capsule = S(data=S(root_link_pose_w=torch.zeros((1,7)),root_com_vel_w=torch.zeros((1,6))))
    term = S(robot=S(data=S(body_pos_w=torch.zeros((1,1,3)),body_quat_w=torch.zeros((1,1,4)),
        joint_pos=torch.zeros((1,9)))),source=0,ids=list(range(9)),_target=torch.zeros((1,9)))
    trace = PhysicalStepRecorder(capsule,term,S(state={'wrench':torch.zeros((1,12))}),'cpu')
    trace.begin()
    for _ in range(240):
        trace.append()
    with pytest.raises(RuntimeError,match='240'):
        trace.append()
    trace.buffer[0,0] = float('nan')
    with pytest.raises(RuntimeError,match='nonfinite'):
        trace.finish()
