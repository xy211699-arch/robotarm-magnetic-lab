"""Read-only per-physical-step tape. Called after real scene physics updates."""
import torch


def tensor(value):
    return getattr(value,'torch',value)


class PhysicalStepRecorder:
    columns = ('capsule_pose_xyzw_7','capsule_velocity_6','applied_robot_capsule_wrench_12',
               'source_pose_xyzw_7','joint_actual_9','joint_issued_reference_9')

    def __init__(self,capsule,term,bridge,device):
        self.capsule,self.term,self.bridge = capsule,term,bridge
        with torch.inference_mode(False):
            self.buffer = torch.empty((240,50),device=device)
        self.count = 0

    def begin(self):
        self.count = 0

    def append(self):
        if self.count >= 240:
            raise RuntimeError('more than 240 physical scene updates in one policy step')
        data = self.term.robot.data
        row = torch.cat((tensor(self.capsule.data.root_link_pose_w)[0],
            tensor(self.capsule.data.root_com_vel_w)[0],self.bridge.state['wrench'][0],
            tensor(data.body_pos_w)[0,self.term.source],tensor(data.body_quat_w)[0,self.term.source],
            tensor(data.joint_pos)[0,self.term.ids],self.term._target[0]))
        self.buffer[self.count].copy_(row)
        self.count += 1

    def finish(self):
        if self.count != 240:
            raise RuntimeError('exactly 240 physical scene updates required')
        if not torch.isfinite(self.buffer).all():
            raise RuntimeError('nonfinite physical trace')
        return self.buffer.clone()
