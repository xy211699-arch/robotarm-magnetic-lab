"""Two real observer cameras and view-local external capsule annotation.

No materials, colliders, robot states, or policy images are edited. External
red marker is deliberately X-ray style when the capsule is behind stomach.
"""
import json
from pathlib import Path
import numpy as np
import cv2
import imageio_ffmpeg
from scipy.spatial.transform import Rotation


def configure_observers(cfg):
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils
    for name in ('observer_external','observer_internal'):
        setattr(cfg.scene,name,CameraCfg(prim_path='{ENV_REGEX_NS}/'+name,
            update_period=.1,width=960,height=540,data_types=['rgb'],update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=36.,
                clipping_range=(.0001 if name.endswith('internal') else .005,10.))))


def project_point(point,eye,quat_ros,intrinsic):
    p=Rotation.from_quat(quat_ros).as_matrix().T@(np.asarray(point)-eye)
    if p[2]<=0: return None
    uv=np.asarray(intrinsic)@p
    return uv[:2]/uv[2]


class DualObserver:
    def __init__(self,env,output,raycaster):
        self.env=env;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.raycaster=raycaster
        self.external=env.scene['observer_external'];self.internal=env.scene['observer_internal']
        self.capsule=env.scene['capsule'];self.robot=env.scene['robot']
        self.last_images={};self.frames=0
        cap=self.capsule.data.root_link_pos_w.torch[0].cpu().numpy()
        # Same fixed interior eye for every strategy; reject obstructed rays.
        self.internal_eye=None
        for offset in ([.010,0,.04],[-.010,0,.04],[0,.02,.03],[0,-.02,.03],[.02,0,.02]):
            eye=cap+offset
            distance=np.linalg.norm(eye-cap)
            hits,_=raycaster.query(cap,eye[None])
            directions=np.r_[np.eye(3),-np.eye(3)]
            nearby,_=raycaster.query(eye,eye[None]+directions*.002)
            if hits[0]>distance+.001 and np.all(nearby>.001):
                self.internal_eye=eye;break
        if self.internal_eye is None: raise RuntimeError('no unobstructed interior observer eye near initial capsule')
        baseid=self.robot.data.body_names.index('base_link')
        base=self.robot.data.body_pos_w.torch[0,baseid].cpu().numpy()
        self.external_target=(base+cap)/2+np.array([0,0,.2])
        self.external_eye=self.external_target+np.array([.9,-1.25,.65])
        self.writers={}
        for name in ('external','internal'):
            writer=imageio_ffmpeg.write_frames(str(self.output/(name+'.mp4')),(960,540),fps=10,
                codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',macro_block_size=2,
                output_params=['-crf','23','-preset','fast','-movflags','+frag_keyframe+empty_moov'])
            writer.send(None);self.writers[name]=writer
        (self.output/'camera_setup.json').write_text(json.dumps(dict(
            internal_eye_world_m=self.internal_eye.tolist(),external_eye_world_m=self.external_eye.tolist(),
            external_target_world_m=self.external_target.tolist(),width=960,height=540,fps=10,
            highlight='external-only projected red x-ray marker; no scene material edit'),indent=2))
        self.pose()

    def pose(self):
        import torch
        cap=self.capsule.data.root_link_pos_w.torch[0].cpu().numpy()
        def place(camera,eye,target):
            camera.set_world_poses_from_view(torch.tensor(eye[None],device=self.env.device,dtype=torch.float32),
                torch.tensor(target[None],device=self.env.device,dtype=torch.float32))
        place(self.external,self.external_eye,self.external_target)
        place(self.internal,self.internal_eye,cap)

    def capture(self,time_s,coverage=None):
        cap=self.capsule.data.root_link_pos_w.torch[0].cpu().numpy()
        for name,camera in (('external',self.external),('internal',self.internal)):
            rgb=camera.data.output['rgb'].torch[0,...,:3].cpu().numpy().copy()
            if rgb.dtype!=np.uint8: rgb=np.clip(rgb*255,0,255).astype(np.uint8)
            if name=='external':
                uv=project_point(cap,camera.data.pos_w.torch[0].cpu().numpy(),
                    camera.data.quat_w_ros.torch[0].cpu().numpy(),camera.data.intrinsic_matrices.torch[0].cpu().numpy())
                if uv is not None and 0<=uv[0]<960 and 0<=uv[1]<540:
                    pt=tuple(np.rint(uv).astype(int))
                    cv2.circle(rgb,pt,9,(255,0,0),-1)
                    cv2.circle(rgb,pt,12,(255,255,255),1)
                    cv2.putText(rgb,'CAPSULE / X-RAY MARKER',(max(pt[0]-80,0),max(pt[1]-18,20)),
                        cv2.FONT_HERSHEY_SIMPLEX,.4,(255,40,40),1)
            label=f'{name.upper()}  t={time_s:.1f}s'
            if coverage is not None:
                label+=f'  coverage={coverage*100:.2f}%'
            cv2.putText(rgb,label,
                (12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            self.writers[name].send(np.ascontiguousarray(rgb))
            self.last_images[name]=rgb
        self.frames+=1

    def snapshot(self,stem):
        for name,rgb in self.last_images.items():
            cv2.imwrite(str(self.output/f'{stem}_{name}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))

    def close(self):
        for writer in self.writers.values(): writer.close()
