"""Single-environment Isaac Lab task for the AUBO arm and magnetic ball assembly.

This first migration stage deliberately keeps the validated USD scene intact.
The complete stage is referenced below an environment namespace, while Isaac Lab
binds an :class:`Articulation` view to the existing robot/ASM articulation.
Later stages can split the stomach, camera and capsule into independently
randomized assets without changing the policy-facing joint interface.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from . import mdp


SCENE_USD_PATH = "/mnt/isaac-linux/robotarm_magnetic_lab/assets/robotarm_magnetic_training.usda"
ROBOT_ROOT_POS = (0.7483759734984385, 0.47921891913436365, 0.0)
# Directly below the settled magl center at the validated arm target.
CAPSULE_START_POS = (1.0608155, 0.1145374, 0.0065)
# Horizontal capsule yaw aligned with the horizontal main-magnet axis at
# BALL_X=+90 deg.  The resulting force points into the ground and the initial
# magnetic torque is minimized, providing a deterministic resting reset.
CAPSULE_START_ROT = (
    -0.2565394892869615,
    0.6589290481048662,
    0.2565394892869615,
    0.6589290481048662,
)
ARM_JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
BALL_JOINT_NAMES = ["ballxj", "ballyj", "ballzj"]
# Capsule camera provisional optical model.
#
# Hardware identification supplied by the project: DS01 camera using a
# VeriSilicon CX93510-series SoC. No authoritative public intrinsic/distortion
# calibration was found, so the stated 120-degree FOV is treated as the full
# diameter of the circular optical field until calibration data is available.
CAPSULE_CAMERA_MODEL = "DS01 / VeriSilicon CX93510-series"
CAPSULE_CAMERA_WIDTH = 1280
CAPSULE_CAMERA_HEIGHT = 720
CAPSULE_CAMERA_CIRCULAR_FOV_DEG = 120.0
CAPSULE_CAMERA_HORIZONTAL_APERTURE = 20.955
# The 120-degree FOV is measured across the circular image, whose diameter is
# the 720-pixel image height. The 16:9 RTX source therefore renders a wider
# horizontal pinhole image before the equidistant circular remap.
CAPSULE_CAMERA_FOCAL_LENGTH = (
    CAPSULE_CAMERA_HORIZONTAL_APERTURE
    * CAPSULE_CAMERA_HEIGHT
    / (
        2.0
        * CAPSULE_CAMERA_WIDTH
        * math.tan(math.radians(CAPSULE_CAMERA_CIRCULAR_FOV_DEG * 0.5))
    )
)
START_JOINT_POS = {
    # Latest collision-validated 5-D solution from the Isaac Sim planner.
    # j6 is intentionally free and selected to keep the mounted ASM clear of
    # the arm; forcing it back to zero recreates the visual interference.
    "j1": -1.115113,
    "j2": 0.014145,
    "j3": -1.863804,
    "j4": 0.309711,
    "j5": 1.577784,
    "j6": -0.28688,
    "ballxj": 1.5707963267948966,
    "ballyj": 0.0,
    "ballzj": 0.0,
}


@configclass
class RobotarmMagneticLabSceneCfg(InteractiveSceneCfg):
    """The saved robot/ASM scene, namespaced as one Isaac Lab environment."""

    # Spawn the complete, already validated stage. Its default prim is /World,
    # therefore robotarm, asm, ground and target_magnet become children of Scene.
    scene_asset = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path=SCENE_USD_PATH),
    )

    # The saved stage has one articulation root on the robot base. The fixed
    # l6-to-ASM mount folds ballxj/ballyj/ballzj into this same articulation.
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Scene/robotarm",
        spawn=None,
        articulation_root_prim_path="/Geometry/world",
        init_state=ArticulationCfg.InitialStateCfg(
            # The referenced stage was authored in this world frame.  Omitting
            # this pose makes reset_scene_to_default teleport the articulation
            # to the environment origin while the capsule/field stay in the
            # saved stage frame.
            pos=ROBOT_ROOT_POS,
            joint_pos=START_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                effort_limit_sim=500.0,
                velocity_limit_sim=1.5,
                stiffness=3000.0,
                damping=300.0,
            ),
            "ball": ImplicitActuatorCfg(
                joint_names_expr=BALL_JOINT_NAMES,
                effort_limit_sim=10.0,
                # Keep commanded magnet motion below 1 rad/s.  The capsule is
                # not actuated; its angular speed remains a passive result of
                # magnetic torque, gravity and contact.
                velocity_limit_sim=0.8,
                stiffness=300.0,
                damping=30.0,
            ),
        },
    )

    # Register the already-authored dynamic capsule with Isaac Lab so every
    # reset restores the validated position below the goal tool pose.  Without
    # this view, reset_scene_to_default only resets the robot and the capsule
    # remains wherever a previous physics run left it.
    capsule = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet",
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CAPSULE_START_POS,
            rot=CAPSULE_START_ROT,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # Provisional DS01 capsule-view camera. The lower-stomach reset places the
    # local +Z end against the wall, so mounting the camera there renders the
    # wall back face at about 1 mm and produces a black image. The optical
    # assembly is therefore mounted at the local -Z end and rotated 180 degrees
    # about local Y to look outward through that end cap.
    capsule_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet/capsule_camera",
        update_period=1.0 / 30.0,
        width=CAPSULE_CAMERA_WIDTH,
        height=CAPSULE_CAMERA_HEIGHT,
        data_types=["rgb", "distance_to_camera"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAPSULE_CAMERA_FOCAL_LENGTH,
            horizontal_aperture=CAPSULE_CAMERA_HORIZONTAL_APERTURE,
            # Capsule endoscopy is a close-range modality. Keep enough range
            # for scene bring-up while resolving tissue a few millimetres away.
            focus_distance=0.03,
            # The former 2 mm near plane clipped the wall during close
            # inspection. A 0.1 mm near plane preserves a flush optical
            # window's close-range tissue rendering
            # rendering without materially reducing depth precision over the
            # 0.30 m endoscopy range.
            clipping_range=(0.0001, 0.30),
        ),
        offset=CameraCfg.OffsetCfg(
            # The procedural capsule spans local Z=[-12.5, +12.5] mm. The
            # former -11.5 mm optical center sat 1 mm inside its opaque end
            # cap, so the camera rendered a hard grey disk. Put the optical
            # center 0.2 mm outside the end surface to model a flush clear
            # window without changing the externally visible capsule mesh.
            pos=(0.0, 0.0, -0.0127),
            rot=(0.0, 1.0, 0.0, 0.0),
            convention="ros",
        ),
    )

    # Created only when ``--capsule_camera_view`` is requested. This keeps the
    # policy/data camera at its task-defined acquisition rate (1 Hz in the
    # stomach task), while allowing a separate 30 Hz engineering preview.
    capsule_camera_preview = None

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # Four close-range white LEDs arranged around the optical axis. Public
    # magnetically-driven capsule designs use this layout; the lights inherit
    # the dynamic capsule transform and illuminate the stomach interior.
    capsule_led_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet/capsule_led_top",
        spawn=sim_utils.SphereLightCfg(
            radius=0.0008,
            intensity=250.0,
            normalize=True,
            enable_color_temperature=True,
            color_temperature=5600.0,
        ),
        # Keep the LED behind the optical center. At the lower stomach wall
        # the visible surface can be only 1--2 mm away; the former -18 mm
        # location placed the light behind the tissue and rendered it black.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0055, -0.0126)),
    )
    capsule_led_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet/capsule_led_bottom",
        spawn=sim_utils.SphereLightCfg(
            radius=0.0008,
            intensity=250.0,
            normalize=True,
            enable_color_temperature=True,
            color_temperature=5600.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.0055, -0.0126)),
    )
    capsule_led_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet/capsule_led_left",
        spawn=sim_utils.SphereLightCfg(
            radius=0.0008,
            intensity=250.0,
            normalize=True,
            enable_color_temperature=True,
            color_temperature=5600.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.0055, 0.0, -0.0126)),
    )
    capsule_led_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/target_magnet/capsule_led_right",
        spawn=sim_utils.SphereLightCfg(
            radius=0.0008,
            intensity=250.0,
            normalize=True,
            enable_color_temperature=True,
            color_temperature=5600.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0055, 0.0, -0.0126)),
    )

    # Visual-only fiducials on the temporary bench plane. They give the
    # capsule camera scale, direction and rotation cues without affecting
    # collision. Replace them with stomach texture/vascular geometry later.
    fiducial_red = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/fiducials/red",
        spawn=sim_utils.CuboidCfg(
            size=(0.014, 0.014, 0.0002),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.03, 0.03),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.088, 0.140, 0.0001)),
    )
    fiducial_green = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/fiducials/green",
        spawn=sim_utils.CuboidCfg(
            size=(0.010, 0.022, 0.0002),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.03, 0.75, 0.08),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.112, 0.162, 0.0001)),
    )
    fiducial_blue = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/fiducials/blue",
        spawn=sim_utils.CuboidCfg(
            size=(0.024, 0.006, 0.0002),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.03, 0.15, 0.9),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.138, 0.188, 0.0001)),
    )

    # The capsule camera is almost parallel to the bench at reset, so flat
    # decals are viewed edge-on. Three raised, asymmetric landmarks provide
    # unmistakable optical-flow and roll cues in the circular image.
    landmark_red = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/fiducials/red_post",
        spawn=sim_utils.CuboidCfg(
            size=(0.008, 0.008, 0.015),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.01, 0.01),
                emissive_color=(0.18, 0.0, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.1124, 0.1619, 0.0075)),
    )
    landmark_green = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/fiducials/green_post",
        spawn=sim_utils.CuboidCfg(
            size=(0.006, 0.012, 0.025),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.01, 0.90, 0.04),
                emissive_color=(0.0, 0.15, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.1136, 0.1901, 0.0125)),
    )
    landmark_blue = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene/MagneticDemo/fiducials/blue_post",
        spawn=sim_utils.CuboidCfg(
            size=(0.014, 0.005, 0.010),
            collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.01, 0.08, 0.95),
                emissive_color=(0.0, 0.0, 0.15),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.1553, 0.1742, 0.005)),
    )


@configclass
class ActionsCfg:
    """Nine normalized absolute joint-position offsets about the reset pose.

    The first six dimensions command the arm and the last three command the
    magnetic ball. These values are not integrated deltas. With
    ``use_default_offset=True``, every policy action maps directly to the reset
    pose plus its configured scale, and zero holds the validated initialization
    pose. This exact meaning is frozen in interface schema 1.0.0.
    """

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES + BALL_JOINT_NAMES,
        scale={
            "j[1-6]": 0.05,
            # A normalized ball command spans +/-90 degrees about the reset
            # pose. The previous 0.08-rad range could not create the requested
            # 45-degree-or-greater magnetic-axis tilt.
            "ball.*j": math.pi / 2.0,
        },
        use_default_offset=True,
        preserve_order=True,
    )
    # Zero-dimensional internal action term. It does not change the policy
    # interface; its apply hook refreshes the analytical magnetic wrench at
    # every 240 Hz PhysX substep while policy commands remain at 20 Hz.
    magnetic_physics = mdp.MagneticPhysicsActionCfg(asset_name="capsule")


@configclass
class ObservationsCfg:
    """Low-dimensional observations used to validate the control interface."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + BALL_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + BALL_JOINT_NAMES)},
        )
        magnetic_wrench = ObsTerm(func=mdp.magnetic_wrench)
        asm_clearance = ObsTerm(func=mdp.asm_clearance)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class VisionCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.capsule_rgb,
            params={
                "sensor_cfg": SceneEntityCfg("capsule_camera"),
                "field_of_view_deg": CAPSULE_CAMERA_CIRCULAR_FOV_DEG,
            },
        )
        depth = ObsTerm(
            func=mdp.capsule_depth,
            params={
                "sensor_cfg": SceneEntityCfg("capsule_camera"),
                "field_of_view_deg": CAPSULE_CAMERA_CIRCULAR_FOV_DEG,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            # Keep RGB and depth as image tensors instead of flattening them
            # into the low-dimensional policy state.
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    vision: VisionCfg = VisionCfg()


@configclass
class EventCfg:
    """Reset state plus legacy magnetic/collision runtime updates."""

    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    magnetic_collision_bridge = EventTerm(
        func=mdp.LegacyMagneticCollisionBridge,
        mode="interval",
        interval_range_s=(0.05, 0.05),
        is_global_time=False,
        resample_interval_on_reset=False,
    )


@configclass
class RewardsCfg:
    """Neutral bring-up rewards; task rewards are added after observations work."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-3)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + BALL_JOINT_NAMES)},
    )
    collision = RewTerm(func=mdp.collision_penalty, weight=-10.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(func=mdp.collision_detected)


@configclass
class RobotarmMagneticLabEnvCfg(ManagerBasedRLEnvCfg):
    """Single environment at 240 Hz physics and 20 Hz policy rate."""

    scene: RobotarmMagneticLabSceneCfg = RobotarmMagneticLabSceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 12
        self.episode_length_s = 30.0
        self.viewer.eye = (1.8, 1.6, 1.35)
        self.viewer.lookat = (0.9, 0.25, 0.45)
        self.sim.dt = 1.0 / 240.0
        # Keep Fabric enabled: GPU PhysX writes the live articulation transforms
        # to Fabric for the viewport. The project launcher disables experimental
        # UJITSO geometry streaming, which otherwise conflicts with the Fabric
        # render delegate and can leave referenced robot meshes at stale poses.
        self.sim.use_fabric = True
        # Render at 30 Hz for the capsule camera while the policy/action loop
        # remains at 20 Hz.
        self.sim.render_interval = 8
