# env.py
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events


def _cube_inside_box(cube_pos, box_pos):
    """Shared logic: True where cube is geometrically inside box."""
    dx = torch.abs(cube_pos[:, 0] - box_pos[:, 0])
    dy = torch.abs(cube_pos[:, 1] - box_pos[:, 1])
    z_diff = cube_pos[:, 2] - box_pos[:, 2]
    # Box interior: 9cm half-width, z between -5cm (floor) and +15cm (rim)
    return (dx < 0.09) & (dy < 0.09) & (z_diff > -0.12) & (z_diff < 0.15)


def grasp_signal(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gripper_threshold: float = 0.05,
    proximity_threshold: float = 0.08,
) -> torch.Tensor:
    """1 when gripper is closed, near the cube, and cube is lifted off table."""
    from isaaclab.assets import Articulation, RigidObject
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[object_cfg.name]
    finger_indices, _ = robot.find_joints("panda_finger_joint.*")
    finger_pos = robot.data.joint_pos[:, finger_indices]
    gripper_width = finger_pos.sum(dim=1)
    ee_frame = env.scene["ee_frame"]
    eef_pos = ee_frame.data.target_pos_w[..., 0, :]
    cube_pos = cube.data.root_pos_w[:, :3]
    dist = torch.norm(eef_pos - cube_pos, dim=1)
    cube_lifted = cube.data.root_pos_w[:, 2] > 0.15
    return ((gripper_width < gripper_threshold) & (dist < proximity_threshold) & cube_lifted).float()


def place_signal(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """1 when cube is inside the box."""
    cube_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3]
    box_pos = env.scene[box_cfg.name].data.root_pos_w[:, :3]
    return _cube_inside_box(cube_pos, box_pos).float()


def cube_in_box(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Boolean termination — True when cube is placed inside box."""
    cube_pos = env.scene[object_cfg.name].data.root_pos_w[:, :3]
    box_pos = env.scene[box_cfg.name].data.root_pos_w[:, :3]
    return _cube_inside_box(cube_pos, box_pos)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=cube_in_box)


_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        cube_pos = ObsTerm(func=place_mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("object"), "return_key": "pos"})
        box_pos = ObsTerm(func=place_mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("box"), "return_key": "pos"})
        eef_pos = ObsTerm(func=stack_mdp.ee_frame_pose_in_base_frame, params={"return_key": "pos"})
        eef_quat = ObsTerm(func=stack_mdp.ee_frame_pose_in_base_frame, params={"return_key": "quat"})
        table_cam = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False})
        wrist_cam = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        wrist_cam_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False})
        table_cam_rgb = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskTermsCfg(ObsGroup):
        grasp = ObsTerm(func=grasp_signal)
        place = ObsTerm(func=place_signal)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy = PolicyCfg()
    rgb_camera = RGBCameraPolicyCfg()
    subtask_terms = SubtaskTermsCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=stack_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    init_cube_position = EventTerm(
        func=franka_stack_events.randomize_object_pose, mode="reset",
        params={"pose_range": {"x": (0.4, 0.5), "y": (-0.1, 0.1), "z": (0.055, 0.055), "yaw": (0.0, 0.0)}, "asset_cfgs": [SceneEntityCfg("object")]},
    )
    init_box_position = EventTerm(
        func=franka_stack_events.randomize_object_pose, mode="reset",
        params={"pose_range": {"x": (0.6, 0.7), "y": (0.15, 0.25), "z": (0.055, 0.055), "yaw": (0.0, 0.0)}, "asset_cfgs": [SceneEntityCfg("box")]},
    )


@configclass
class FrankaPlaceCubeIntoBoxEnvCfg_PLAY(LiftEnvCfg):
    observations = ObservationsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.viewer.debug_vis = False
        self.events = EventCfg()
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.object_pose.body_name = "panda_hand"

        # Disable command goal/current pose markers from LiftEnvCfg base class
        self.commands.object_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.0, 0.0, 0.0)
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.0, 0.0, 0.0)

        self.episode_length_s = 100.0
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.45, 0.0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd", scale=(0.8, 0.8, 0.8), rigid_props=_RIGID_PROPS),
        )
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.65, 0.2, 0.055], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Box/box.usd", scale=(0.8, 0.8, 0.8), rigid_props=_RIGID_PROPS),
        )

        # EE frame — debug_vis=False hides the axis gizmo (same pattern as official Isaac Lab franka tasks)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="end_effector", offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]))],
        )

        # Wrist camera — attached to panda_hand, update_period=0.0 ensures sensor buffer updates every frame
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros"
            ),
        )

        # Table camera — fixed overhead view, update_period=0.0 ensures sensor buffer updates every frame
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(2.0, 0.0, 1.2),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros"
            ),
        )

        self.sim.dt = 1 / 60
        self.decimation = 2

        # Camera rendering settings — matches Isaac Lab visuomotor stack env
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"