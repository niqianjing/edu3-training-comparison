"""Edu3-Flat: MiniFlat-style phase RL on EDU3 nqj13, without reference gait."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.tasks.direct.base import BaseEnvCfg, CommandRangesCfg, RewardCfg, SceneCfg, mdp

from . import mdp as edu3_mdp
from .gait_params import (
    ANKLE_HEIGHT_OFFSET,
    CMD_ACTIVE_THRESHOLD,
    DEFAULT_WALK_FOOT_LIFT,
    FEET_SWING_HEIGHT_SUCCESS_MIN,
    GAIT_PHASE_OFFSET,
    GAIT_STANCE_DUTY,
    HIP_YAW_DEADZONE,
    LIN_VEL_STALL_MIN_RATIO,
    LIN_VEL_SUCCESS_ERROR,
    LIN_VEL_TRACK_STD,
    LIN_VEL_X_MAX,
    LIN_VEL_X_MIN,
    STEP_CYCLE_PERIOD,
    SWING_KNEE_STD_RAD,
    SWING_KNEE_TARGET_RAD,
)
from .robot_cfg import EDU3_NQJ13_TRAINABLE_CFG

# Body / joint name patterns for EDU3 (no ``_leg_`` prefix, no head).
_FEET = ".*_ankle_roll.*"
_KNEES = ".*_knee.*"
_HIP_YAW = ".*_thigh_yaw.*"
_HIP_ROLL = ".*_thigh_roll.*"
_HIP_PITCH = ".*_thigh_pitch.*"
_ANKLE_PITCH = ".*_ankle_pitch.*"
_ANKLE_ROLL = ".*_ankle_roll.*"


@configclass
class Edu3FlatRewardCfg(RewardCfg):
    """Phase-conditioned locomotion rewards. No ``joint_reference_tracking``."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"std": LIN_VEL_TRACK_STD},
    )
    lin_vel_x_stall = RewTerm(
        func=mdp.lin_vel_x_stall_penalty,
        weight=-2.0,
        params={"min_ratio": LIN_VEL_STALL_MIN_RATIO, "cmd_threshold": CMD_ACTIVE_THRESHOLD},
    )
    # Cap overspeed so policies stop braking by leaning the torso back.
    lin_vel_x_overspeed = RewTerm(
        func=mdp.lin_vel_x_overspeed_penalty,
        weight=-2.0,
        params={"max_ratio": 1.0, "cmd_threshold": CMD_ACTIVE_THRESHOLD},
    )
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2e-2)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=f"(?!{_FEET}).*")},
    )
    # Keep torso upright: roll+yaw tilt via flat_orientation; pitch both ways via base_pitch.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    base_pitch = RewTerm(func=edu3_mdp.base_pitch_l2, weight=-4.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_phase_contact = RewTerm(
        func=mdp.feet_phase_contact_xnor,
        weight=1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=_FEET)},
    )
    feet_swing_contact = RewTerm(
        func=mdp.feet_swing_contact_penalty,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=_FEET)},
    )
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height_penalty,
        # Slightly lower so clearance is earned via knee flex, not yaw sweep.
        weight=-35.0,
        params={
            "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
            "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
            "target_height": DEFAULT_WALK_FOOT_LIFT,
            "ankle_height": ANKLE_HEIGHT_OFFSET,
        },
    )
    swing_knee_flexion = RewTerm(
        func=edu3_mdp.swing_knee_flexion,
        weight=2.0,
        params={
            "target_flex_rad": SWING_KNEE_TARGET_RAD,
            "std": SWING_KNEE_STD_RAD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
        },
    )
    swing_hip_yaw = RewTerm(
        func=edu3_mdp.swing_hip_yaw_l2,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[_HIP_YAW])},
    )
    feet_contact_no_vel = RewTerm(
        func=mdp.feet_contact_no_vel,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=_FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FEET),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=_FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FEET),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=_FEET),
            "threshold": 60,
            "max_reward": 100,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[_FEET]), "min": 0.14, "max": 0.50},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[_KNEES]), "min": 0.10, "max": 0.35},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[_FEET])},
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[_FEET])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_deviation_hip_yaw = RewTerm(
        func=edu3_mdp.hip_yaw_excess_l2,
        weight=-6.0,
        params={
            "deadzone_rad": HIP_YAW_DEADZONE,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[_HIP_YAW]),
        },
    )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.hip_pos_l2,
        weight=-1.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[_HIP_ROLL])},
    )
    joint_deviation_upper_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["torso_joint", ".*_arm_roll_joint", ".*_arm_yaw_joint", ".*_elbow_pitch_joint"],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=edu3_mdp.arm_default_when_standing,
        weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm_pitch_joint"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[_HIP_PITCH, _KNEES, _ANKLE_PITCH, _ANKLE_ROLL],
            )
        },
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.4,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[_FEET])},
    )
    single_foot_stance_without_cmd = RewTerm(
        func=mdp.single_foot_stance_without_cmd,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[_FEET])},
    )
    upward = RewTerm(func=mdp.upward, weight=0.8)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.3,
        params={
            "pos_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*", "torso_joint"],
            ),
            "vel_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm.*", ".*_thigh.*", ".*_knee.*", ".*_ankle.*", "torso_joint"],
            ),
            "pos_weight": 1.0,
            "vel_weight": 0.04,
            "body_vel_threshold": 0.15,
        },
    )


@configclass
class Edu3FlatEnvCfg(BaseEnvCfg):
    reward = Edu3FlatRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # 21 DoF + phase(2): actor 3+3+3+2+21+21+21 = 74
        # critic ≈ 74 + lin_vel(3) + contact(2) + force(6) + air(2) + height(2) + acc(21) + torque(21) = 131
        self.action_space = 21
        self.observation_space = 74
        self.state_space = 131
        self.scene_context.robot = EDU3_NQJ13_TRAINABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.terrain_type = "plane"
        self.scene_context.terrain_generator = None
        self.scene_context.height_scanner.enable_height_scan = False
        # Base SceneCfg already targets left/right_ankle_roll_link (EDU3 names).
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )

        self.robot.terminate_contacts_body_names = [
            "base_link",
            ".*_thigh_yaw.*",
            ".*_thigh_roll.*",
        ]
        self.robot.feet_body_names = [_FEET]
        self.robot.terminate_base_height = None

        self.commands.resampling_time_range = (10.0, 10.0)
        self.commands.rel_standing_envs = 0.15
        self.commands.rel_heading_envs = 0.0
        self.commands.heading_command = False
        self.commands.debug_vis = True
        self.commands.ranges = CommandRangesCfg(
            lin_vel_x=(LIN_VEL_X_MIN, LIN_VEL_X_MAX),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.scale_link_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.push_robot = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints = None

        self.robot.action_scale = 0.25
        self.robot.cmd_active_threshold = CMD_ACTIVE_THRESHOLD
        self.robot.lin_vel_success_error = LIN_VEL_SUCCESS_ERROR
        self.robot.debug_gait_metrics_strict = True
        self.robot.debug_feet_height_min_m = FEET_SWING_HEIGHT_SUCCESS_MIN
        self.robot.gait_phase_period = STEP_CYCLE_PERIOD
        self.robot.gait_phase_offset = GAIT_PHASE_OFFSET
        self.robot.gait_phase_duty = GAIT_STANCE_DUTY
        # Overspeed penalty fully on from iter 0 (lean-back was already appearing).
        self.robot.overspeed_curriculum_start_iter = 0
        self.robot.overspeed_curriculum_end_iter = 0
        self.robot.actor_obs_history_length = 10
        self.robot.critic_obs_history_length = 10
        self.normalization.clip_actions = 1.0
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03
