"""Student 21-joint reference-motion task driven by the single-source contract."""

from isaaclab.utils import configclass

from mimic_real.assets.usd.edu3_reference import EDU3_REFERENCE_25_10_CFG
from mimic_real.assets.usd.edu3_reference.contract import CONTRACT
from .hi_mimic_capture_rsi_config import HIMimicCaptureRSIAgentCfg, HIMimicCaptureRSIEnvCfg


@configclass
class EDU3ReferenceMimicEnvCfg(HIMimicCaptureRSIEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        control = CONTRACT["control"]
        robot = CONTRACT["robot"]
        self.scene.robot = EDU3_REFERENCE_25_10_CFG
        self.motion_data.motion_file_path = CONTRACT["provenance"]["motion"]["path"]
        self.motion_data.kinematics_urdf_path = CONTRACT["provenance"]["urdf"]["path"]
        self.motion_data.local_root_marker_name = robot["root_link"]
        self.motion_data.use_local_capture_points = False
        self.robot.actor_obs_history_length = int(control["actor_history_frames"])
        self.robot.critic_obs_history_length = int(control["actor_history_frames"])
        self.robot.feet_body_names = list(robot["feet_body_patterns"])
        self.robot.base_link_body_names = [robot["root_link"]]
        self.terminate.terminate_contacts_body_names = [
            "base_link", ".*_arm_pitch_link", ".*_arm_yaw_link", ".*_elbow_pitch_link"
        ]
        self.domain_rand.add_rigid_body_mass.params["body_names"] = robot["root_link"]
        self.sim.dt = float(control["isaac_physics_dt_s"])
        self.decimation = int(control["decimation"])
        self.robot.action_scale = float(control["action_scale_rad"])
        delay = control["environment_action_delay_policy_steps"]
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {"min_delay": int(delay[0]), "max_delay": int(delay[1])}
        randomization = CONTRACT["training_randomization"]
        noise = randomization["observation_noise"]
        self.noise.add_noise = bool(noise["enabled"])
        self.noise.noise_scales.ang_vel = float(noise["angular_velocity"])
        self.noise.noise_scales.projected_gravity = float(noise["projected_gravity"])
        self.noise.noise_scales.joint_pos = float(noise["joint_position"])
        self.noise.noise_scales.joint_vel = float(noise["joint_velocity"])
        self.domain_rand.reset_robot_joints.params["position_range"] = tuple(randomization["reset_joint_position_offset_rad"])
        self.domain_rand.reset_robot_joints.params["velocity_range"] = tuple(randomization["reset_joint_velocity_rad_s"])
        self.domain_rand.reset_robot_base.params["pose_range"]["z"] = tuple(randomization["reset_root_z_offset_m"])
        material = randomization["contact_material"]
        self.domain_rand.randomize_robot_friction.enable = True
        self.domain_rand.randomize_robot_friction.params = {
            "static_friction_range": list(material["static_friction"]),
            "dynamic_friction_range": list(material["dynamic_friction"]),
            "restitution_range": list(material["restitution"]),
            "num_buckets": int(material["buckets"]),
        }
        self.domain_rand.add_rigid_body_mass.enable = True
        self.domain_rand.add_rigid_body_mass.params["mass_distribution_params"] = list(randomization["base_mass_additive_kg"])
        push = randomization["push"]
        self.domain_rand.push_robot.enable = bool(push["enabled"])
        self.domain_rand.push_robot.push_interval_s = float(push["interval_s"])
        self.domain_rand.push_robot.params["velocity_range"]["x"] = tuple(push["velocity_xy_m_s"])
        self.domain_rand.push_robot.params["velocity_range"]["y"] = tuple(push["velocity_xy_m_s"])


@configclass
class EDU3ReferenceMimicAgentCfg(HIMimicCaptureRSIAgentCfg):
    experiment_name = "edu3_reference_mimic_r1"
    wandb_project = "edu3_reference_mimic_r1"
