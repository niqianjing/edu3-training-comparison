"""A/B configs for training the original EDU3 body with Xiaohai's task."""

from gym.envs.hi_12.hi_controller_config import HiControllerCfg, HiControllerRunnerCfg


class Edu3XiaohaiBaseCfg(HiControllerCfg):
    class env(HiControllerCfg.env):
        num_envs = 4096
        episode_length_s = 5

    class sim(HiControllerCfg.sim):
        dt = 0.0025

    class init_state(HiControllerCfg.init_state):
        pos = [0.0, 0.0, 0.40]
        default_joint_angles = {
            "l_hip_pitch_joint": -0.10,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.30,
            "l_ankle_pitch_joint": -0.20,
            "l_ankle_roll_joint": 0.0,
            "r_hip_pitch_joint": 0.10,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": -0.30,
            "r_ankle_pitch_joint": 0.20,
            "r_ankle_roll_joint": 0.0,
        }
        dof_pos_range = {
            "l_hip_pitch_joint": [-1.5708, 0.7854],
            "l_hip_roll_joint": [-0.3491, 0.7854],
            "l_thigh_joint": [-0.7854, 0.7854],
            "l_calf_joint": [0.0, 1.2217],
            "l_ankle_pitch_joint": [-0.4363, 0.4363],
            "l_ankle_roll_joint": [-0.4363, 0.4363],
            "r_hip_pitch_joint": [-0.7854, 1.5708],
            "r_hip_roll_joint": [-0.7854, 0.3491],
            "r_thigh_joint": [-0.7854, 0.7854],
            "r_calf_joint": [-1.2217, 0.0],
            "r_ankle_pitch_joint": [-0.4363, 0.4363],
            "r_ankle_roll_joint": [-0.4363, 0.4363],
        }
        dof_vel_range = {name: [-0.1, 0.1] for name in default_joint_angles}

    class control(HiControllerCfg.control):
        decimation = 8
        actuation_scale = 1.0
        exp_avg_decay = None
        roll25 = False
        coulomb_vel_eps = 0.01
        enforce_right_left_feet = False
        stiffness = {
            "l_hip_pitch_joint": 100.0,
            "l_hip_roll_joint": 80.0,
            "l_thigh_joint": 80.0,
            "l_calf_joint": 100.0,
            "l_ankle_pitch_joint": 40.0,
            "l_ankle_roll_joint": 40.0,
            "r_hip_pitch_joint": 100.0,
            "r_hip_roll_joint": 80.0,
            "r_thigh_joint": 80.0,
            "r_calf_joint": 100.0,
            "r_ankle_pitch_joint": 40.0,
            "r_ankle_roll_joint": 40.0,
        }
        damping = {
            "l_hip_pitch_joint": 3.0,
            "l_hip_roll_joint": 2.5,
            "l_thigh_joint": 2.5,
            "l_calf_joint": 3.0,
            "l_ankle_pitch_joint": 1.5,
            "l_ankle_roll_joint": 1.5,
            "r_hip_pitch_joint": 3.0,
            "r_hip_roll_joint": 2.5,
            "r_thigh_joint": 2.5,
            "r_calf_joint": 3.0,
            "r_ankle_pitch_joint": 1.5,
            "r_ankle_roll_joint": 1.5,
        }

    class domain_rand(HiControllerCfg.domain_rand):
        # First prove the body/motor A/B in one nominal world. Robustness
        # randomization is added later, one measured source at a time.
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False
        randomize_com_displacement = False
        randomize_motor_strength = False
        randomize_Kp_factor = False
        randomize_Kd_factor = False
    class asset(HiControllerCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/edu3_xiaohai12/urdf/edu3_xiaohai12_original.urdf"
        foot_name = "ankle_roll"
        terminate_after_contacts_on = [
            "base_link", "l_hip_pitch_link", "l_hip_roll_link", "l_thigh_link",
            "l_calf_link", "l_ankle_pitch_link", "r_hip_pitch_link", "r_hip_roll_link",
            "r_thigh_link", "r_calf_link", "r_ankle_pitch_link",
        ]
        self_collisions = 1
        collapse_fixed_joints = True
        angular_damping = 0.0
        rotor_inertia = [
            0.02649, 0.0067367, 0.0067367, 0.02649, 0.0067367, 0.0067367,
            0.02649, 0.0067367, 0.0067367, 0.02649, 0.0067367, 0.0067367,
        ]

    class rewards(HiControllerCfg.rewards):
        base_height_target = 0.36
        base_height_range = 0.025
        min_dist_feet = 0.115
        max_dist_feet = 0.285


class Edu3XiaohaiOriginalCfg(Edu3XiaohaiBaseCfg):
    pass


class Edu3XiaohaiRoll25Cfg(Edu3XiaohaiBaseCfg):
    class control(Edu3XiaohaiBaseCfg.control):
        roll25 = True
        stiffness = dict(Edu3XiaohaiBaseCfg.control.stiffness)
        stiffness.update({"l_hip_roll_joint": 100.0, "r_hip_roll_joint": 100.0})
        damping = dict(Edu3XiaohaiBaseCfg.control.damping)
        damping.update({"l_hip_roll_joint": 3.0, "r_hip_roll_joint": 3.0})

    class asset(Edu3XiaohaiBaseCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/edu3_xiaohai12/urdf/edu3_xiaohai12_roll25.urdf"
        rotor_inertia = [
            0.02649, 0.02649, 0.0067367, 0.02649, 0.0067367, 0.0067367,
            0.02649, 0.02649, 0.0067367, 0.02649, 0.0067367, 0.0067367,
        ]


class Edu3XiaohaiOriginalRunnerCfg(HiControllerRunnerCfg):
    seed = 42

    class runner(HiControllerRunnerCfg.runner):
        max_iterations = 1001
        save_interval = 100
        experiment_name = "EDU3_Xiaohai12_OriginalMotors"
        run_name = "seed42"


class Edu3XiaohaiRoll25RunnerCfg(HiControllerRunnerCfg):
    seed = 42

    class runner(HiControllerRunnerCfg.runner):
        max_iterations = 1001
        save_interval = 100
        experiment_name = "EDU3_Xiaohai12_Roll25"
        run_name = "seed42"


class Edu3XiaohaiOriginalFeetFixCfg(Edu3XiaohaiOriginalCfg):
    class control(Edu3XiaohaiOriginalCfg.control):
        enforce_right_left_feet = True


class Edu3XiaohaiRoll25FeetFixCfg(Edu3XiaohaiRoll25Cfg):
    class control(Edu3XiaohaiRoll25Cfg.control):
        enforce_right_left_feet = True


class Edu3XiaohaiOriginalFeetFixRunnerCfg(HiControllerRunnerCfg):
    seed = 42

    class runner(HiControllerRunnerCfg.runner):
        max_iterations = 1001
        save_interval = 100
        experiment_name = "EDU3_Xiaohai12_OriginalMotors_FeetFix_R2"
        run_name = "seed42"


class Edu3XiaohaiRoll25FeetFixRunnerCfg(HiControllerRunnerCfg):
    seed = 42

    class runner(HiControllerRunnerCfg.runner):
        max_iterations = 1001
        save_interval = 100
        experiment_name = "EDU3_Xiaohai12_Roll25_FeetFix_R2"
        run_name = "seed42"

# R3 timing-only A/B. Rewards, body, motors, limits, contact geometry,
# default pose, and seed remain identical to the feet-semantics-fixed R2.
class Edu3XiaohaiTiming20msCfg(Edu3XiaohaiOriginalFeetFixCfg):
    """Deployable 50 Hz policy; preserve Xiaohai's 0.30 s half-step in seconds."""

    class commands(Edu3XiaohaiOriginalFeetFixCfg.commands):
        class ranges(Edu3XiaohaiOriginalFeetFixCfg.commands.ranges):
            sample_period = [15, 16]


class Edu3XiaohaiTiming10msCfg(Edu3XiaohaiOriginalFeetFixCfg):
    """Diagnostic 100 Hz policy; exact Xiaohai policy rate and 30-tick half-step."""

    class control(Edu3XiaohaiOriginalFeetFixCfg.control):
        decimation = 4


class Edu3XiaohaiTiming20msRunnerCfg(HiControllerRunnerCfg):
    seed = 42

    class runner(HiControllerRunnerCfg.runner):
        max_iterations = 1001
        save_interval = 100
        experiment_name = "EDU3_Xiaohai12_Timing20ms_Period15_R3"
        run_name = "seed42"


class Edu3XiaohaiTiming10msRunnerCfg(HiControllerRunnerCfg):
    seed = 42

    class runner(HiControllerRunnerCfg.runner):
        max_iterations = 1001
        save_interval = 100
        experiment_name = "EDU3_Xiaohai12_Timing10ms_Period30_R3"
        run_name = "seed42"

# R4: keep the proven 20 ms / 15-tick timing and change only hip-roll modules.
class Edu3XiaohaiRoll25Timing20msCfg(Edu3XiaohaiRoll25FeetFixCfg):
    class commands(Edu3XiaohaiRoll25FeetFixCfg.commands):
        class ranges(Edu3XiaohaiRoll25FeetFixCfg.commands.ranges):
            sample_period = [15, 16]


class Edu3XiaohaiRoll25Timing20msSeed42RunnerCfg(HiControllerRunnerCfg):
    seed = 42

    class runner(HiControllerRunnerCfg.runner):
        max_iterations = 1001
        save_interval = 100
        experiment_name = "EDU3_Xiaohai12_Roll25_Timing20ms_Period15_R4"
        run_name = "seed42"


class Edu3XiaohaiRoll25Timing20msSeed43RunnerCfg(HiControllerRunnerCfg):
    seed = 43

    class runner(HiControllerRunnerCfg.runner):
        max_iterations = 1001
        save_interval = 100
        experiment_name = "EDU3_Xiaohai12_Roll25_Timing20ms_Period15_R4"
        run_name = "seed43"