from mimic_real.utils import task_registry

import argparse

from isaaclab.app import AppLauncher
import mimic_real.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="Train Xiaohai with synchronized reference-state initialization.")
parser.add_argument("--task", type=str, default="hi_mimic_capture_rsi_portable_dcmotor")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint_path", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from mimic_real.agents.on_policy_runner import OnPolicyRunner
from mimic_real.envs import *  # noqa:F401,F403
from mimic_real.envs.mimic.hi_mimic_capture_rsi_env import HIMimicCaptureRSIEnv
from mimic_real.envs.mimic.hi_mimic_capture_rsi_config import (
    HIMimicCaptureRSIAgentCfg,
    HIMimicCaptureRSIEnvCfg,
)
from mimic_real.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.actuators import DCMotorCfg
import os
from datetime import datetime
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
class HIMimicPortableDCMotorEnv(HIMimicCaptureRSIEnv):
    """RSI task with portable target clamp and explicit DC torque-speed actuators."""

    def step(self, actions: torch.Tensor):
        self.pre_step_callback(actions)
        delayed_actions = self.action_buffer.compute(actions)
        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        target = (
            clipped_actions * self.action_scale
            + self.robot.data.default_joint_pos
            + self.motion_loader.get_dof_pos_batch(self.phase)
        )
        limits = self.robot.data.soft_joint_pos_limits
        target = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])
        for _ in range(self.cfg.sim.decimation):
            self.robot.set_joint_position_target(target)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        if not self.headless:
            self.sim.render()
        self.episode_length_buf += 1
        self.phase = self.episode_length_buf / self.max_episode_length
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.post_step_callback(actions)
        self.reset_buf, self.time_out_buf = self.check_reset()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras


def portable_actuators():
    common = dict(
        effort_limit_sim=1.0e9,
        velocity_limit=6.28,
        velocity_limit_sim=1000.0,
        stiffness=80.0,
        damping=1.0,
        armature=0.01,
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.1,
    )
    return {
        "portable_body24": DCMotorCfg(
            joint_names_expr=[
                "waist_joint", ".*_hip_pitch_joint", ".*_hip_roll_joint",
                ".*_thigh_joint", ".*_calf_joint",
                ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
            ],
            effort_limit=24.0,
            saturation_effort=24.0,
            **common,
        ),
        "portable_arms17": DCMotorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                ".*_upper_arm_joint", ".*_elbow_joint", ".*_wrist_joint",
            ],
            effort_limit=17.0,
            saturation_effort=17.0,
            **common,
        ),
    }

portable_agent_cfg = HIMimicCaptureRSIAgentCfg()
portable_agent_cfg.experiment_name = "hi_mimic_capture_rsi_portable_dcmotor"
portable_agent_cfg.wandb_project = "hi_mimic_capture_rsi_portable_dcmotor"
task_registry.register(
    "hi_mimic_capture_rsi_portable_dcmotor",
    HIMimicPortableDCMotorEnv,
    HIMimicCaptureRSIEnvCfg(),
    portable_agent_cfg,
)


def train():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.device = args_cli.device
    env_cfg.scene.robot.actuators = portable_actuators()
    agent_cfg.device = args_cli.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.scene.seed = args_cli.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    print(f"[PORTABLE_CONTRACT] target_clamp=soft_limits actuator=DCMotor v0=6.28 armature=0.01 viscous=0.1")
    print(f"[PORTABLE_CHECKPOINT] {args_cli.checkpoint_path}")
    runner.load(args_cli.checkpoint_path)

    from mimic_real.utils.save_file import copy_py_files
    source_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "envs", "mimic"))
    copy_py_files(source_folder, log_dir + "/py")

    # The environment itself performs synchronized RSI. Runner clock randomization must stay off.
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=False,
    )


if __name__ == "__main__":
    train()
    simulation_app.close()
