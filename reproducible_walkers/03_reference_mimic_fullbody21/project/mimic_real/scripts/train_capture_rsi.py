from mimic_real.utils import task_registry

import argparse

from isaaclab.app import AppLauncher
import mimic_real.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="Train Xiaohai with synchronized reference-state initialization.")
parser.add_argument("--task", type=str, default="hi_mimic_capture_rsi")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
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
import os
from datetime import datetime
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

task_registry.register(
    "hi_mimic_capture_rsi",
    HIMimicCaptureRSIEnv,
    HIMimicCaptureRSIEnvCfg(),
    HIMimicCaptureRSIAgentCfg(),
)


def train():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.device = args_cli.device
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