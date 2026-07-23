"""Train the isolated student 21-joint Xiaohai-reference task."""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher
import mimic_real.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="Train EDU3 with synchronized reference-state initialization.")
parser.add_argument("--task", type=str, default="edu3_reference_mimic_r1")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from mimic_real.agents.on_policy_runner import OnPolicyRunner
from mimic_real.assets.usd.edu3_reference.training_provenance import write_training_provenance
from mimic_real.envs import *  # noqa:F401,F403
from mimic_real.utils import task_registry
from mimic_real.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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
    actor_obs, _ = env.get_observations()
    print(f"EDU3_ACTOR_OBS_SHAPE={tuple(actor_obs.shape)}", flush=True)
    if actor_obs.shape[1] != 750:
        raise RuntimeError(f"Expected 750 actor inputs, got {actor_obs.shape[1]}")

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Perform the compiled-runtime gate and bind the run to all evidence.
    write_training_provenance(env, log_dir)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    from mimic_real.utils.save_file import copy_py_files
    source_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "envs", "mimic"))
    copy_py_files(source_folder, log_dir + "/py")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)


if __name__ == "__main__":
    train()
    simulation_app.close()
