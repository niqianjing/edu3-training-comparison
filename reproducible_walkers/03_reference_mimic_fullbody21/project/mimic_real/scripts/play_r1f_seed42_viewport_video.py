import os
import torch
import time
import numpy as np
import imageio.v2 as imageio
from PIL import ImageGrab
from mimic_real.utils import task_registry

import argparse

from isaaclab.app import AppLauncher
from mimic_real.agents.on_policy_runner import OnPolicyRunner
# local imports
import mimic_real.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

from mimic_real.envs import *  # noqa:F401, F403
from mimic_real.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_cfg.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.domain_rand.randomize_robot_friction.enable = False
    env_cfg.domain_rand.add_rigid_body_mass.enable = False
    env_cfg.domain_rand.push_robot.enable = False
    env_cfg.domain_rand.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.domain_rand.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    for key in env_cfg.domain_rand.reset_robot_base.params["pose_range"]:
        env_cfg.domain_rand.reset_robot_base.params["pose_range"][key] = (0.0, 0.0)
    for key in env_cfg.domain_rand.reset_robot_base.params["velocity_range"]:
        env_cfg.domain_rand.reset_robot_base.params["velocity_range"][key] = (0.0, 0.0)


    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.load_run = "2026-07-22_01-31-03_ORIGINAL_FULLBODY_WALK_R1F_seed42"
    # agent_cfg.load_run = "0server"
    agent_cfg.load_checkpoint = "model_1000.pt"
    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, )
    print('resume_path=============',resume_path)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy_walk_seed42_model1000.pt")
    export_policy_as_onnx(runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy_walk_seed42_model1000.onnx")

    if not args_cli.headless:
        from mimic_real.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    env.sim.set_camera_view(eye=[1.15, -1.15, 0.90], target=[0.0, 0.0, 0.48])
    video_path = "/home/zero/xiaohai_fullbody_eval/Xiaohai_OriginalFullbody_seed42_Isaac_viewport.mp4"
    writer = imageio.get_writer(video_path, fps=50, codec="libx264", quality=7,
                                pixelformat="yuv420p")
    try:
        for step in range(405):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            frame = ImageGrab.grab(bbox=(32, 96, 1032, 696), xdisplay=":1")
            frame = frame.resize((1280, 720))
            writer.append_data(np.asarray(frame))
            if step % 50 == 0:
                print(f"XIAOHAI_VIDEO_STEP={step}", flush=True)
    finally:
        writer.close()
        print(f"XIAOHAI_VIDEO_DONE={video_path}", flush=True)

if __name__ == '__main__':
    play()
    simulation_app.close()
