from mimic_real.utils import task_registry
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from mimic_real.envs import *  # noqa:F401,F403
from mimic_real.envs.mimic.hi_mimic_capture_rsi_env import HIMimicCaptureRSIEnv
from mimic_real.envs.mimic.hi_mimic_capture_rsi_config import HIMimicCaptureRSIAgentCfg, HIMimicCaptureRSIEnvCfg

task_registry.register("hi_mimic_capture_rsi_probe", HIMimicCaptureRSIEnv, HIMimicCaptureRSIEnvCfg(), HIMimicCaptureRSIAgentCfg())
cfg, _ = task_registry.get_cfgs("hi_mimic_capture_rsi_probe")
cfg.device = args.device
cfg.scene.num_envs = args.num_envs
cfg.scene.seed = 42
env = HIMimicCaptureRSIEnv(cfg, True)

ref_joint = env.motion_loader.get_dof_pos_batch(env.phase)
joint_abs = torch.abs(env.robot.data.joint_pos - ref_joint)
capture_err = env.global_capture_points_error()
capture_sq = torch.sum(capture_err.reshape(capture_err.shape[0], -1) ** 2, dim=1)

print("XIAOHAI_RSI_PROBE_BEGIN")
print("phase_min_max=", float(env.phase.min()), float(env.phase.max()))
print("unique_phase_count=", int(torch.unique(env.episode_length_buf).numel()))
print("joint_ref_abs_mean_max=", float(joint_abs.mean()), float(joint_abs.max()))
print("capture_sq_mean_max=", float(capture_sq.mean()), float(capture_sq.max()))
print("capture_over_0p5=", int((capture_sq > 0.5).sum()))
print("joint_velocity_rms=", float(torch.sqrt(torch.mean(env.robot.data.joint_vel ** 2))))
print("root_velocity_rms=", float(torch.sqrt(torch.mean(env.robot.data.root_com_vel_w ** 2))))
print("episode_steps_sample=", env.episode_length_buf[:8].tolist())
print("XIAOHAI_RSI_PROBE_END")
simulation_app.close()