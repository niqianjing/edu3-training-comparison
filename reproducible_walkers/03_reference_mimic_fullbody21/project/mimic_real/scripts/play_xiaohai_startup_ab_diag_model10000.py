import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import ImageGrab

from isaaclab.app import AppLauncher
from mimic_real.agents.on_policy_runner import OnPolicyRunner
from mimic_real.utils import task_registry
import mimic_real.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="Strict Xiaohai synchronized-RSI evaluation")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import get_checkpoint_path
from mimic_real.envs import *  # noqa:F401,F403
from mimic_real.utils.cli_args import update_rsl_rl_cfg


from mimic_real.envs.mimic.hi_mimic_capture_rsi_env import HIMimicCaptureRSIEnv
from mimic_real.envs.mimic.hi_mimic_capture_rsi_config import (
    HIMimicCaptureRSIAgentCfg,
    HIMimicCaptureRSIEnvCfg,
)

task_registry.register(
    "hi_mimic_capture_rsi",
    HIMimicCaptureRSIEnv,
    HIMimicCaptureRSIEnvCfg(),
    HIMimicCaptureRSIAgentCfg(),
)

RUN = "2026-07-22_14-25-44_CAPTURE_RSI_R2_seed42_RESUME7000_TO10000"
CHECKPOINT = "model_10000.pt"
DIAG_MODE = os.environ.get("XIAOHAI_DIAG_MODE", "A_STAND_BLEND").upper()
OUT_DIR = "/home/zero/xiaohai_startup_ab_diag"
TAG = f"Xiaohai_model10000_{DIAG_MODE}"
VIDEO = os.path.join(OUT_DIR, TAG + ".mp4")
DUMP = os.path.join(OUT_DIR, TAG + "_dump.npz")
SUMMARY = os.path.join(OUT_DIR, TAG + "_summary.json")


def play():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
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

    # Strict evaluation must not auto-reset. We calculate the same capture gate manually.
    capture_threshold = float(env_cfg.terminate.capture_points_distance_threshold)
    env_cfg.terminate.terminate_contacts = False
    env_cfg.terminate.terminate_capture_points_far = False

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)


    # Strict test: initialize the physical robot and reference clock at frame zero.
    # Call the original phase-zero reset rather than the RSI random-phase override.
    from mimic_real.envs.mimic.hi_mimic_env import BaseEnv as OriginalBaseEnv
    eval_ids = torch.arange(env.num_envs, device=env.device)
    OriginalBaseEnv.reset(env, eval_ids)

    # Keep the original 8.1 s reference clock but suppress every automatic reset,
    # including the normal episode timeout at the end of the motion.
    def strict_no_reset():
        zeros = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return zeros, zeros

    env.check_reset = strict_no_reset
    print(
        f"XIAOHAI_STRICT_PHASE0 phase={float(env.phase[0])} "
        f"episode_step={int(env.episode_length_buf[0])} "
        f"max_steps={int(env.max_episode_length)}",
        flush=True,
    )

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.experiment_name = "hi_mimic_capture_rsi"
    agent_cfg.load_run = RUN
    agent_cfg.load_checkpoint = CHECKPOINT
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    print("resume_path=============", resume_path, flush=True)
    policy = runner.get_inference_policy(device=env.device)

    original_max_steps = int(env.max_episode_length)

    def write_reference_state(phase_value: float, zero_velocity: bool = False):
        phase_tensor = torch.full((env.num_envs,), phase_value, device=env.device)
        env.phase[:] = phase_tensor
        joint_pos = env.motion_loader.get_dof_pos_batch(phase_tensor)
        joint_vel = env.motion_loader.get_dof_vel_batch(phase_tensor)
        root_pos = env.motion_loader.get_root_trans_batch(phase_tensor) + env.scene.env_origins
        root_quat = env.motion_loader.get_root_rot_batch(phase_tensor)
        root_vel = env.robot.data.default_root_state[:, 7:13].clone()
        root_vel[:, 0:3] = env.motion_loader.get_root_vel_batch(phase_tensor)
        root_vel[:, 3:6] = env.motion_loader.get_root_omega_batch(phase_tensor)
        if zero_velocity:
            joint_vel.zero_()
            root_vel.zero_()
        env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        env.robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
        env.robot.write_root_velocity_to_sim(root_vel)
        env.scene.write_data_to_sim()
        env.sim.forward()

    def actor_obs_from_buffer():
        value = env.actor_obs_buffer.buffer.reshape(env.num_envs, -1)
        return torch.clip(value, -env.clip_obs, env.clip_obs)

    def phase_a(step_index: int) -> float:
        stand_steps = 25
        blend_steps = 15
        if step_index < stand_steps:
            return 0.0
        if step_index < stand_steps + blend_steps:
            u = (step_index - stand_steps) / float(blend_steps)
            smooth_u = u * u * (3.0 - 2.0 * u)
            return smooth_u * blend_steps / float(original_max_steps)
        return min((step_index - stand_steps) / float(original_max_steps), 0.999999)

    env.actor_obs_buffer.reset()
    env.critic_obs_buffer.reset()
    env.last_action.zero_()
    env.action_buffer.reset()

    if DIAG_MODE == "A_STAND_BLEND":
        write_reference_state(0.0, zero_velocity=True)
        env.episode_length_buf.zero_()
        obs, _ = env.get_observations()
        total_steps = original_max_steps + 25
        print("XIAOHAI_DIAG_A stand=0.5s blend=0.3s policy_active_from_t0", flush=True)
    elif DIAG_MODE == "B_HISTORY":
        # Build ten chronological, sensor-like reference observations for frames 0..9.
        # The current physical state is frame 9 when the frozen policy takes over.
        for history_step in range(10):
            history_phase = history_step / float(original_max_steps)
            write_reference_state(history_phase, zero_velocity=False)
            env.phase[:] = history_phase
            actor_frame, critic_frame = env.compute_current_observations()
            env.actor_obs_buffer.append(actor_frame)
            env.critic_obs_buffer.append(critic_frame)
        env.episode_length_buf[:] = 9
        env.phase[:] = 9 / float(original_max_steps)
        obs = actor_obs_from_buffer()
        total_steps = original_max_steps - 9
        print("XIAOHAI_DIAG_B history=reference_frames_0_to_9 policy_starts_at_frame9", flush=True)
    else:
        raise ValueError(f"Unknown XIAOHAI_DIAG_MODE={DIAG_MODE}")

    env.sim.set_camera_view(eye=[1.15, -1.15, 0.90], target=[0.0, 0.0, 0.48])
    os.makedirs(OUT_DIR, exist_ok=True)
    writer = imageio.get_writer(VIDEO, fps=50, codec="libx264", quality=7, pixelformat="yuv420p")

    heights = []
    tilts_deg = []
    capture_errors = []
    base_positions = []
    actions_log = []
    torques_log = []
    first_capture_step = None
    dof_tracking_rewards = []
    capture_tracking_rewards = []

    try:
        for step in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                if DIAG_MODE == "A_STAND_BLEND":
                    next_phase = phase_a(step + 1)
                    env.phase[:] = next_phase
                    env.episode_length_buf.zero_()
                    actor_frame, _ = env.compute_current_observations()
                    env.actor_obs_buffer._buffer[env.actor_obs_buffer._pointer] = actor_frame
                    obs = actor_obs_from_buffer()

                base_pos = env.robot.data.body_pos_w[:, env.base_link_body_ids, :].squeeze(1)
                base_quat = env.robot.data.body_quat_w[:, env.base_link_body_ids, :].squeeze(1)
                projected_gravity = math_utils.quat_apply_inverse(base_quat, env.gravity_vec)
                upright_cos = torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)
                tilt = torch.rad2deg(torch.acos(upright_cos))
                cap = env.local_capture_points_error_sum() if env.use_local_capture_points else env.global_capture_points_error_sum()

                heights.append(float(base_pos[0, 2]))
                tilts_deg.append(float(tilt[0]))
                capture_errors.append(float(cap[0]))
                desired_joint_pos = env.motion_loader.get_dof_pos_batch(env.phase)
                joint_error_sum = torch.sum((desired_joint_pos - env.robot.data.joint_pos) ** 2, dim=1)
                dof_tracking_rewards.append(float(torch.exp(-joint_error_sum[0] / (0.5 ** 2))))
                capture_tracking_rewards.append(float(torch.exp(-cap[0] / (0.5 ** 2))))
                base_positions.append(base_pos[0].detach().cpu().numpy())
                actions_log.append(actions[0].detach().cpu().numpy())
                torques_log.append(env.robot.data.applied_torque[0].detach().cpu().numpy())

                if first_capture_step is None and float(cap[0]) > capture_threshold:
                    first_capture_step = step
                    print(f"XIAOHAI_FIRST_CAPTURE_TERMINATION_STEP={step}", flush=True)

            frame = ImageGrab.grab(bbox=(32, 96, 1032, 696), xdisplay=":1")
            writer.append_data(np.asarray(frame.resize((1280, 720))))
            if step % 50 == 0:
                print(f"XIAOHAI_VIDEO_STEP={step}", flush=True)
    finally:
        writer.close()

    base_positions_np = np.asarray(base_positions)
    torques_np = np.asarray(torques_log)
    summary = {
        "checkpoint": resume_path,
        "diagnostic_mode": DIAG_MODE,
        "steps": total_steps,
        "seconds": total_steps * 0.02,
        "capture_threshold": capture_threshold,
        "first_capture_termination_step": first_capture_step,
        "first_capture_termination_s": None if first_capture_step is None else first_capture_step * 0.02,
        "minimum_base_height_m": float(np.min(heights)),
        "maximum_tilt_deg": float(np.max(tilts_deg)),
        "final_displacement_m": (base_positions_np[-1] - base_positions_np[0]).tolist(),
        "max_capture_error": float(np.max(capture_errors)),
        "tracking_dof_pos_mean": float(np.mean(dof_tracking_rewards)),
        "tracking_capture_points_mean": float(np.mean(capture_tracking_rewards)),
        "torque_rms_Nm": np.sqrt(np.mean(np.square(torques_np), axis=0)).tolist(),
        "torque_abs_peak_Nm": np.max(np.abs(torques_np), axis=0).tolist(),
        "startup_0p5s_torque_vector_norm_mean_Nm": float(np.mean(np.linalg.norm(torques_np[:25], axis=1))),
        "startup_0p5s_joint_rms_median_Nm": float(np.median(np.sqrt(np.mean(np.square(torques_np[:25]), axis=0)))),
        "startup_0p5s_max_step_delta_norm_Nm": float(np.max(np.linalg.norm(np.diff(torques_np[:25], axis=0), axis=1))),
        "automatic_reset": False,
        "domain_randomization": False,
    }
    np.savez(DUMP, heights=np.asarray(heights), tilts_deg=np.asarray(tilts_deg),
             capture_errors=np.asarray(capture_errors), base_positions=base_positions_np,
             actions=np.asarray(actions_log), applied_torques=torques_np)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("XIAOHAI_EVAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    print("XIAOHAI_VIDEO_DONE=" + VIDEO, flush=True)
    print("XIAOHAI_DUMP_DONE=" + DUMP, flush=True)


if __name__ == "__main__":
    play()
    simulation_app.close()




