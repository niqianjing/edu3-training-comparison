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
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils import math as math_utils
from isaaclab.actuators import DCMotorCfg
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

OUT_DIR = args_cli.out_dir
VIDEO = os.path.join(OUT_DIR, f"{args_cli.name}.mp4")
DUMP = os.path.join(OUT_DIR, f"{args_cli.name}_dump.npz")
SUMMARY = os.path.join(OUT_DIR, f"{args_cli.name}_summary.json")


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

    # Portable explicit actuator diagnostic: same torque PD, passive damping,
    # armature, effort ceiling and operational speed ceiling used by MuJoCo.
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
    env_cfg.scene.robot.actuators = {
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
    resume_path = os.path.abspath(args_cli.checkpoint_path)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(resume_path)
    log_dir = os.path.dirname(resume_path)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    print("resume_path=============", resume_path, flush=True)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.get_observations()
    env.sim.set_camera_view(eye=[1.15, -1.15, 0.90], target=[0.0, 0.0, 0.48])
    os.makedirs(OUT_DIR, exist_ok=True)
    writer = imageio.get_writer(VIDEO, fps=50, codec="libx264", quality=7, pixelformat="yuv420p")

    heights = []
    tilts_deg = []
    capture_errors = []
    base_positions = []
    actions_log = []
    torques_log = []
    joint_pos_log = []
    joint_vel_log = []
    reference_pos_log = []
    raw_targets_log = []
    clamped_targets_log = []
    policy_obs_log = []
    first_capture_step = None
    dof_tracking_rewards = []
    capture_tracking_rewards = []
    raw_target_oob_elements = 0
    raw_target_total_elements = 0
    raw_target_oob_steps = 0
    soft_limits = env.robot.data.soft_joint_pos_limits[0]
    print("XIAOHAI_TARGETCLAMP_LIMITS=" + json.dumps({
        name: [float(soft_limits[i, 0]), float(soft_limits[i, 1])]
        for i, name in enumerate(env.robot.joint_names)
    }, ensure_ascii=False), flush=True)

    try:
        for step in range(405):
            with torch.inference_mode():
                policy_obs_log.append(obs[0].detach().cpu().numpy())
                raw_actions = policy(obs)
                ref_now = env.motion_loader.get_dof_pos_batch(env.phase)
                raw_targets = ref_now + env.robot.data.default_joint_pos + raw_actions * env.action_scale
                oob = (raw_targets < soft_limits[:, 0]) | (raw_targets > soft_limits[:, 1])
                raw_target_oob_elements += int(torch.sum(oob).item())
                raw_target_total_elements += int(oob.numel())
                raw_target_oob_steps += int(torch.any(oob).item())
                clamped_targets = torch.maximum(torch.minimum(raw_targets, soft_limits[:, 1]), soft_limits[:, 0])
                actions = (clamped_targets - ref_now - env.robot.data.default_joint_pos) / env.action_scale
                obs, _, _, _ = env.step(actions)

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
                joint_pos_log.append(env.robot.data.joint_pos[0].detach().cpu().numpy())
                joint_vel_log.append(env.robot.data.joint_vel[0].detach().cpu().numpy())
                reference_pos_log.append(desired_joint_pos[0].detach().cpu().numpy())
                raw_targets_log.append(raw_targets[0].detach().cpu().numpy())
                clamped_targets_log.append(clamped_targets[0].detach().cpu().numpy())

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
        "steps": 405,
        "seconds": 8.1,
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
        "automatic_reset": False,
        "domain_randomization": False,
        "target_clamp_enabled": True,
        "raw_target_oob_element_rate": raw_target_oob_elements / raw_target_total_elements,
        "raw_target_oob_step_rate": raw_target_oob_steps / 405.0,
        "joint_names": list(env.robot.joint_names),
    }
    np.savez(DUMP, heights=np.asarray(heights), tilts_deg=np.asarray(tilts_deg),
             capture_errors=np.asarray(capture_errors), base_positions=base_positions_np,
             actions=np.asarray(actions_log), applied_torques=torques_np,
             joint_pos=np.asarray(joint_pos_log), joint_vel=np.asarray(joint_vel_log),
             reference_pos=np.asarray(reference_pos_log),
             raw_targets=np.asarray(raw_targets_log),
             clamped_targets=np.asarray(clamped_targets_log),
             joint_names=np.asarray(env.robot.joint_names),
             policy_obs=np.asarray(policy_obs_log))
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("XIAOHAI_EVAL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    print("XIAOHAI_VIDEO_DONE=" + VIDEO, flush=True)
    print("XIAOHAI_DUMP_DONE=" + DUMP, flush=True)


if __name__ == "__main__":
    play()
    simulation_app.close()

