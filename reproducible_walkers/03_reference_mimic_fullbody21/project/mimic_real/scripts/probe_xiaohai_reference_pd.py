"""No-policy, no-reset PD tracking gate for an original or edited reference."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import isaaclab.utils.math as math_utils
from mimic_real.envs.mimic.hi_mimic_capture_rsi_env import HIMimicCaptureRSIEnv
from mimic_real.envs.mimic.hi_mimic_capture_rsi_config import HIMimicCaptureRSIEnvCfg
from mimic_real.envs.mimic.hi_mimic_env import BaseEnv


def no_reset(self):
    z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return z, z.clone()


def main():
    cfg = HIMimicCaptureRSIEnvCfg()
    cfg.device = args.device
    cfg.scene.num_envs = 1
    cfg.noise.add_noise = False
    cfg.motion_data.motion_file_path = args.motion
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.randomize_robot_friction.enable = False
    cfg.domain_rand.add_rigid_body_mass.enable = False
    cfg.domain_rand.push_robot.enable = False
    cfg.domain_rand.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.domain_rand.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    for key in cfg.domain_rand.reset_robot_base.params["pose_range"]:
        cfg.domain_rand.reset_robot_base.params["pose_range"][key] = (0.0, 0.0)
    for key in cfg.domain_rand.reset_robot_base.params["velocity_range"]:
        cfg.domain_rand.reset_robot_base.params["velocity_range"][key] = (0.0, 0.0)

    env = HIMimicCaptureRSIEnv(cfg, args.headless)
    ids = torch.arange(env.num_envs, device=env.device)
    BaseEnv.reset(env, ids)
    env.check_reset = types.MethodType(no_reset, env)
    actions = torch.zeros((1, env.num_actions), device=env.device)

    body_names = list(env.robot.body_names)
    base_id = body_names.index("base_link")
    feet_ids = [body_names.index("l_ankle_roll_link"), body_names.index("r_ankle_roll_link")]
    heights, tilts, q_errors, torques, foot_pos, contacts = [], [], [], [], [], []
    first_failed = None
    for step in range(int(env.max_episode_length)):
        env.step(actions)
        q_ref = env.motion_loader.get_dof_pos_batch(env.phase)
        q_errors.append((env.robot.data.joint_pos - q_ref).detach().cpu().numpy()[0])
        torques.append(env.robot.data.applied_torque.detach().cpu().numpy()[0])
        heights.append(float(env.robot.data.body_pos_w[0, base_id, 2]))
        quat = env.robot.data.body_quat_w[0, base_id].unsqueeze(0)
        gravity = torch.tensor([[0.0, 0.0, -1.0]], device=env.device)
        local_g = math_utils.quat_apply_inverse(quat, gravity)[0]
        tilt = float(torch.rad2deg(torch.acos(torch.clamp(-local_g[2], -1.0, 1.0))))
        tilts.append(tilt)
        foot_pos.append(env.robot.data.body_pos_w[0, feet_ids].detach().cpu().numpy())
        f = env.contact_sensor.data.net_forces_w_history[0, :, env.feet_cfg.body_ids]
        contacts.append(torch.max(torch.linalg.vector_norm(f, dim=-1), dim=0).values.detach().cpu().numpy())
        if first_failed is None and (heights[-1] < 0.20 or tilt > 60.0):
            first_failed = step

    q_errors = np.asarray(q_errors)
    torques = np.asarray(torques)
    foot_pos = np.asarray(foot_pos)
    contacts = np.asarray(contacts)
    summary = {
        "motion": args.motion,
        "steps": int(env.max_episode_length),
        "duration_s": float(env.max_episode_length * env.step_dt),
        "first_failure_step": first_failed,
        "first_failure_s": None if first_failed is None else float(first_failed * env.step_dt),
        "minimum_base_height_m": float(np.min(heights)),
        "maximum_tilt_deg": float(np.max(tilts)),
        "joint_tracking_rms_rad": float(np.sqrt(np.mean(q_errors ** 2))),
        "joint_tracking_peak_rad": float(np.max(np.abs(q_errors))),
        "torque_rms_max_joint_nm": float(np.max(np.sqrt(np.mean(torques ** 2, axis=0)))),
        "torque_peak_nm": float(np.max(np.abs(torques))),
        "maximum_foot_contact_force_n": float(np.max(contacts)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".npz"), heights=np.asarray(heights), tilts=np.asarray(tilts),
        q_errors=q_errors, torques=torques, foot_pos=foot_pos, contacts=contacts,
    )
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    app.close()
