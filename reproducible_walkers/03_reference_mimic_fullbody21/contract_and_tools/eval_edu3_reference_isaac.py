"""Strict Isaac evaluation for the EDU3 student reference-motion policy.

The evaluator intentionally records the first two control cycles at every
contract boundary.  These traces are the source of truth when the MuJoCo
replay disagrees with Isaac.
"""

import argparse
import json
import os
from types import MethodType

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import ImageGrab

from isaaclab.app import AppLauncher
from mimic_real.agents.on_policy_runner import OnPolicyRunner
from mimic_real.utils import task_registry
import mimic_real.utils.cli_args as cli_args


parser = argparse.ArgumentParser(description="Strict EDU3 reference-policy Isaac evaluation")
parser.add_argument("--task", default="edu3_reference_mimic_r1")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--steps", type=int, default=405)
parser.add_argument("--record-video", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils import math as math_utils
from isaaclab_rl.rsl_rl import export_policy_as_jit
from mimic_real.envs import *  # noqa: F401,F403,E402
from mimic_real.utils.cli_args import update_rsl_rl_cfg  # noqa: E402


def as_numpy(value):
    return value.detach().cpu().numpy()


def disable_randomization(env_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.domain_rand.randomize_robot_friction.enable = False
    env_cfg.domain_rand.add_rigid_body_mass.enable = False
    env_cfg.domain_rand.push_robot.enable = False
    env_cfg.domain_rand.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.domain_rand.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    for value in env_cfg.domain_rand.reset_robot_base.params["pose_range"].values():
        value = value  # keep keys stable for older config objects
    for key in env_cfg.domain_rand.reset_robot_base.params["pose_range"]:
        env_cfg.domain_rand.reset_robot_base.params["pose_range"][key] = (0.0, 0.0)
    for key in env_cfg.domain_rand.reset_robot_base.params["velocity_range"]:
        env_cfg.domain_rand.reset_robot_base.params["velocity_range"][key] = (0.0, 0.0)
    env_cfg.terminate.terminate_contacts = False
    env_cfg.terminate.terminate_capture_points_far = False


def actuator_ledger(env):
    joint_count = len(env.robot.joint_names)
    motor = torch.zeros((1, joint_count), device=env.device)
    passive = torch.zeros_like(motor)
    net = torch.zeros_like(motor)
    for actuator in env.robot.actuators.values():
        indices = actuator.joint_indices
        motor[:, indices] = actuator.motor_effort
        passive[:, indices] = actuator.passive_friction_effort
        net[:, indices] = actuator.applied_effort
    return motor, passive, net


def no_reset(self):
    zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.termination_buf = zeros.clone()
    return zeros, zeros.clone()


def evaluate():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.device = args_cli.device
    disable_randomization(env_cfg)
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    # Force the exact phase-zero physical/reference state used by strict evaluation.
    from mimic_real.envs.mimic.hi_mimic_env import BaseEnv as OriginalBaseEnv
    OriginalBaseEnv.reset(env, torch.arange(env.num_envs, device=env.device))
    env.check_reset = MethodType(no_reset, env)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = args_cli.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(args_cli.checkpoint), device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    export_dir = os.path.join(args_cli.out_dir, "exported")
    export_policy_as_jit(
        runner.alg.policy,
        runner.obs_normalizer,
        path=export_dir,
        filename=args_cli.name + "_policy.pt",
    )

    obs, _ = env.get_observations()
    if tuple(obs.shape) != (1, 750):
        raise RuntimeError(f"Actor input contract mismatch: {tuple(obs.shape)} != (1, 750)")
    env.sim.set_camera_view(eye=[1.15, -1.15, 0.85], target=[0.0, 0.0, 0.42])

    video_path = os.path.join(args_cli.out_dir, args_cli.name + "_Isaac.mp4")
    writer = None
    if args_cli.record_video:
        writer = imageio.get_writer(video_path, fps=50, codec="libx264", quality=7, pixelformat="yuv420p")

    logs = {key: [] for key in (
        "obs", "raw_action", "delayed_action", "reference_target", "raw_target",
        "executed_target", "effective_action", "joint_pos", "joint_vel",
        "motor_torque", "passive_friction_torque", "net_torque", "base_pos",
        "tilt_deg", "capture_error", "foot_contact_force_w", "foot_pos_w",
    )}
    traces = []
    foot_robot_body_ids = []
    for foot_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        body_ids, _ = env.robot.find_bodies(foot_name)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one robot body for {foot_name}, got {body_ids}")
        foot_robot_body_ids.append(body_ids[0])
    lower = as_numpy(env.robot.data.soft_joint_pos_limits[0, :, 0])
    upper = as_numpy(env.robot.data.soft_joint_pos_limits[0, :, 1])
    first_capture_step = None
    capture_threshold = float(env_cfg.terminate.capture_points_distance_threshold)

    try:
        for step in range(args_cli.steps):
            obs_before = obs.clone()
            last_action_before = env.last_action.clone()
            with torch.inference_mode():
                raw_action = policy(obs_before)
                obs, _, _, _ = env.step(raw_action)

            motor, passive, net = actuator_ledger(env)
            base_pos = env.robot.data.body_pos_w[:, env.base_link_body_ids, :].squeeze(1)
            base_quat = env.robot.data.body_quat_w[:, env.base_link_body_ids, :].squeeze(1)
            projected_gravity = math_utils.quat_apply_inverse(base_quat, env.gravity_vec)
            tilt = torch.rad2deg(torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)))
            capture = env.local_capture_points_error_sum() if env.use_local_capture_points else env.global_capture_points_error_sum()
            foot_contact_force = env.contact_sensor.data.net_forces_w_history[:, -1, env.feet_cfg.body_ids, :]
            foot_pos_w = env.robot.data.body_pos_w[:, foot_robot_body_ids, :]
            if first_capture_step is None and float(capture[0]) > capture_threshold:
                first_capture_step = step

            values = {
                "obs": obs_before,
                "raw_action": raw_action,
                "delayed_action": env.delayed_policy_action,
                "reference_target": env.reference_joint_target,
                "raw_target": env.raw_joint_target,
                "executed_target": env.executed_joint_target,
                "effective_action": env.action,
                "joint_pos": env.robot.data.joint_pos,
                "joint_vel": env.robot.data.joint_vel,
                "motor_torque": motor,
                "passive_friction_torque": passive,
                "net_torque": net,
                "base_pos": base_pos,
                "tilt_deg": tilt.unsqueeze(1),
                "capture_error": capture.unsqueeze(1),
                "foot_contact_force_w": foot_contact_force,
                "foot_pos_w": foot_pos_w,
            }
            for key, value in values.items():
                logs[key].append(as_numpy(value[0]))

            if step < 2:
                raw_np = as_numpy(env.raw_joint_target[0])
                exec_np = as_numpy(env.executed_joint_target[0])
                traces.append({
                    "step": step,
                    "observation": as_numpy(obs_before[0]).tolist(),
                    "last_action_input": as_numpy(last_action_before[0]).tolist(),
                    "raw_policy_action": as_numpy(raw_action[0]).tolist(),
                    "delayed_policy_action": as_numpy(env.delayed_policy_action[0]).tolist(),
                    "reference_target_rad": as_numpy(env.reference_joint_target[0]).tolist(),
                    "raw_target_rad": raw_np.tolist(),
                    "clamped_target_rad": exec_np.tolist(),
                    "effective_action": as_numpy(env.action[0]).tolist(),
                    "raw_target_oob_count": int(np.sum((raw_np < lower) | (raw_np > upper))),
                    "executed_target_oob_count": int(np.sum((exec_np < lower - 1e-7) | (exec_np > upper + 1e-7))),
                    "motor_torque_nm": as_numpy(motor[0]).tolist(),
                    "passive_friction_torque_nm": as_numpy(passive[0]).tolist(),
                    "net_torque_nm": as_numpy(net[0]).tolist(),
                    "foot_contact_force_w": as_numpy(foot_contact_force[0]).tolist(),
                })

            if writer is not None:
                frame = ImageGrab.grab(bbox=(32, 96, 1032, 696), xdisplay=":1")
                writer.append_data(np.asarray(frame.resize((1280, 720))))
    finally:
        if writer is not None:
            writer.close()

    arrays = {key: np.asarray(value) for key, value in logs.items()}
    raw_oob = (arrays["raw_target"] < lower) | (arrays["raw_target"] > upper)
    exec_oob = (arrays["executed_target"] < lower - 1e-7) | (arrays["executed_target"] > upper + 1e-7)
    peak = np.asarray([float(v) for v in env.robot.data.joint_effort_limits[0].detach().cpu()])
    rms = np.sqrt(np.mean(np.square(arrays["net_torque"]), axis=0))
    summary = {
        "engine": "Isaac Lab / PhysX",
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "steps": args_cli.steps,
        "seconds": args_cli.steps * float(env.step_dt),
        "actor_input": 750,
        "actor_output": 21,
        "joint_names": list(env.robot.joint_names),
        "first_capture_threshold_step": first_capture_step,
        "first_capture_threshold_s": None if first_capture_step is None else first_capture_step * float(env.step_dt),
        "minimum_base_height_m": float(np.min(arrays["base_pos"][:, 2])),
        "maximum_tilt_deg": float(np.max(arrays["tilt_deg"])),
        "final_displacement_m": (arrays["base_pos"][-1] - arrays["base_pos"][0]).tolist(),
        "raw_target_oob_total": int(raw_oob.sum()),
        "executed_target_oob_total": int(exec_oob.sum()),
        "target_edge_rate": np.mean((np.abs(arrays["executed_target"] - lower) < 1e-3) | (np.abs(arrays["executed_target"] - upper) < 1e-3), axis=0).tolist(),
        "net_torque_rms_nm": rms.tolist(),
        "net_torque_abs_peak_nm": np.max(np.abs(arrays["net_torque"]), axis=0).tolist(),
        "peak_saturation_rate": np.mean(np.abs(arrays["motor_torque"]) >= peak * 0.999, axis=0).tolist(),
        "automatic_reset": False,
        "domain_randomization": False,
        "video": video_path if writer is not None else None,
        "jit_policy": os.path.join(export_dir, args_cli.name + "_policy.pt"),
    }
    np.savez(os.path.join(args_cli.out_dir, args_cli.name + "_Isaac_dump.npz"), **arrays)
    with open(os.path.join(args_cli.out_dir, args_cli.name + "_Isaac_first2.json"), "w", encoding="utf-8") as handle:
        json.dump(traces, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args_cli.out_dir, args_cli.name + "_Isaac_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("EDU3_REFERENCE_ISAAC_EVAL=PASS", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    evaluate()
    simulation_app.close()
