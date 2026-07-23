"""Strict 8 s Isaac Gym evaluation for the two EDU3/Xiaohai transfer tasks."""

from __future__ import print_function

import json
import math
import os

import isaacgym  # noqa: F401 -- must precede torch
import numpy as np
import torch

from gym import LEGGED_GYM_ROOT_DIR
from gym.envs import *  # noqa: F401,F403
from gym.envs.edu3_12 import edu3_tasks  # noqa: F401
from gym.utils import get_args, task_registry


def quat_to_rpy(q):
    x, y, z, w = [float(v) for v in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def main(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 100000
    env_cfg.env.env_spacing = 2.0
    env_cfg.seed = 42
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_Kp_factor = False
    env_cfg.domain_rand.randomize_Kd_factor = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.init_state.reset_mode = "reset_to_basic"
    env_cfg.commands.resampling_time = -1
    env_cfg.commands.curriculum = False
    env_cfg.commands.ranges.lin_vel_x = [0.3, 0.3]
    env_cfg.commands.ranges.lin_vel_y = 0.0
    env_cfg.commands.ranges.yaw_vel = 0.0

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = 0.3
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    runner.alg.actor_critic.eval()

    export_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR, "strict_eval",
        "%s_%s_model%s_exported" % (args.task, args.load_run, args.checkpoint),
    )
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)
    runner.export(export_dir)
    print("EDU3_POLICY_EXPORT_DIR", export_dir)

    names = list(env.dof_names)
    lower = np.asarray([env_cfg.init_state.dof_pos_range[n][0] for n in names])
    upper = np.asarray([env_cfg.init_state.dof_pos_range[n][1] for n in names])
    continuous = env.edu3_continuous[0].detach().cpu().numpy().copy()
    peak = env.torque_limits.detach().cpu().numpy().reshape(-1).copy()
    dt = float(env.dt)
    steps = int(round(8.0 / dt))

    logs = {k: [] for k in [
        "base_pos", "base_quat", "dof_pos", "dof_vel", "target",
        "motor", "friction", "net", "reset"
    ]}
    for _ in range(steps):
        actions = runner.get_inference_actions()
        runner.set_actions(actions)
        env.step()
        logs["base_pos"].append(env.base_pos[0].detach().cpu().numpy().copy())
        logs["base_quat"].append(env.base_quat[0].detach().cpu().numpy().copy())
        logs["dof_pos"].append(env.dof_pos[0].detach().cpu().numpy().copy())
        logs["dof_vel"].append(env.dof_vel[0].detach().cpu().numpy().copy())
        desired = env.dof_pos_target[0] + env.default_dof_pos[0]
        logs["target"].append(desired.detach().cpu().numpy().copy())
        logs["motor"].append(env.edu3_motor_torques[0].detach().cpu().numpy().copy())
        logs["friction"].append(env.edu3_friction_torques[0].detach().cpu().numpy().copy())
        logs["net"].append(env.edu3_net_torques[0].detach().cpu().numpy().copy())
        reset = int(env.reset_buf[0].item())
        logs["reset"].append(reset)
        if reset:
            break
        runner.reset_envs()

    arr = {k: np.asarray(v) for k, v in logs.items()}
    pos = arr["base_pos"]
    quat = arr["base_quat"]
    rpy = np.asarray([quat_to_rpy(q) for q in quat])
    q = arr["dof_pos"]
    motor = arr["motor"]
    net = arr["net"]
    target = arr["target"]
    span = upper - lower
    lower_edge = q <= (lower + 0.10 * span)
    upper_edge = q >= (upper - 0.10 * span)
    target_oob = np.logical_or(target < lower, target > upper)

    joints = {}
    for i, name in enumerate(names):
        joints[name] = {
            "q_mean_deg": float(np.degrees(np.mean(q[:, i]))),
            "q_min_deg": float(np.degrees(np.min(q[:, i]))),
            "q_max_deg": float(np.degrees(np.max(q[:, i]))),
            "lower_edge_rate": float(np.mean(lower_edge[:, i])),
            "upper_edge_rate": float(np.mean(upper_edge[:, i])),
            "target_oob_rate": float(np.mean(target_oob[:, i])),
            "motor_rms_Nm": float(np.sqrt(np.mean(motor[:, i] ** 2))),
            "net_rms_Nm": float(np.sqrt(np.mean(net[:, i] ** 2))),
            "continuous_ratio": float(np.sqrt(np.mean(motor[:, i] ** 2)) / continuous[i]),
            "peak_saturation_rate": float(np.mean(np.abs(motor[:, i]) >= 0.99 * peak[i])),
        }

    result = {
        "task": args.task,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "dt_s": dt,
        "steps": int(len(pos)),
        "duration_s": float(len(pos) * dt),
        "reset_count": int(np.sum(arr["reset"])),
        "forward_m": float(pos[-1, 0] - pos[0, 0]),
        "lateral_m": float(pos[-1, 1] - pos[0, 1]),
        "min_height_m": float(np.min(pos[:, 2])),
        "yaw_change_deg": float(np.degrees(np.unwrap(rpy[:, 2])[-1] - np.unwrap(rpy[:, 2])[0])),
        "max_abs_roll_deg": float(np.degrees(np.max(np.abs(rpy[:, 0])))),
        "max_abs_pitch_deg": float(np.degrees(np.max(np.abs(rpy[:, 1])))),
        "target_oob_total": int(np.sum(target_oob)),
        "joints": joints,
    }
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "strict_eval")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    stem = "%s_%s_model%s" % (args.task, args.load_run, args.checkpoint)
    npz_path = os.path.join(out_dir, stem + ".npz")
    json_path = os.path.join(out_dir, stem + ".json")
    np.savez(npz_path, names=np.asarray(names), lower=lower, upper=upper, **arr)
    with open(json_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("EDU3_STRICT_JSON", json_path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    with torch.inference_mode():
        main(get_args())
