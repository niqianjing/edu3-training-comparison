"""First/second-step evidence for the student reference action contract."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="/home/zero/edu3_reference_mimic_v1/provenance/target_mapping_probe.json")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from mimic_real.envs import *  # noqa:F401,F403
from mimic_real.utils import task_registry


def tensor_list(value):
    return value.detach().cpu().tolist()


def snapshot(env, requested):
    limits = env.robot.data.soft_joint_pos_limits
    lower, upper = limits[..., 0], limits[..., 1]
    actual_target = env.robot.data.joint_pos_target
    return {
        "requested_action": tensor_list(requested[0]),
        "delayed_action": tensor_list(env.delayed_policy_action[0]),
        "reference_target_rad": tensor_list(env.reference_joint_target[0]),
        "raw_target_rad": tensor_list(env.raw_joint_target[0]),
        "executed_target_rad": tensor_list(env.executed_joint_target[0]),
        "effective_action": tensor_list(env.action[0]),
        "last_action": tensor_list(env.last_action[0]),
        "actual_target_readback_rad": tensor_list(actual_target[0]),
        "raw_target_oob_count": int(
            ((env.raw_joint_target < lower) | (env.raw_joint_target > upper)).sum().item()
        ),
        "executed_target_oob_count": int(
            ((env.executed_joint_target < lower) | (env.executed_joint_target > upper)).sum().item()
        ),
        "target_readback_max_abs_error_rad": float(
            (actual_target - env.executed_joint_target).abs().max().item()
        ),
        "history_effective_action_max_abs_error": float(
            (env.last_action - env.action).abs().max().item()
        ),
    }


def main():
    env_cfg, _ = task_registry.get_cfgs("edu3_reference_mimic_r1")
    env_cfg.device = args.device
    env_cfg.scene.num_envs = 1
    env_cfg.scene.seed = 42
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.domain_rand.randomize_robot_friction.enable = False
    env_cfg.domain_rand.add_rigid_body_mass.enable = False
    env_cfg.domain_rand.push_robot.enable = False
    env_cfg.terminate.terminate_contacts = False
    env_cfg.terminate.terminate_capture_points_far = False

    env_class = task_registry.get_task_class("edu3_reference_mimic_r1")
    env = env_class(env_cfg, True)
    obs0, _ = env.get_observations()

    zero = torch.zeros((1, env.num_actions), device=env.device)
    env.step(zero)
    step0 = snapshot(env, zero)

    high = torch.full((1, env.num_actions), 10.0, device=env.device)
    env.step(high)
    step1 = snapshot(env, high)

    report = {
        "task": "edu3_reference_mimic_r1",
        "joint_names": env.robot.joint_names,
        "actor_input_shape": list(obs0.shape),
        "per_frame_actor_input": int(obs0.shape[1] // 10),
        "history_frames": 10,
        "step0_zero_action": step0,
        "step1_forced_clamp": step1,
        "pass": (
            list(obs0.shape) == [1, 750]
            and step0["executed_target_oob_count"] == 0
            and step1["raw_target_oob_count"] > 0
            and step1["executed_target_oob_count"] == 0
            and step1["target_readback_max_abs_error_rad"] < 1.0e-6
            and step1["history_effective_action_max_abs_error"] < 1.0e-6
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EDU3_TARGET_MAPPING_PROBE={'PASS' if report['pass'] else 'FAIL'}")
    print(json.dumps({
        "actor_input_shape": report["actor_input_shape"],
        "step0_raw_oob": step0["raw_target_oob_count"],
        "step1_raw_oob": step1["raw_target_oob_count"],
        "step1_executed_oob": step1["executed_target_oob_count"],
        "target_readback_max_error": step1["target_readback_max_abs_error_rad"],
        "history_effective_action_max_error": step1["history_effective_action_max_abs_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
    simulation_app.close()
