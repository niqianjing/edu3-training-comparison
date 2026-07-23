"""First 20 ms Isaac parity probe for Xiaohai model 15000."""
import argparse
import json
import os
import torch

from isaaclab.app import AppLauncher
from mimic_real.agents.on_policy_runner import OnPolicyRunner
from mimic_real.utils import task_registry
import mimic_real.utils.cli_args as cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--task", default="hi_mimic_capture_rsi")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.actuators import IdealPDActuatorCfg
from mimic_real.envs import *  # noqa: F401,F403
from mimic_real.utils.cli_args import update_rsl_rl_cfg
from mimic_real.envs.mimic.hi_mimic_capture_rsi_env import HIMimicCaptureRSIEnv
from mimic_real.envs.mimic.hi_mimic_capture_rsi_config import HIMimicCaptureRSIAgentCfg, HIMimicCaptureRSIEnvCfg
from mimic_real.envs.mimic.hi_mimic_env import BaseEnv as OriginalBaseEnv

task_registry.register("hi_mimic_capture_rsi", HIMimicCaptureRSIEnv, HIMimicCaptureRSIEnvCfg(), HIMimicCaptureRSIAgentCfg())
RUN = "2026-07-22_16-15-18_CAPTURE_RSI_R2_seed42_RESUME10000_TO15000"
CHECKPOINT = "model_15000.pt"


def snapshot(env, target, ms):
    return {
        "ms": ms,
        "q": env.robot.data.joint_pos[0].detach().cpu().tolist(),
        "dq": env.robot.data.joint_vel[0].detach().cpu().tolist(),
        "applied_torque": env.robot.data.applied_torque[0].detach().cpu().tolist(),
        "target": target[0].detach().cpu().tolist(),
        "root_pose": env.robot.data.root_state_w[0, :7].detach().cpu().tolist(),
    }


def main():
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
    env_cfg.terminate.terminate_contacts = False
    env_cfg.terminate.terminate_capture_points_far = False

    # Isolated portable-actuator probe: explicit torque PD in PhysX.
    # Keep armature/friction at the asset contract (zero); do not use the PhysX velocity brake.
    common = dict(
        effort_limit_sim=1.0e9,
        velocity_limit=6.28,
        velocity_limit_sim=1000.0,
        stiffness=80.0,
        damping=1.0,
        armature=0.01,
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
    )
    env_cfg.scene.robot.actuators = {
        "portable_body24": IdealPDActuatorCfg(
            joint_names_expr=[
                "waist_joint", ".*_hip_pitch_joint", ".*_hip_roll_joint",
                ".*_thigh_joint", ".*_calf_joint",
                ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
            ],
            effort_limit=24.0,
            **common,
        ),
        "portable_arms17": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                ".*_upper_arm_joint", ".*_elbow_joint", ".*_wrist_joint",
            ],
            effort_limit=17.0,
            **common,
        ),
    }

    env = task_registry.get_task_class(args_cli.task)(env_cfg, args_cli.headless)
    OriginalBaseEnv.reset(env, torch.arange(env.num_envs, device=env.device))
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.experiment_name = "hi_mimic_capture_rsi"
    checkpoint = get_checkpoint_path(os.path.abspath(os.path.join("logs", agent_cfg.experiment_name)), RUN, CHECKPOINT)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(checkpoint), device=agent_cfg.device)
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    obs, _ = env.get_observations()
    with torch.inference_mode():
        action = policy(obs)
    delayed = env.action_buffer.compute(action)
    clipped = torch.clip(delayed, -env.clip_actions, env.clip_actions).to(env.device)
    target = clipped * env.action_scale + env.robot.data.default_joint_pos + env.motion_loader.get_dof_pos_batch(env.phase)
    limits = env.robot.data.soft_joint_pos_limits[0]
    target = torch.maximum(torch.minimum(target, limits[:, 1]), limits[:, 0])

    samples = [snapshot(env, target, 0)]
    for step in range(4):
        env.robot.set_joint_position_target(target)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        samples.append(snapshot(env, target, (step + 1) * 5))

    physx = env.robot.root_physx_view
    payload = {
        "engine": "isaac_physx_explicit_ideal_pd",
        "dt": env.physics_dt,
        "joint_names": list(env.robot.joint_names),
        "action": action[0].detach().cpu().tolist(),
        "stiffness": env.robot.data.joint_stiffness[0].detach().cpu().tolist(),
        "damping": env.robot.data.joint_damping[0].detach().cpu().tolist(),
        "armature": physx.get_dof_armatures()[0].detach().cpu().tolist(),
        "velocity_limit": physx.get_dof_max_velocities()[0].detach().cpu().tolist(),
        "samples": samples,
    }
    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("FIRST_CYCLE=" + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

