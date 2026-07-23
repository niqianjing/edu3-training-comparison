"""Portable actuator pulse probe in Isaac/PhysX."""
import argparse
import json
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--joint", default="l_ankle_roll_joint")
parser.add_argument("--delta", type=float, default=0.01)
parser.add_argument("--torque", type=float, default=0.8)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
simulation_app = AppLauncher(args).app

from isaaclab.actuators import IdealPDActuatorCfg
from mimic_real.envs.mimic.hi_mimic_capture_rsi_env import HIMimicCaptureRSIEnv
from mimic_real.envs.mimic.hi_mimic_capture_rsi_config import (
    HIMimicCaptureRSIAgentCfg,
    HIMimicCaptureRSIEnvCfg,
)
from mimic_real.envs.mimic.hi_mimic_env import BaseEnv as OriginalBaseEnv
from mimic_real.utils import task_registry

TASK = "hi_mimic_portable_r1_pulse"
task_registry.register(TASK, HIMimicCaptureRSIEnv, HIMimicCaptureRSIEnvCfg(), HIMimicCaptureRSIAgentCfg())


def snap(env, target, commanded, ms):
    return {
        "ms": ms,
        "q": env.robot.data.joint_pos[0].detach().cpu().tolist(),
        "dq": env.robot.data.joint_vel[0].detach().cpu().tolist(),
        "applied_torque": env.robot.data.applied_torque[0].detach().cpu().tolist(),
        "commanded_torque": commanded[0].detach().cpu().tolist(),
        "target": target[0].detach().cpu().tolist(),
        "root_pose": env.robot.data.root_state_w[0, :7].detach().cpu().tolist(),
        "root_velocity": env.robot.data.root_state_w[0, 7:13].detach().cpu().tolist(),
    }


def main():
    env_cfg, _ = task_registry.get_cfgs(TASK)
    env_cfg.device = args.device
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
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    common = dict(
        effort_limit_sim=1.0e9,
        velocity_limit=6.28,
        velocity_limit_sim=1000.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
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

    env = task_registry.get_task_class(TASK)(env_cfg, args.headless)
    OriginalBaseEnv.reset(env, torch.arange(env.num_envs, device=env.device))
    root_pose = env.robot.data.root_state_w[:, :7].clone()
    root_pose[:, 2] += 1.0
    env.robot.write_root_pose_to_sim(root_pose)
    env.robot.write_root_velocity_to_sim(torch.zeros_like(env.robot.data.root_state_w[:, 7:13]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    q0 = env.robot.data.joint_pos.clone()
    target = q0.clone()
    jid = env.robot.joint_names.index(args.joint)
    commanded = torch.zeros_like(q0)
    commanded[:, jid] = args.torque
    samples = [snap(env, target, commanded, 0)]
    for step in range(4):
        env.robot.set_joint_effort_target(commanded)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        samples.append(snap(env, target, commanded, (step + 1) * 5))

    physx = env.robot.root_physx_view
    payload = {
        "engine": "isaac_physx_direct_torque_pulse",
        "dt": env.physics_dt,
        "joint_names": list(env.robot.joint_names),
        "pulse_joint": args.joint,
        "pulse_delta_rad": 0.0,
        "pulse_torque_nm": args.torque,
        "gravity_disabled": True,
        "contact_isolated_by_root_lift_m": 1.0,
        "sim_effort_limit": physx.get_dof_max_forces()[0].detach().cpu().tolist(),
        "sim_velocity_limit": physx.get_dof_max_velocities()[0].detach().cpu().tolist(),
        "armature": physx.get_dof_armatures()[0].detach().cpu().tolist(),
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("PORTABLE_PULSE=" + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
