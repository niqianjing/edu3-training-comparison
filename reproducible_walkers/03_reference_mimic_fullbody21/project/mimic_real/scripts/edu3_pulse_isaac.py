"""EDU3 21-joint deterministic motor pulse in Isaac/PhysX.

Policy, randomization, gravity, contacts and PD feedback are disabled.  The
contract armature and explicit SI friction remain active.  Each joint starts
at the middle of its range and receives the same +1 Nm motor pulse.
"""

import argparse
import hashlib
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--torque", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
simulation_app = AppLauncher(args).app

import torch
from mimic_real.assets.usd.edu3_reference.contract import CONTRACT, CONTRACT_PATH
from mimic_real.envs import *  # noqa: F401,F403
from mimic_real.utils import task_registry


def main():
    cfg, _ = task_registry.get_cfgs("edu3_reference_mimic_r1")
    cfg.device = args.device
    cfg.scene.num_envs = 1
    cfg.scene.seed = 42
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.randomize_robot_friction.enable = False
    cfg.domain_rand.add_rigid_body_mass.enable = False
    cfg.domain_rand.push_robot.enable = False
    cfg.terminate.terminate_contacts = False
    cfg.terminate.terminate_capture_points_far = False
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 2.0)

    # Direct feed-forward effort only.  Keep contract armature and the custom
    # measured-friction actuator; remove only its PD feedback for this probe.
    for actuator_cfg in cfg.scene.robot.actuators.values():
        actuator_cfg.stiffness = 0.0
        actuator_cfg.damping = 0.0

    env = task_registry.get_task_class("edu3_reference_mimic_r1")(cfg, True)
    robot = env.robot
    for actuator in robot.actuators.values():
        actuator.stiffness[:] = 0.0
        actuator.damping[:] = 0.0

    names = list(robot.joint_names)
    limits = robot.data.joint_pos_limits[0]
    midpoint = ((limits[:, 0] + limits[:, 1]) * 0.5).unsqueeze(0)
    zero_vel = torch.zeros_like(midpoint)
    root = robot.data.default_root_state.clone()
    root[:, 2] = 2.0
    root[:, 7:] = 0.0
    results = {}

    for index, name in enumerate(names):
        robot.write_joint_state_to_sim(midpoint, zero_vel)
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        motor = torch.zeros_like(midpoint)
        motor[0, index] = args.torque
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        samples = {"0": {"q_delta_rad": 0.0, "velocity_rad_s": 0.0, "applied_torque_nm": 0.0}}
        for step in range(4):
            robot.set_joint_effort_target(motor)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if step + 1 in (1, 4):
                samples[str((step + 1) * 5)] = {
                    "q_delta_rad": float((robot.data.joint_pos[0, index] - midpoint[0, index]).item()),
                    "velocity_rad_s": float(robot.data.joint_vel[0, index].item()),
                    "applied_torque_nm": float(robot.data.applied_torque[0, index].item()),
                }
        results[name] = samples

    physx = robot.root_physx_view
    payload = {
        "engine": "isaac_physx_explicit_si_friction_motor_pulse",
        "contract_path": str(CONTRACT_PATH),
        "contract_version": CONTRACT["version"],
        "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        "physics_dt_s": env.physics_dt,
        "pulse_motor_torque_nm": args.torque,
        "joint_names": names,
        "gravity_disabled": True,
        "contacts_isolated": True,
        "policy_disabled": True,
        "randomization_disabled": True,
        "physx_legacy_joint_friction": physx.get_dof_friction_coefficients()[0].detach().cpu().tolist(),
        "armature": physx.get_dof_armatures()[0].detach().cpu().tolist(),
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(f"EDU3_ISAAC_PULSE=PASS output={args.out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()


