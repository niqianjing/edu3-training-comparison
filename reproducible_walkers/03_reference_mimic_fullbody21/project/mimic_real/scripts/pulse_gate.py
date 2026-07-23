"""Deterministic 5 ms / 20 ms position-target pulse in Xiaohai Isaac Lab."""
import argparse
import json
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
import torch
from mimic_real.utils import task_registry
from mimic_real.envs import *  # noqa: F401,F403

def main():
    cfg, _ = task_registry.get_cfgs("hi_mimic")
    cfg.device = args.device
    cfg.scene.num_envs = 1
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg.scene.seed = 42
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.randomize_robot_friction.enable = False
    cfg.domain_rand.add_rigid_body_mass.enable = False
    cfg.domain_rand.push_robot.enable = False
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 2.0)
    cfg.scene.robot.spawn.articulation_props.fix_root_link = False
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    for actuator in cfg.scene.robot.actuators.values():
        actuator.stiffness = {".*": 0.0}
        actuator.damping = {".*": 0.0}
    env = task_registry.get_task_class("hi_mimic")(cfg, True)
    robot = env.robot
    robot.write_joint_stiffness_to_sim(torch.zeros_like(robot.data.joint_stiffness))
    robot.write_joint_damping_to_sim(torch.zeros_like(robot.data.joint_damping))
    for actuator in robot.actuators.values():
        actuator.stiffness[:] = 0.0
        actuator.damping[:] = 0.0
    names = list(robot.joint_names)
    q0 = robot.data.default_joint_pos.clone()
    dq0 = torch.zeros_like(robot.data.default_joint_vel)
    q_limits = robot.data.joint_pos_limits.clone()
    root = robot.data.default_root_state.clone()
    root[:, 2] = 2.0
    root[:, 7:] = 0.0
    full_pulses = {"hip_pitch": 0.05, "hip_roll": 0.01, "waist": 0.01, "shoulder_pitch": 0.01, "shoulder_roll": 0.01, "elbow": 0.01, "upper_arm": 0.001, "wrist": 0.00005, "thigh": 0.005, "calf": 0.005, "ankle_pitch": 0.0005, "ankle_roll": 0.0005}
    results = {}
    for idx, name in enumerate(names):
        q_init = q0.clone()
        robot.write_joint_state_to_sim(q_init, dq0)
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        efforts = torch.zeros_like(q0)
        direction = -1.0 if q0[0, idx] >= q_limits[0, idx, 1] - 1.0e-5 else 1.0
        pulse_amp = next(value for key, value in full_pulses.items() if key in name)
        efforts[0, idx] = pulse_amp * direction
        env.scene.write_data_to_sim()
        env.sim.forward()
        samples = {}
        for step in range(4):
            robot.set_joint_effort_target(efforts)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if step + 1 in (1, 4):
                samples[str((step + 1) * 5)] = {"q_delta": float((robot.data.joint_pos[0, idx] - q_init[0, idx]).item()), "velocity": float(robot.data.joint_vel[0, idx].item()), "torque": float(robot.data.applied_torque[0, idx].item())}
        results[name] = samples
    masses = robot.root_physx_view.get_masses()[0]
    inertias = robot.root_physx_view.get_inertias()[0]
    coms = robot.root_physx_view.get_coms()[0]
    payload = {"engine": "isaac", "route": "full", "dt": env.physics_dt, "joint_names": names, "total_mass_kg": float(masses.sum().item()), "body_masses_kg": dict(zip(robot.body_names, [float(x) for x in masses.tolist()])), "body_inertias": dict(zip(robot.body_names, [[float(v) for v in row] for row in inertias.tolist()])), "body_coms": dict(zip(robot.body_names, [[float(v) for v in row] for row in coms.tolist()])), "runtime_stiffness": [float(x) for x in robot.data.joint_stiffness[0].tolist()], "runtime_damping": [float(x) for x in robot.data.joint_damping[0].tolist()], "results": results}
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    print(json.dumps(payload, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()