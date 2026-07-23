"""No-gravity/no-contact 5 ms and 20 ms pulse export for EDU3 Isaac Gym."""

import argparse
import json

import isaacgym
import torch
from isaacgym import gymtorch

from gym.envs import *  # noqa: F401,F403
from gym.envs.edu3_12 import edu3_tasks  # noqa: F401
from gym.utils import get_args, task_registry


def main():
    args = get_args()
    cfg = task_registry.env_cfgs[args.task]
    cfg.env.num_envs = 1
    cfg.seed = 42
    cfg.sim.gravity = [0.0, 0.0, 0.0]
    cfg.terrain.mesh_type = "plane"
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.push_robots = False
    cfg.asset.fix_base_link = False
    cfg.asset.self_collisions = 1
    cfg.init_state.pos = [0.0, 0.0, 2.0]
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    dt = float(env.gym.get_sim_params(env.sim).dt)
    checkpoints = {int(round(0.005 / dt)): "5ms", int(round(0.020 / dt)): "20ms"}
    if any(abs(k * dt - (0.005 if v == "5ms" else 0.020)) > 1e-9 for k, v in checkpoints.items()):
        raise RuntimeError(f"timestep {dt} cannot hit 5/20 ms exactly")

    names = list(env.dof_names)
    q0 = env.default_dof_pos.clone()
    root0 = env.root_states.clone()
    root0[:, 0:3] = torch.tensor([0.0, 0.0, 2.0], device=root0.device)
    root0[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=root0.device)
    root0[:, 7:] = 0.0
    results = {}
    for idx, name in enumerate(names):
        env.dof_pos[:] = q0
        env.dof_vel[:] = 0.0
        env.root_states[:] = root0
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        pulse = torch.zeros_like(q0)
        pulse[:, idx] = 0.1
        samples = {}
        for step in range(1, max(checkpoints) + 1):
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(pulse))
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
            if step in checkpoints:
                samples[checkpoints[step]] = {
                    "q_delta": float((env.dof_pos[0, idx] - q0[0, idx]).item()),
                    "velocity": float(env.dof_vel[0, idx].item()),
                }
        results[name] = samples
    payload = {"engine": "isaac", "dt": dt, "joint_names": names, "results": results}
    out = f"/home/zero/xiaohai_training/{args.task}_isaac_pulse.json"
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    print(out)


if __name__ == "__main__":
    main()
