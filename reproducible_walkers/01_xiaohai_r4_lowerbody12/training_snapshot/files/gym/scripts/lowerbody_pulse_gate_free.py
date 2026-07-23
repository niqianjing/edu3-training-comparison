"""Deterministic 5 ms / 20 ms free-root pulse for Xiaohai Isaac Gym."""
import json

from isaacgym import gymtorch
import torch

from gym.envs import *  # noqa: F401,F403
from gym.utils import get_args, task_registry


def main():
    args = get_args()
    cfg = task_registry.env_cfgs[args.task]
    cfg.env.num_envs = 1
    cfg.sim.gravity = [0.0, 0.0, 0.0]
    cfg.seed = 42
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
    params = env.gym.get_sim_params(env.sim)
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
        pulse_tau = torch.zeros_like(q0)
        pulse_tau[:, idx] = 0.1
        samples = {}
        for step in range(20):
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(pulse_tau))
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_dof_state_tensor(env.sim)
            if step + 1 in (5, 20):
                samples[str(step + 1)] = {
                    "q_delta": float((env.dof_pos[0, idx] - q0[0, idx]).item()),
                    "velocity": float(env.dof_vel[0, idx].item()),
                    "torque": float(pulse_tau[0, idx].item()),
                }
        results[name] = samples
    props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
    payload = {
        "engine": "isaac", "route": "lower", "root": "free", "gravity": 0.0,
        "dt": float(params.dt), "joint_names": names,
        "total_mass_kg": sum(float(p.mass) for p in props), "results": results,
    }
    out = "/home/zero/xiaohai_training/lowerbody_isaac_pulse_free_aligned.json"
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()



