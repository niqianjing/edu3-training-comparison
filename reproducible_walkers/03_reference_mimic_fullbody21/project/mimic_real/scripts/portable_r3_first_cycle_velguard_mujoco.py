"""First 20 ms MuJoCo parity probe using explicit torque PD."""
import argparse
import json
import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.001)
    args = ap.parse_args()

    with open(args.isaac, encoding="utf-8") as f:
        source = json.load(f)
    lab_names = source["joint_names"]
    initial = source["samples"][0]
    target_lab = np.asarray(initial["target"], dtype=np.float64)
    q_lab = np.asarray(initial["q"], dtype=np.float64)
    dq_lab = np.asarray(initial["dq"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = args.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)

    joint_ids = model.actuator_trnid[:, 0].astype(int)
    mj_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(jid))
        for jid in joint_ids
    ]
    lab_index = np.asarray([lab_names.index(name) for name in mj_names])
    qadr = model.jnt_qposadr[joint_ids]
    dadr = model.jnt_dofadr[joint_ids]

    data.qpos[:7] = np.asarray(initial["root_pose"], dtype=np.float64)
    data.qpos[qadr] = q_lab[lab_index]
    data.qvel[:6] = np.asarray(initial.get("root_velocity", np.zeros(6)), dtype=np.float64)
    data.qvel[dadr] = dq_lab[lab_index]
    mujoco.mj_forward(model, data)

    target = target_lab[lab_index]
    effort = np.max(np.abs(model.actuator_ctrlrange), axis=1)

    samples = []

    def torque():
        return np.clip(80.0 * (target - data.qpos[qadr]) - data.qvel[dadr], -effort, effort)

    def snap(ms, tau):
        return {
            "ms": ms,
            "q": data.qpos[qadr].tolist(),
            "dq": data.qvel[dadr].tolist(),
            "applied_torque": tau.tolist(),
            "target": target.tolist(),
            "root_pose": data.qpos[:7].tolist(),
            "qacc_max": float(np.max(np.abs(data.qacc))),
            "contacts": int(data.ncon),
        }

    tau = torque()
    samples.append(snap(0.0, tau))
    steps = int(round(0.020 / args.dt))
    wanted = {
        int(round(0.005 / args.dt)),
        int(round(0.010 / args.dt)),
        int(round(0.015 / args.dt)),
        steps,
    }
    for step in range(1, steps + 1):
        tau = torque()
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        # Diagnostic parity guard: same operational joint-speed ceiling as PhysX.
        data.qvel[dadr] = np.clip(data.qvel[dadr], -6.28, 6.28)
        if step in wanted:
            samples.append(snap(step * args.dt * 1000.0, data.actuator_force.copy()))

    payload = {
        "engine": "mujoco_explicit_ideal_pd",
        "dt": args.dt,
        "joint_names": mj_names,
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
