"""Portable actuator pulse probe in MuJoCo using identical explicit PD."""
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
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    data = mujoco.MjData(model)
    joint_ids = model.actuator_trnid[:, 0].astype(int)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)) for i in joint_ids]
    lab_index = np.asarray([lab_names.index(name) for name in names])
    qadr = model.jnt_qposadr[joint_ids]
    dadr = model.jnt_dofadr[joint_ids]

    data.qpos[:7] = np.asarray(initial["root_pose"], dtype=np.float64)
    data.qpos[2] += 1.0
    data.qvel[:6] = np.asarray(initial["root_velocity"], dtype=np.float64)
    data.qpos[qadr] = q_lab[lab_index]
    data.qvel[dadr] = dq_lab[lab_index]
    mujoco.mj_forward(model, data)

    target = target_lab[lab_index]
    pulse_index = names.index(source["pulse_joint"])
    pulse_torque = float(source["pulse_torque_nm"])
    diagnostic_armature = float(source["diagnostic_armature_kgm2"])
    model.dof_armature[dadr[pulse_index]] = diagnostic_armature
    mujoco.mj_forward(model, data)

    def torque():
        tau = np.zeros(len(names), dtype=np.float64)
        tau[pulse_index] = pulse_torque
        return tau

    def snap(ms, tau):
        return {
            "ms": ms,
            "q": data.qpos[qadr].tolist(),
            "dq": data.qvel[dadr].tolist(),
            "applied_torque": data.actuator_force.copy().tolist(),
            "commanded_torque": tau.tolist(),
            "target": target.tolist(),
            "root_pose": data.qpos[:7].tolist(),
            "root_velocity": data.qvel[:6].tolist(),
            "qacc_max": float(np.max(np.abs(data.qacc))),
            "contacts": int(data.ncon),
        }

    samples = [snap(0.0, torque())]
    steps = int(round(0.020 / args.dt))
    wanted = {int(round(x / args.dt)) for x in (0.005, 0.010, 0.015, 0.020)}
    for step in range(1, steps + 1):
        tau = torque()
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if step in wanted:
            samples.append(snap(step * args.dt * 1000.0, torque()))

    payload = {
        "engine": "mujoco_direct_torque_pulse",
        "dt": args.dt,
        "joint_names": names,
        "pulse_joint": source["pulse_joint"],
        "pulse_delta_rad": source["pulse_delta_rad"],
        "diagnostic_armature_kgm2": diagnostic_armature,
        "gravity_disabled": True,
        "contacts_disabled": True,
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
