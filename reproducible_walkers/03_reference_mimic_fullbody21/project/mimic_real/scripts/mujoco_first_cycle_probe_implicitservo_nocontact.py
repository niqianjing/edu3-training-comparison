"""Replay the frozen first Isaac policy target for exactly 20 ms in MuJoCo."""
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
    ap.add_argument("--velocity-clamp", action="store_true")
    args = ap.parse_args()

    source = json.load(open(args.isaac, encoding="utf-8"))
    lab_names = source["joint_names"]
    initial = source["samples"][0]
    target_lab = np.asarray(initial["target"], dtype=np.float64)
    q_lab = np.asarray(initial["q"], dtype=np.float64)
    dq_lab = np.asarray(initial["dq"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = args.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    joint_ids = model.actuator_trnid[:, 0].astype(int)
    mj_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)) for i in joint_ids]
    lab_index = np.asarray([lab_names.index(name) for name in mj_names])
    qadr = model.jnt_qposadr[joint_ids]
    dadr = model.jnt_dofadr[joint_ids]

    data.qpos[qadr] = q_lab[lab_index]
    data.qvel[dadr] = dq_lab[lab_index]
    data.qpos[:7] = np.asarray(initial["root_pose"], dtype=np.float64)
    data.qvel[:6] = 0.0
    mujoco.mj_forward(model, data)

    target = target_lab[lab_index]
    effort = np.max(np.abs(model.actuator_ctrlrange), axis=1)
    hard = model.jnt_range[joint_ids].astype(np.float64)
    centers = 0.5 * (hard[:, 0] + hard[:, 1])
    half = 0.5 * (hard[:, 1] - hard[:, 0]) * 0.95
    target = np.clip(target, centers - half, centers + half)
    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    model.actuator_gainprm[:, 0] = 80.0
    model.actuator_biasprm[:, 1] = -80.0
    model.actuator_biasprm[:, 2] = -1.0
    model.actuator_ctrllimited[:] = 1
    model.actuator_ctrlrange[:, 0] = centers - half
    model.actuator_ctrlrange[:, 1] = centers + half
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:, 0] = -effort
    model.actuator_forcerange[:, 1] = effort
    samples = []

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

    tau = np.clip((target - data.qpos[qadr]) * 80.0 - data.qvel[dadr], -effort, effort)
    samples.append(snap(0.0, tau))
    steps = int(round(0.020 / args.dt))
    wanted = {int(round(0.005 / args.dt)), int(round(0.010 / args.dt)), int(round(0.015 / args.dt)), steps}
    for step in range(1, steps + 1):
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        tau = data.actuator_force.copy()
        if args.velocity_clamp:
            data.qvel[dadr] = np.clip(data.qvel[dadr], -6.28, 6.28)
        if step in wanted:
            samples.append(snap(step * args.dt * 1000.0, tau))

    payload = {
        "engine": "mujoco",
        "dt": args.dt,
        "velocity_clamp": args.velocity_clamp,
        "joint_names": mj_names,
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()



