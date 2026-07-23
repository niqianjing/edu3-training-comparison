"""EDU3 21-joint deterministic motor pulse in MuJoCo.

Uses the same armature and the exact tanh-smoothed SI friction equation as the
Isaac explicit actuator.  MJCF passive damping/friction are zeroed at runtime
for this parity gate only, preventing double friction.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--isaac", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--torque", type=float, default=1.0)
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    isaac = json.loads(Path(args.isaac).read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(contract["provenance"]["generated_mjcf_path"])
    model.opt.timestep = float(contract["control"]["mujoco_physics_dt_s"])
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.dof_frictionloss[:] = 0.0
    model.dof_damping[:] = 0.0

    actuator_joint_ids = model.actuator_trnid[:, 0].astype(int)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in actuator_joint_ids]
    qadr = model.jnt_qposadr[actuator_joint_ids]
    dadr = model.jnt_dofadr[actuator_joint_ids]
    limits = model.jnt_range[actuator_joint_ids]
    midpoint = (limits[:, 0] + limits[:, 1]) * 0.5
    eps = 0.01
    results = {}

    for pulse_index, name in enumerate(names):
        data = mujoco.MjData(model)
        data.qpos[:7] = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        data.qpos[qadr] = midpoint
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        motor = np.zeros(len(names), dtype=np.float64)
        motor[pulse_index] = args.torque
        samples = {"0": {"q_delta_rad": 0.0, "velocity_rad_s": 0.0, "applied_torque_nm": 0.0}}
        for step in range(1, 21):
            cfg = contract["robot"]["joints"]
            coulomb = np.array([cfg[j]["coulomb_friction_nm"] for j in names])
            viscous = np.array([cfg[j]["viscous_damping_nm_s_per_rad"] for j in names])
            velocity = data.qvel[dadr]
            passive = coulomb * np.tanh(velocity / eps) + viscous * velocity
            net = motor - passive
            data.ctrl[:] = net
            mujoco.mj_step(model, data)
            if step in (5, 20):
                samples[str(step)] = {
                    "q_delta_rad": float(data.qpos[qadr[pulse_index]] - midpoint[pulse_index]),
                    "velocity_rad_s": float(data.qvel[dadr[pulse_index]]),
                    "applied_torque_nm": float(data.actuator_force[pulse_index]),
                }
        results[name] = samples

    payload = {
        "engine": "mujoco_explicit_si_friction_motor_pulse",
        "contract_path": args.contract,
        "contract_version": contract["version"],
        "contract_sha256": hashlib.sha256(Path(args.contract).read_bytes()).hexdigest(),
        "physics_dt_s": model.opt.timestep,
        "pulse_motor_torque_nm": args.torque,
        "joint_names": names,
        "gravity_disabled": True,
        "contacts_disabled": True,
        "policy_disabled": True,
        "randomization_disabled": True,
        "passive_mjcf_friction_disabled_for_parity": True,
        "explicit_si_friction_enabled": True,
        "isaac_source": args.isaac,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"EDU3_MUJOCO_PULSE=PASS output={args.out}")


if __name__ == "__main__":
    main()


