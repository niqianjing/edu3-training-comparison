"""Closed-loop MuJoCo replay for Xiaohai 23-DoF DeepMimic policies.

This evaluator keeps the trained policy frozen and exposes the MJCF dynamics
contract explicitly.  It is intentionally independent of Isaac Lab so that a
bad simulator conversion cannot be hidden by shared environment code.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation


LAB_NAMES = [
    "l_hip_pitch_joint", "r_hip_pitch_joint", "waist_joint",
    "l_hip_roll_joint", "r_hip_roll_joint",
    "l_shoulder_pitch_joint", "r_shoulder_pitch_joint",
    "l_thigh_joint", "r_thigh_joint",
    "l_shoulder_roll_joint", "r_shoulder_roll_joint",
    "l_calf_joint", "r_calf_joint",
    "l_upper_arm_joint", "r_upper_arm_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint",
    "l_elbow_joint", "r_elbow_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
    "l_wrist_joint", "r_wrist_joint",
]


def quat_rotate_inverse_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_vec = q[:3]
    q_w = q[3]
    return (
        v * (2.0 * q_w * q_w - 1.0)
        - np.cross(q_vec, v) * q_w * 2.0
        + q_vec * np.dot(q_vec, v) * 2.0
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="Xiaohai_model15000_MuJoCo_OFFICIAL_DYNAMICS")
    ap.add_argument("--duration", type=float, default=8.1)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--control-dt", type=float, default=0.020)
    ap.add_argument("--target-clamp", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{args.name}.mp4"
    metrics_path = out_dir / f"{args.name}_metrics.json"
    dump_path = out_dir / f"{args.name}_dump.npz"

    with open(args.motion, encoding="utf-8") as f:
        motion = json.load(f)
    fps = float(motion["fps"])
    root_trans = np.asarray(motion["root_trans"], dtype=np.float64)
    root_wxyz = np.asarray(motion["root_wxyz"], dtype=np.float64)
    motion_dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    source_names = list(motion["data_joint_names"])

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = args.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)

    actuator_joint_ids = model.actuator_trnid[:, 0].astype(int)
    mj_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(jid))
        for jid in actuator_joint_ids
    ]
    if set(mj_names) != set(LAB_NAMES):
        raise RuntimeError(f"joint-name mismatch: MJCF={mj_names}")
    qadr = model.jnt_qposadr[actuator_joint_ids]
    dadr = model.jnt_dofadr[actuator_joint_ids]
    lab_to_mj = np.asarray([LAB_NAMES.index(name) for name in mj_names])
    motion_to_mj = np.asarray([source_names.index(name) for name in mj_names])

    # The phase-0 Isaac examination starts from the first reference pose with
    # zero velocity.  Match that exact condition here.
    data.qpos[:3] = root_trans[0]
    data.qpos[3:7] = root_wxyz[0]
    data.qpos[qadr] = motion_dof[0, motion_to_mj]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(args.policy, map_location="cpu").eval()
    history = np.zeros((10, 81), dtype=np.float32)
    last_action_lab = np.zeros(23, dtype=np.float32)
    target_mj = data.qpos[qadr].copy()

    joint_ranges = model.jnt_range[actuator_joint_ids].copy()
    limited = model.jnt_limited[actuator_joint_ids].astype(bool)
    # Training actuator contract: waist/legs 24 Nm, ten arm joints 17 Nm.
    is_arm = np.asarray([
        any(key in name for key in ("shoulder", "upper_arm", "elbow", "wrist"))
        for name in mj_names
    ])
    effort = np.where(is_arm, 17.0, 24.0)

    control_stride = int(round(args.control_dt / args.dt))
    total_steps = int(round(args.duration / args.dt))
    render_stride = max(1, int(round(1.0 / (50.0 * args.dt))))

    times: list[float] = []
    heights: list[float] = []
    tilts: list[float] = []
    base_pos: list[np.ndarray] = []
    actions_lab: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    q_log: list[np.ndarray] = []
    dq_log: list[np.ndarray] = []
    raw_target_oob: list[bool] = []
    actual_target_oob: list[bool] = []
    first_fall = None

    # The official MJCF declares a 640x480 offscreen framebuffer.
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.25
    camera.azimuth = 135.0
    camera.elevation = -12.0
    writer = imageio.get_writer(video_path, fps=50, codec="libx264", quality=8)

    def phase_at(t: float) -> float:
        return min(t / (len(root_trans) / fps), 0.999999)

    def reference_mj(phase: float) -> np.ndarray:
        frame = min(int(phase * (len(motion_dof) - 1)), len(motion_dof) - 1)
        return motion_dof[frame, motion_to_mj]

    try:
        for step in range(total_steps + 1):
            t = step * args.dt
            phase = phase_at(t)
            if step % control_stride == 0:
                q_mj = data.qpos[qadr].copy()
                dq_mj = data.qvel[dadr].copy()
                q_lab = np.empty(23, dtype=np.float64)
                dq_lab = np.empty(23, dtype=np.float64)
                for mj_i, lab_i in enumerate(lab_to_mj):
                    q_lab[lab_i] = q_mj[mj_i]
                    dq_lab[lab_i] = dq_mj[mj_i]

                # MuJoCo framequat is wxyz; scipy and this helper use xyzw.
                quat_wxyz = np.asarray(data.sensor("orientation").data, dtype=np.float64)
                quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
                omega = np.asarray(data.sensor("angular-velocity").data, dtype=np.float64)
                projected_gravity = quat_rotate_inverse_xyzw(
                    quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float64)
                )
                a = 2.0 * math.pi * phase
                obs = np.concatenate([
                    omega,
                    projected_gravity,
                    q_lab,
                    dq_lab,
                    last_action_lab,
                    np.array([
                        math.sin(a), math.cos(a),
                        math.sin(2.0 * a), math.cos(2.0 * a),
                        math.sin(4.0 * a), math.cos(4.0 * a),
                    ]),
                ]).astype(np.float32)
                if step == 0:
                    # Capture-RSI training repeats the first synchronized
                    # observation into all ten history slots.
                    history[:] = obs
                else:
                    history[:-1] = history[1:]
                    history[-1] = obs
                with torch.inference_mode():
                    action_lab = policy(torch.from_numpy(history.reshape(1, -1))).cpu().numpy()[0]
                last_action_lab = action_lab.astype(np.float32)
                action_mj = action_lab[lab_to_mj]
                raw_target = reference_mj(phase) + action_mj * 0.25
                raw_target_oob.append(bool(np.any(limited & ((raw_target < joint_ranges[:, 0]) | (raw_target > joint_ranges[:, 1])))))
                if args.target_clamp:
                    margin = 0.05 * (joint_ranges[:, 1] - joint_ranges[:, 0])
                    low = joint_ranges[:, 0] + margin
                    high = joint_ranges[:, 1] - margin
                    target_mj = np.where(limited, np.clip(raw_target, low, high), raw_target)
                else:
                    target_mj = raw_target
                actual_target_oob.append(bool(np.any(limited & ((target_mj < joint_ranges[:, 0]) | (target_mj > joint_ranges[:, 1])))))

            tau = 80.0 * (target_mj - data.qpos[qadr]) - 1.0 * data.qvel[dadr]
            tau = np.clip(tau, -effort, effort)
            data.ctrl[:] = tau

            if step < total_steps:
                mujoco.mj_step(model, data)

            root_quat_wxyz = data.qpos[3:7].copy()
            rot = Rotation.from_quat(root_quat_wxyz[[1, 2, 3, 0]])
            up = rot.apply(np.array([0.0, 0.0, 1.0]))
            tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
            height = float(data.qpos[2])
            if first_fall is None and (height < 0.35 or tilt > 60.0):
                first_fall = t

            times.append(t)
            heights.append(height)
            tilts.append(tilt)
            base_pos.append(data.qpos[:3].copy())
            actions_lab.append(last_action_lab.copy())
            targets.append(target_mj.copy())
            torques.append(tau.copy())
            q_log.append(data.qpos[qadr].copy())
            dq_log.append(data.qvel[dadr].copy())

            if step % render_stride == 0:
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    heights_a = np.asarray(heights)
    tilts_a = np.asarray(tilts)
    base_a = np.asarray(base_pos)
    tau_a = np.asarray(torques)
    dq_a = np.asarray(dq_log)
    metrics = {
        "duration_s": args.duration,
        "fell": first_fall is not None,
        "fall_time_s": first_fall,
        "delta_xyz_m": (base_a[-1] - base_a[0]).tolist(),
        "min_root_height_m": float(heights_a.min()),
        "max_tilt_deg": float(tilts_a.max()),
        "max_abs_joint_velocity_rad_s": float(np.max(np.abs(dq_a))),
        "raw_target_oob_control_step_rate": float(np.mean(raw_target_oob)),
        "actual_target_oob_control_step_rate": float(np.mean(actual_target_oob)),
        "target_clamp_enabled": bool(args.target_clamp),
        "joint_names_mujoco": mj_names,
        "armature_by_joint": {
            name: float(model.dof_armature[dadr[i]]) for i, name in enumerate(mj_names)
        },
        "damping_by_joint": {
            name: float(model.dof_damping[dadr[i]]) for i, name in enumerate(mj_names)
        },
        "torque_rms_nm": {
            name: float(np.sqrt(np.mean(tau_a[:, i] ** 2))) for i, name in enumerate(mj_names)
        },
        "torque_saturation_rate": {
            name: float(np.mean(np.abs(tau_a[:, i]) >= effort[i] - 1e-6)) for i, name in enumerate(mj_names)
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        dump_path,
        time=np.asarray(times), heights=heights_a, tilts_deg=tilts_a,
        base_positions=base_a, actions_lab=np.asarray(actions_lab),
        targets_mujoco=np.asarray(targets), torques_mujoco=tau_a,
        q_mujoco=np.asarray(q_log), dq_mujoco=dq_a,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
