import argparse
import json
import math
import os
from collections import deque

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

class MotionLoader:
    """Minimal walk.json loader matching the original frame-index contract."""

    def __init__(self, path, simulator_joint_names, device="cpu", add_static_frame=False):
        if add_static_frame:
            raise ValueError("This evaluator intentionally uses the original walk without added static frames")
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.root_trans = torch.tensor(loaded["root_trans"], dtype=torch.float32, device=device)
        self.root_rot = torch.tensor(loaded["root_wxyz"], dtype=torch.float32, device=device)
        joint_index = [loaded["data_joint_names"].index(name) for name in simulator_joint_names]
        self.dof_pos = torch.tensor(loaded["dof_pos"], dtype=torch.float32, device=device)[:, joint_index]
        self.fps = float(loaded["fps"])
        self.frame_num = int(self.root_trans.shape[0])
        self.record_time = self.frame_num / self.fps

    def _ids(self, phase):
        return (phase * (self.frame_num - 1)).to(torch.long)

    def get_dof_pos_batch(self, phase):
        return self.dof_pos[self._ids(phase)]

    def get_root_trans_batch(self, phase):
        return self.root_trans[self._ids(phase)]

    def get_root_rot_batch(self, phase):
        return self.root_rot[self._ids(phase)]

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
MJ_NAMES = [
    "waist_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_upper_arm_joint", "l_elbow_joint", "l_wrist_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_upper_arm_joint", "r_elbow_joint", "r_wrist_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
]
N = 23
SINGLE_OBS = 81
HISTORY = 10
TAU_LIMIT = np.asarray([24.0] + [17.0] * 10 + [24.0] * 12, dtype=np.float64)
KP = np.full(N, 80.0)
KD = np.full(N, 1.0)


def qinv_apply(q_xyzw, v):
    qv = q_xyzw[:3]
    qw = q_xyzw[3]
    return v * (2.0 * qw * qw - 1.0) - 2.0 * qw * np.cross(qv, v) + 2.0 * qv * np.dot(qv, v)


def mappings():
    mj_to_lab = [LAB_NAMES.index(name) for name in MJ_NAMES]
    lab_to_mj = [MJ_NAMES.index(name) for name in LAB_NAMES]
    return mj_to_lab, lab_to_mj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--duration", type=float, default=8.1666667)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    motion = MotionLoader(args.motion, MJ_NAMES, device="cpu", add_static_frame=False)
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    policy = torch.jit.load(args.model, map_location="cpu")
    policy.eval()

    data.qpos[-N:] = motion.get_dof_pos_batch(torch.tensor([0.0]))[0].cpu().numpy()
    data.qpos[:3] = motion.get_root_trans_batch(torch.tensor([0.0]))[0].cpu().numpy()
    data.qpos[3:7] = motion.get_root_rot_batch(torch.tensor([0.0]))[0].cpu().numpy()
    mujoco.mj_forward(model, data)

    mj_to_lab, lab_to_mj = mappings()
    hist = np.zeros((1, SINGLE_OBS * HISTORY), dtype=np.float32)
    last_action_lab = np.zeros(N, dtype=np.float64)
    target_mj = data.qpos[-N:].copy()

    total_steps = int(round(args.duration / model.opt.timestep))
    decimation = 20
    render_every = max(1, int(round(1.0 / (args.fps * model.opt.timestep))))
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 135.0
    cam.elevation = -12.0
    cam.distance = 2.2
    writer = imageio.get_writer(args.video, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)

    times, roots, eulers, qlabs, refs_lab, actions, taus, targets = [], [], [], [], [], [], [], []
    fell_at = None

    try:
        for count in range(total_steps):
            if count % decimation == 0:
                q = data.qpos[-N:].astype(np.float64)
                dq = data.qvel[-N:].astype(np.float64)
                q_lab = q[np.asarray(lab_to_mj)]
                dq_lab = dq[np.asarray(lab_to_mj)]

                quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float64)
                omega = data.sensor("angular-velocity").data.astype(np.float64)
                gravity = qinv_apply(quat, np.asarray([0.0, 0.0, -1.0]))
                phase = (count * model.opt.timestep / motion.record_time) % 0.99
                a = 2.0 * math.pi * phase
                obs = np.zeros((1, SINGLE_OBS), dtype=np.float32)
                obs[0, 0:3] = omega
                obs[0, 3:6] = gravity
                obs[0, 6:29] = q_lab
                obs[0, 29:52] = dq_lab
                obs[0, 52:75] = last_action_lab
                obs[0, 75:81] = [math.sin(a), math.cos(a), math.sin(2*a), math.cos(2*a), math.sin(4*a), math.cos(4*a)]
                hist = np.concatenate((hist[:, SINGLE_OBS:], obs), axis=1).astype(np.float32)
                with torch.inference_mode():
                    action_lab = policy(torch.from_numpy(hist)).detach().cpu().numpy()[0]
                action_lab = np.clip(action_lab, -100.0, 100.0)
                last_action_lab = action_lab.copy()

                delta_mj = action_lab[np.asarray(mj_to_lab)] * 0.25
                ref_mj = motion.get_dof_pos_batch(torch.tensor([phase]))[0].cpu().numpy()
                target_mj = ref_mj + delta_mj
                ref_lab = ref_mj[np.asarray(lab_to_mj)]

                rot = R.from_quat(quat)
                euler = rot.as_euler("xyz", degrees=True)
                times.append(count * model.opt.timestep)
                roots.append(data.qpos[:3].copy())
                eulers.append(euler)
                qlabs.append(q_lab.copy())
                refs_lab.append(ref_lab.copy())
                actions.append(action_lab.copy())
                targets.append(target_mj.copy())

                tilt = float(np.hypot(euler[0], euler[1]))
                if fell_at is None and (data.qpos[2] < 0.25 or tilt > 60.0):
                    fell_at = count * model.opt.timestep

            q = data.qpos[-N:].astype(np.float64)
            dq = data.qvel[-N:].astype(np.float64)
            tau = np.clip((target_mj - q) * KP - dq * KD, -TAU_LIMIT, TAU_LIMIT)
            data.ctrl[:] = tau
            if count % decimation == 0:
                taus.append(tau.copy())
            mujoco.mj_step(model, data)

            if count % render_every == 0:
                cam.lookat[:] = data.qpos[:3]
                cam.lookat[2] = max(0.35, data.qpos[2] * 0.75)
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    arr_root = np.asarray(roots)
    arr_euler = np.asarray(eulers)
    arr_q = np.asarray(qlabs)
    arr_ref = np.asarray(refs_lab)
    arr_tau = np.asarray(taus)
    arr_action = np.asarray(actions)
    arr_target = np.asarray(targets)
    sat = np.mean(np.abs(arr_tau) >= (TAU_LIMIT[None, :] - 1e-6), axis=0)
    metrics = {
        "duration_s": args.duration,
        "fell": fell_at is not None,
        "fall_time_s": fell_at,
        "delta_x_m": float(arr_root[-1, 0] - arr_root[0, 0]),
        "delta_y_m": float(arr_root[-1, 1] - arr_root[0, 1]),
        "horizontal_distance_m": float(np.linalg.norm(arr_root[-1, :2] - arr_root[0, :2])),
        "min_root_height_m": float(np.min(arr_root[:, 2])),
        "max_tilt_deg": float(np.max(np.hypot(arr_euler[:, 0], arr_euler[:, 1]))),
        "yaw_change_deg": float(arr_euler[-1, 2] - arr_euler[0, 2]),
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean((arr_q - arr_ref) ** 2))),
        "torque_rms_Nm": {name: float(v) for name, v in zip(MJ_NAMES, np.sqrt(np.mean(arr_tau ** 2, axis=0)))},
        "torque_saturation_rate": {name: float(v) for name, v in zip(MJ_NAMES, sat)},
        "arm_ranges_deg": {
            name: [float(np.degrees(np.min(arr_q[:, LAB_NAMES.index(name)]))), float(np.degrees(np.max(arr_q[:, LAB_NAMES.index(name)])))]
            for name in LAB_NAMES if any(k in name for k in ("shoulder", "upper_arm", "elbow", "wrist"))
        },
    }
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.savez_compressed(args.raw, time=np.asarray(times), root=arr_root, euler_deg=arr_euler, q_lab=arr_q,
                        ref_lab=arr_ref, action_lab=arr_action, tau_mj=arr_tau, target_mj=arr_target,
                        lab_names=np.asarray(LAB_NAMES), mj_names=np.asarray(MJ_NAMES))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

