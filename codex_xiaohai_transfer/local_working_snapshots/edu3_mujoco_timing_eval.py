"""Closed-loop MuJoCo replay for an exported EDU3/Xiaohai policy."""

from __future__ import print_function

import argparse
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


NAMES = [
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint", "r_hip_pitch_joint",
    "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint",
    "r_ankle_pitch_joint", "r_ankle_roll_joint",
]
DEFAULT = np.asarray([-0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
                       0.10, 0.0, 0.0, -0.30, 0.20, 0.0])
LOWER = np.asarray([-1.5708, -0.3491, -0.7854, 0.0, -0.4363, -0.4363,
                    -0.7854, -0.7854, -0.7854, -1.2217, -0.4363, -0.4363])
UPPER = np.asarray([0.7854, 0.7854, 0.7854, 1.2217, 0.4363, 0.4363,
                    1.5708, 0.3491, 0.7854, 0.0, 0.4363, 0.4363])


def make_mjcf(urdf, out_xml, roll25):
    urdf_tree = ET.parse(urdf)
    urdf_root = urdf_tree.getroot()
    source_dir = os.path.dirname(os.path.abspath(urdf))
    for mesh in urdf_root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and not os.path.isabs(filename):
            mesh.set("filename", os.path.abspath(os.path.join(source_dir, filename)))
    # Preserve original STL visual meshes while keeping collision primitives for physics.
    mujoco_node = ET.SubElement(urdf_root, "mujoco")
    ET.SubElement(mujoco_node, "compiler", {
        "discardvisual": "false", "strippath": "false",
    })
    urdf_root.insert(0, ET.Element("link", {"name": "mujoco_world"}))
    floating = ET.Element("joint", {"name": "mujoco_free", "type": "floating"})
    ET.SubElement(floating, "parent", {"link": "mujoco_world"})
    ET.SubElement(floating, "child", {"link": "base_link"})
    urdf_root.append(floating)
    floating_urdf = out_xml.replace(".xml", "_floating.urdf")
    urdf_tree.write(floating_urdf, encoding="utf-8", xml_declaration=True)
    src = mujoco.MjModel.from_xml_path(floating_urdf)
    mujoco.mj_saveLastXML(out_xml, src)
    tree = ET.parse(out_xml)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.001")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", {
        "name": "ground", "type": "plane", "pos": "0 0 0",
        "size": "0 0 0.05", "friction": "0.8 0.005 0.0001",
        "condim": "3", "group": "2", "rgba": "0.25 0.28 0.32 1",
    })
    old = root.find("actuator")
    if old is not None:
        root.remove(old)
    actuator = ET.SubElement(root, "actuator")
    for name in NAMES:
        is25 = "hip_pitch" in name or "calf" in name or (roll25 and "hip_roll" in name)
        limit = 25.0 if is25 else 10.0
        ET.SubElement(actuator, "motor", {
            "name": "m_" + name, "joint": name, "gear": "1",
            "ctrllimited": "true", "ctrlrange": "%g %g" % (-limit, limit),
        })
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def rpy_from_wxyz(q):
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--roll25", action="store_true")
    ap.add_argument("--control_ms", type=int, required=True)
    ap.add_argument("--full_period_steps", type=float, required=True)
    ap.add_argument("--no_video", action="store_true")
    args = ap.parse_args()

    out_xml = args.out_prefix + "_compiled.xml"
    make_mjcf(args.urdf, out_xml, args.roll25)
    model = mujoco.MjModel.from_xml_path(out_xml)
    data = mujoco.MjData(model)
    policy = torch.jit.load(args.policy, map_location="cpu")
    policy.eval()

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in NAMES]
    qadr = np.asarray([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
    dadr = np.asarray([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
    act_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_" + n) for n in NAMES])
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    free_id = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    base_qadr = int(model.jnt_qposadr[free_id])

    is25 = np.asarray([
        ("hip_pitch" in n or "calf" in n or (args.roll25 and "hip_roll" in n))
        for n in NAMES
    ])
    peak = np.where(is25, 25.0, 10.0)
    continuous = np.where(is25, 7.0, 3.75)
    armature = np.where(is25, 0.02649, 0.0067367)
    coulomb = np.where(is25, 0.51, 0.146)
    viscous = np.where(is25, 0.0432, 0.0306)
    kp = np.asarray([100, 80, 80, 100, 40, 40, 100, 80, 80, 100, 40, 40], dtype=float)
    kd = np.asarray([3, 2.5, 2.5, 3, 1.5, 1.5, 3, 2.5, 2.5, 3, 1.5, 1.5], dtype=float)
    if args.roll25:
        kp[[1, 7]] = 100.0
        kd[[1, 7]] = 3.0
    model.dof_armature[dadr] = armature
    model.dof_damping[dadr] = 0.0
    model.dof_frictionloss[dadr] = 0.0

    mujoco.mj_resetData(model, data)
    data.qpos[base_qadr:base_qadr + 3] = [0.0, 0.0, 0.40]
    data.qpos[base_qadr + 3:base_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = DEFAULT
    mujoco.mj_forward(model, data)

    renderer = None
    camera = None
    scene_option = None
    frames = []
    if not args.no_video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 130.0
        camera.elevation = -12.0
        camera.distance = 1.7
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[0] = 0  # Hide collision boxes and spheres.
        scene_option.geomgroup[1] = 1  # Show original STL visual meshes.
        scene_option.geomgroup[2] = 1  # Show ground.
    logs = {k: [] for k in ["pos", "quat", "q", "dq", "target", "motor", "friction", "net"]}
    phase = 0.0
    action = np.zeros(12)
    target = DEFAULT.copy()
    total_steps = 8000
    control_decimation = args.control_ms
    fell = False
    for step in range(total_steps):
        q = data.qpos[qadr].copy()
        dq = data.qvel[dadr].copy()
        quat = data.xquat[base_id].copy()
        rpy = rpy_from_wxyz(quat)
        if step % control_decimation == 0:
            phase += 1.0 / args.full_period_steps
            omega_world = data.qvel[3:6].copy()
            omega_body = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).apply(omega_world, inverse=True)
            obs = np.concatenate([
                [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
                [0.3, 0.0, 0.0], q, dq, omega_body, rpy,
            ]).astype(np.float32)
            obs = np.clip(obs, -20.0, 20.0)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0))[0].cpu().numpy()
            action = np.clip(action, -10.0, 10.0)
            target = DEFAULT + action
        motor = np.clip(kp * (target - q) - kd * dq, -peak, peak)
        friction = coulomb * np.tanh(dq / 0.01) + viscous * dq
        net = np.clip(motor - friction, -peak, peak)
        data.ctrl[act_ids] = net
        mujoco.mj_step(model, data)
        if step % control_decimation == 0:
            logs["pos"].append(data.xpos[base_id].copy())
            logs["quat"].append(data.xquat[base_id].copy())
            logs["q"].append(q)
            logs["dq"].append(dq)
            logs["target"].append(target.copy())
            logs["motor"].append(motor)
            logs["friction"].append(friction)
            logs["net"].append(net)
        if renderer is not None and step % 20 == 0:
            camera.lookat[:] = data.xpos[base_id]
            camera.lookat[2] = max(0.3, camera.lookat[2])
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            frames.append(renderer.render().copy())
        if data.xpos[base_id, 2] < 0.30 or abs(rpy[0]) > 1.15 or abs(rpy[1]) > 1.15:
            fell = True
            break

    arr = {k: np.asarray(v) for k, v in logs.items()}
    rpy = np.asarray([rpy_from_wxyz(q) for q in arr["quat"]])
    span = UPPER - LOWER
    joints = {}
    for i, name in enumerate(NAMES):
        rms = float(np.sqrt(np.mean(arr["motor"][:, i] ** 2)))
        joints[name] = {
            "q_mean_deg": float(np.degrees(np.mean(arr["q"][:, i]))),
            "q_min_deg": float(np.degrees(np.min(arr["q"][:, i]))),
            "q_max_deg": float(np.degrees(np.max(arr["q"][:, i]))),
            "lower_edge_rate": float(np.mean(arr["q"][:, i] <= LOWER[i] + 0.10 * span[i])),
            "upper_edge_rate": float(np.mean(arr["q"][:, i] >= UPPER[i] - 0.10 * span[i])),
            "motor_rms_Nm": rms,
            "continuous_ratio": float(rms / continuous[i]),
            "peak_saturation_rate": float(np.mean(np.abs(arr["motor"][:, i]) >= 0.99 * peak[i])),
        }
    result = {
        "engine": "mujoco", "fell": bool(fell), "duration_s": float(data.time),
        "forward_m": float(arr["pos"][-1, 0] - arr["pos"][0, 0]),
        "lateral_m": float(arr["pos"][-1, 1] - arr["pos"][0, 1]),
        "min_height_m": float(np.min(arr["pos"][:, 2])),
        "yaw_change_deg": float(np.degrees(np.unwrap(rpy[:, 2])[-1] - np.unwrap(rpy[:, 2])[0])),
        "max_abs_roll_deg": float(np.degrees(np.max(np.abs(rpy[:, 0])))),
        "max_abs_pitch_deg": float(np.degrees(np.max(np.abs(rpy[:, 1])))),
        "target_oob_total": int(np.sum(np.logical_or(arr["target"] < LOWER, arr["target"] > UPPER))),
        "joints": joints,
    }
    np.savez(args.out_prefix + ".npz", names=np.asarray(NAMES), **arr)
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(result, f, indent=2)
    print("EDU3_MUJOCO_RESULT", json.dumps(result, sort_keys=True))
    if renderer is not None:
        video_path = args.out_prefix + ".mp4"
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (640, 480))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        if hasattr(renderer, "close"):
            renderer.close()
        print("EDU3_MUJOCO_VIDEO", video_path)


if __name__ == "__main__":
    main()



