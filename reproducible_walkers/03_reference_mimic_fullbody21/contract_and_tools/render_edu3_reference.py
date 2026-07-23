#!/usr/bin/env python3
"""Render the retargeted EDU3 reference motion without a policy."""

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", type=Path, required=True)
    ap.add_argument("--motion", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    motion = json.loads(args.motion.read_text(encoding="utf-8"))
    q = np.asarray(motion["dof_pos"], dtype=np.float64)
    root_pos = np.asarray(motion["root_trans"], dtype=np.float64)
    root_quat = np.asarray(motion["root_wxyz"], dtype=np.float64)
    names = motion["data_joint_names"]
    fps = int(motion["fps"])

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    free = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if len(free) != 1:
        raise RuntimeError(f"expected one free joint, got {len(free)}")
    root_addr = int(model.jnt_qposadr[int(free[0])])
    addrs = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(name)
        addrs.append(int(model.jnt_qposadr[jid]))

    width, height = 640, 480
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 145.0
    camera.elevation = -12.0
    camera.distance = 1.35
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")

    for i in range(len(q)):
        data.qpos[:] = 0.0
        data.qpos[root_addr:root_addr + 3] = root_pos[i]
        data.qpos[root_addr + 3:root_addr + 7] = root_quat[i]
        for value, addr in zip(q[i], addrs):
            data.qpos[addr] = value
        mujoco.mj_forward(model, data)
        camera.lookat[:] = root_pos[i]
        camera.lookat[2] = max(0.20, root_pos[i, 2] * 0.55)
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"EDU3 retarget V1  frame={i:03d}  t={i/fps:5.2f}s", (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 230, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    renderer.close()
    print(args.output)


if __name__ == "__main__":
    main()
