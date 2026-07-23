#!/usr/bin/env python3
"""Retarget Xiaohai 23-DoF walk reference to EDU3 21-DoF semantics.

The first version deliberately preserves the motion shape relative to frame 0:
target_q(t) = edu3_default + gain * sign * (source_q(t) - source_q(0)).
Gain starts at 1 and is reduced only when needed to stay inside 98% of the
student's measured joint range.  Target-link positions are recomputed from the
student MJCF with MuJoCo forward kinematics; source key-point coordinates are
never scaled and reused as if the two bodies were identical.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


MAPPING = [
    ("torso_joint", "waist_joint", +1.0, 0.0),
    ("left_thigh_pitch_joint", "l_hip_pitch_joint", +1.0, -0.10),
    ("left_thigh_roll_joint", "l_hip_roll_joint", +1.0, 0.0),
    ("left_thigh_yaw_joint", "l_thigh_joint", +1.0, 0.0),
    ("left_knee_joint", "l_calf_joint", +1.0, 0.30),
    ("left_ankle_pitch_joint", "l_ankle_pitch_joint", +1.0, -0.20),
    ("left_ankle_roll_joint", "l_ankle_roll_joint", +1.0, 0.0),
    ("right_thigh_pitch_joint", "r_hip_pitch_joint", -1.0, 0.10),
    ("right_thigh_roll_joint", "r_hip_roll_joint", +1.0, 0.0),
    ("right_thigh_yaw_joint", "r_thigh_joint", +1.0, 0.0),
    ("right_knee_joint", "r_calf_joint", -1.0, -0.30),
    ("right_ankle_pitch_joint", "r_ankle_pitch_joint", -1.0, 0.20),
    ("right_ankle_roll_joint", "r_ankle_roll_joint", +1.0, 0.0),
    ("left_arm_pitch_joint", "l_shoulder_pitch_joint", +1.0, -0.15),
    ("left_arm_roll_joint", "l_shoulder_roll_joint", +1.0, 0.05),
    ("left_arm_yaw_joint", "l_upper_arm_joint", +1.0, 0.0),
    ("left_elbow_pitch_joint", "l_elbow_joint", -1.0, 0.60),
    ("right_arm_pitch_joint", "r_shoulder_pitch_joint", -1.0, 0.15),
    ("right_arm_roll_joint", "r_shoulder_roll_joint", +1.0, -0.05),
    ("right_arm_yaw_joint", "r_upper_arm_joint", +1.0, 0.0),
    ("right_elbow_pitch_joint", "r_elbow_joint", +1.0, -0.60),
]


TARGET_LINKS = [
    "left_thigh_pitch_link", "left_knee_link", "left_ankle_roll_link",
    "right_thigh_pitch_link", "right_knee_link", "right_ankle_roll_link",
    "left_arm_pitch_link", "left_arm_yaw_link", "left_elbow_pitch_link",
    "right_arm_pitch_link", "right_arm_yaw_link", "right_elbow_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link", "torso_link",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_meshes(xml_path: Path) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    for geom in list(root.iter("geom")):
        if geom.attrib.get("type") == "mesh" or "mesh" in geom.attrib:
            parent = next((p for p in root.iter() if geom in list(p)), None)
            if parent is not None:
                parent.remove(geom)
    return ET.tostring(root, encoding="unicode")


def gain_to_fit(delta: np.ndarray, default: float, low: float, high: float, margin: float) -> float:
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low) * margin
    safe_low, safe_high = mid - half, mid + half
    gain = 1.0
    dmin, dmax = float(delta.min()), float(delta.max())
    if dmin < 0:
        gain = min(gain, (default - safe_low) / (-dmin))
    if dmax > 0:
        gain = min(gain, (safe_high - default) / dmax)
    if gain <= 0:
        raise ValueError(f"default {default} lies outside safe range [{safe_low}, {safe_high}]")
    return float(min(1.0, gain))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--mjcf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--root-height", type=float, default=0.40)
    ap.add_argument("--length-scale", type=float, default=0.88)
    ap.add_argument("--limit-margin", type=float, default=0.98)
    args = ap.parse_args()

    src = json.loads(args.walk.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_names = list(src["data_joint_names"])
    source_index = {n: i for i, n in enumerate(source_names)}
    source_q = np.asarray(src["dof_pos"], dtype=np.float64)
    frames = source_q.shape[0]

    target_names = [m[0] for m in MAPPING]
    target_q = np.zeros((frames, len(MAPPING)), dtype=np.float64)
    joint_report = []
    for j, (target, source, sign, default) in enumerate(MAPPING):
        if source not in source_index:
            raise KeyError(f"source joint missing: {source}")
        lim = manifest["limits"][target]
        low, high = float(lim["lower"]), float(lim["upper"])
        signed = sign * source_q[:, source_index[source]]
        delta = signed - signed[0]
        gain = gain_to_fit(delta, default, low, high, args.limit_margin)
        mapped = default + gain * delta
        if mapped.min() < low - 1e-9 or mapped.max() > high + 1e-9:
            raise AssertionError(f"mapped range violation: {target}")
        target_q[:, j] = mapped
        joint_report.append({
            "target": target,
            "source": source,
            "sign": sign,
            "default": default,
            "gain": gain,
            "source_first": float(source_q[0, source_index[source]]),
            "source_min": float(source_q[:, source_index[source]].min()),
            "source_max": float(source_q[:, source_index[source]].max()),
            "mapped_min": float(mapped.min()),
            "mapped_max": float(mapped.max()),
            "limit_low": low,
            "limit_high": high,
        })

    root_trans_src = np.asarray(src["root_trans"], dtype=np.float64)
    root_trans = np.empty_like(root_trans_src)
    root_trans[:, :2] = (root_trans_src[:, :2] - root_trans_src[0, :2]) * args.length_scale
    root_trans[:, 2] = args.root_height + (root_trans_src[:, 2] - root_trans_src[0, 2]) * args.length_scale
    root_wxyz = np.asarray(src["root_wxyz"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_string(strip_meshes(args.mjcf))
    data = mujoco.MjData(model)
    joint_qadr = {}
    for name in target_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"MJCF joint missing: {name}")
        joint_qadr[name] = int(model.jnt_qposadr[jid])
    body_ids = []
    for name in TARGET_LINKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"MJCF body missing: {name}")
        body_ids.append(bid)

    free_joints = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if len(free_joints) != 1:
        raise RuntimeError(f"expected exactly one free joint, found {len(free_joints)}")
    root_qadr = int(model.jnt_qposadr[int(free_joints[0])])
    target_link_pos = np.zeros((frames, len(TARGET_LINKS), 3), dtype=np.float64)
    for i in range(frames):
        data.qpos[:] = 0.0
        data.qpos[root_qadr:root_qadr + 3] = root_trans[i]
        data.qpos[root_qadr + 3:root_qadr + 7] = root_wxyz[i]
        for j, name in enumerate(target_names):
            data.qpos[joint_qadr[name]] = target_q[i, j]
        mujoco.mj_forward(model, data)
        target_link_pos[i] = data.xpos[body_ids]

    output = {
        "fps": int(src["fps"]),
        "target_link_names": TARGET_LINKS,
        "data_joint_names": target_names,
        "root_trans": root_trans.tolist(),
        "root_wxyz": root_wxyz.tolist(),
        "target_link_pos": target_link_pos.tolist(),
        "dof_pos": target_q.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")

    velocity = np.gradient(target_q, 1.0 / float(src["fps"]), axis=0)
    report = {
        "contract": "EDU3_FULLBODY_REFERENCE_RETARGET_V1",
        "source_walk_sha256": sha256(args.walk),
        "student_manifest_sha256": sha256(args.manifest),
        "student_mjcf_sha256": sha256(args.mjcf),
        "output_sha256": sha256(args.output),
        "frames": frames,
        "fps": int(src["fps"]),
        "duration_s": frames / float(src["fps"]),
        "input_plan": "75 values/frame x 10 frames = 750",
        "action_dim": 21,
        "root_height_m": args.root_height,
        "length_scale": args.length_scale,
        "limit_margin": args.limit_margin,
        "joint_mapping": joint_report,
        "max_abs_reference_velocity_rad_s": {
            name: float(np.max(np.abs(velocity[:, i]))) for i, name in enumerate(target_names)
        },
        "target_links": TARGET_LINKS,
        "all_joint_positions_inside_full_limits": True,
        "notes": [
            "Wrist joints are removed because EDU3 has no wrist actuators.",
            "Right hip pitch, knee, ankle pitch, right shoulder pitch, and left elbow use explicit sign transforms.",
            "Reference key points are recomputed with EDU3 MJCF forward kinematics.",
            "This file is a preflight reference candidate, not authorization to train before visual and runtime gates pass.",
        ],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "report": str(args.report),
        "frames": frames,
        "duration_s": report["duration_s"],
        "min_gain": min(x["gain"] for x in joint_report),
        "scaled_joints": [x["target"] for x in joint_report if x["gain"] < 0.999999],
        "output_sha256": report["output_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
