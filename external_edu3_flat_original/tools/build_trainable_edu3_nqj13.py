#!/usr/bin/env python3
"""Build a trainable, portable EDU3 nqj13 asset set from the SW URDF.

The SolidWorks export is the only source for link geometry, inertials, joint
tree and meshes.  The validated reference is used only for collision proxies;
all semantic names, limits and actuator contracts are explicit below.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


LINK_MAP = {
    "base_link": "base_link",
    "link_3": "left_thigh_pitch_link", "link_2": "left_thigh_roll_link",
    "link_1": "left_thigh_yaw_link", "link_4": "left_knee_link",
    "link_5": "left_ankle_pitch_link", "link_6": "left_ankle_roll_link",
    "link_9": "right_thigh_pitch_link", "link_8": "right_thigh_roll_link",
    "link_7": "right_thigh_yaw_link", "link_10": "right_knee_link",
    "link_11": "right_ankle_pitch_link", "link_12": "right_ankle_roll_link",
    "link_13": "torso_link",
    "link_14": "left_arm_pitch_link", "link_15": "left_arm_roll_link",
    "link_16": "left_arm_yaw_link", "link_17": "left_elbow_pitch_link",
    "link_19": "right_arm_pitch_link", "link_20": "right_arm_roll_link",
    "link_21": "right_arm_yaw_link", "link_22": "right_elbow_pitch_link",
}

JOINT_MAP = {
    "joint_3": "left_thigh_pitch_joint", "joint_2": "left_thigh_roll_joint",
    "joint_1": "left_thigh_yaw_joint", "joint_4": "left_knee_joint",
    "joint_5": "left_ankle_pitch_joint", "joint_6": "left_ankle_roll_joint",
    "joint_9": "right_thigh_pitch_joint", "joint_8": "right_thigh_roll_joint",
    "joint_7": "right_thigh_yaw_joint", "joint_10": "right_knee_joint",
    "joint_11": "right_ankle_pitch_joint", "joint_12": "right_ankle_roll_joint",
    "joint_13": "torso_joint",
    "joint_14": "left_arm_pitch_joint", "joint_15": "left_arm_roll_joint",
    "joint_16": "left_arm_yaw_joint", "joint_17": "left_elbow_pitch_joint",
    "joint_19": "right_arm_pitch_joint", "joint_20": "right_arm_roll_joint",
    "joint_21": "right_arm_yaw_joint", "joint_22": "right_elbow_pitch_joint",
}

# lower, upper, effort Nm, velocity rad/s
LIMITS = {
    "left_thigh_pitch_joint": (-1.5708, 0.7854, 50.0, 24.0),
    "left_thigh_roll_joint": (-0.3491, 0.7854, 20.0, 24.0),
    "left_thigh_yaw_joint": (-0.7854, 0.7854, 20.0, 24.0),
    "left_knee_joint": (0.0, 1.2217, 50.0, 24.0),
    "left_ankle_pitch_joint": (-0.4363, 0.4363, 40.0, 12.0),
    "left_ankle_roll_joint": (-0.4363, 0.4363, 40.0, 12.0),
    "right_thigh_pitch_joint": (-0.7854, 1.5708, 50.0, 24.0),
    "right_thigh_roll_joint": (-0.7854, 0.3491, 20.0, 24.0),
    "right_thigh_yaw_joint": (-0.7854, 0.7854, 20.0, 24.0),
    "right_knee_joint": (-1.2217, 0.0, 50.0, 24.0),
    "right_ankle_pitch_joint": (-0.4363, 0.4363, 40.0, 12.0),
    "right_ankle_roll_joint": (-0.4363, 0.4363, 40.0, 12.0),
    "torso_joint": (-0.7854, 0.7854, 20.0, 24.0),
    "left_arm_pitch_joint": (-2.3562, 0.7854, 10.0, 24.0),
    "left_arm_roll_joint": (0.0, 2.3562, 10.0, 24.0),
    "left_arm_yaw_joint": (-0.7854, 0.7854, 10.0, 24.0),
    "left_elbow_pitch_joint": (0.0, 1.5708, 10.0, 24.0),
    "right_arm_pitch_joint": (-0.7854, 2.3562, 10.0, 24.0),
    "right_arm_roll_joint": (-2.3562, 0.0, 10.0, 24.0),
    "right_arm_yaw_joint": (-0.7854, 0.7854, 10.0, 24.0),
    "right_elbow_pitch_joint": (-1.5708, 0.0, 10.0, 24.0),
}

# SW exports the two ankle axes as a serial offset.  The real differential
# ankle is represented as a coincident universal joint, as already validated.
ANKLE_PATCH = {
    "joint_5": ("-0.00830791449 -0.0199937364 -0.175852828", "0 1 0"),
    "joint_6": ("0 0 0", "1 0 0"),
    "joint_11": ("-0.00830791507 0.0199937364 -0.175852613", "0 -1 0"),
    "joint_12": ("0 0 0", "1 0 0"),
}
ANKLE_LINK_OFFSETS = {
    "link_5": (0.0116588173, -9.22009349e-08, 0.0964972153),
    "link_6": (0.0115125745, -9.22009349e-08, 0.0541032232),
    "link_11": (0.0116588178, 9.22009349e-08, 0.0964971632),
    "link_12": (0.0115125751, 9.22009349e-08, 0.0541031710),
}
AXIS_OVERRIDES = {
    "joint_15": "1 0 0",
    "joint_16": "0 0 1",
    "joint_17": "0 -1 0",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(x: float) -> str:
    return f"{x:.10g}"


def armature(name: str) -> float:
    if "thigh_pitch" in name or "knee" in name:
        return 0.02649
    return 0.0067367


def measured_dynamics(name: str) -> tuple[float, float]:
    """Return measured output-side (Coulomb Nm, viscous Nm*s/rad)."""
    if "thigh_pitch" in name or "knee" in name:
        return 0.51, 0.0432
    return 0.146, 0.0306


def build_urdf(raw_path: Path, reference_path: Path, out_path: Path) -> ET.Element:
    raw = ET.parse(raw_path).getroot()
    ref = ET.parse(reference_path).getroot()
    assert len(raw.findall("link")) == 22 and len(raw.findall("joint")) == 21
    assert abs(sum(float(x.attrib["value"]) for x in raw.findall(".//mass")) - 9.2699999944) < 1e-9
    ref_links = {x.attrib["name"]: x for x in ref.findall("link")}

    raw.attrib["name"] = "edu3_nqj13_trainable_fullbody"
    for link in raw.findall("link"):
        old = link.attrib["name"]
        link.attrib["name"] = LINK_MAP[old]
        if old in ANKLE_LINK_OFFSETS:
            offset = ANKLE_LINK_OFFSETS[old]
            origin_nodes = [link.find("inertial/origin")] + link.findall("visual/origin")
            for origin in origin_nodes:
                old_xyz = [float(x) for x in origin.attrib.get("xyz", "0 0 0").split()]
                origin.attrib["xyz"] = " ".join(fmt(a + b) for a, b in zip(old_xyz, offset))
        for mesh in link.findall(".//mesh"):
            mesh.attrib["filename"] = "../meshes/" + Path(mesh.attrib["filename"]).name
        # Collision proxies are a training representation, not CAD truth.
        # Use the validated box/four-sphere proxies; inertial and visual stay raw.
        for node in list(link.findall("collision")):
            link.remove(node)
        for collision in ref_links[LINK_MAP[old]].findall("collision"):
            link.append(copy.deepcopy(collision))

    for joint in raw.findall("joint"):
        old = joint.attrib["name"]
        name = JOINT_MAP[old]
        joint.attrib["name"] = name
        joint.find("parent").attrib["link"] = LINK_MAP[joint.find("parent").attrib["link"]]
        joint.find("child").attrib["link"] = LINK_MAP[joint.find("child").attrib["link"]]
        if old in ANKLE_PATCH:
            joint.find("origin").attrib.update({"xyz": ANKLE_PATCH[old][0], "rpy": "0 0 0"})
            joint.find("axis").attrib["xyz"] = ANKLE_PATCH[old][1]
        elif old in AXIS_OVERRIDES:
            joint.find("axis").attrib["xyz"] = AXIS_OVERRIDES[old]
        lo, hi, effort, velocity = LIMITS[name]
        limit = joint.find("limit")
        limit.attrib.clear()
        limit.attrib.update(lower=fmt(lo), upper=fmt(hi), effort=fmt(effort), velocity=fmt(velocity))
        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.attrib.clear()
        friction, damping = measured_dynamics(name)
        dynamics.attrib.update(friction=fmt(friction), damping=fmt(damping))

    ET.indent(raw, space="  ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(raw).write(out_path, encoding="utf-8", xml_declaration=True)
    return raw


def make_mjcf(urdf: ET.Element, out_path: Path) -> None:
    model = ET.Element("mujoco", model=urdf.attrib["name"])
    ET.SubElement(model, "compiler", angle="radian", meshdir="../meshes", autolimits="true")
    ET.SubElement(model, "option", timestep="0.005", gravity="0 0 -9.81", integrator="implicitfast")
    default = ET.SubElement(model, "default")
    ET.SubElement(default, "joint", damping="0", frictionloss="0")
    ET.SubElement(default, "geom", condim="3", friction="1 0.005 0.0001")
    asset = ET.SubElement(model, "asset")
    mesh_names = {}
    for mesh in urdf.findall(".//visual//mesh"):
        fn = Path(mesh.attrib["filename"]).name
        if fn not in mesh_names:
            mesh_names[fn] = Path(fn).stem
            ET.SubElement(asset, "mesh", name=mesh_names[fn], file=fn)

    links = {x.attrib["name"]: x for x in urdf.findall("link")}
    children = {}
    for joint in urdf.findall("joint"):
        children.setdefault(joint.find("parent").attrib["link"], []).append(joint)
    world = ET.SubElement(model, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.1", rgba="0.25 0.25 0.25 1")

    def add_link(parent_xml: ET.Element, link_name: str, via_joint: ET.Element | None = None) -> None:
        attrs = {"name": link_name}
        if via_joint is not None:
            origin = via_joint.find("origin")
            attrs["pos"] = origin.attrib.get("xyz", "0 0 0")
            attrs["euler"] = origin.attrib.get("rpy", "0 0 0")
        body = ET.SubElement(parent_xml, "body", **attrs)
        if via_joint is None:
            ET.SubElement(body, "freejoint", name="root")
        else:
            name = via_joint.attrib["name"]
            lo, hi, _, _ = LIMITS[name]
            friction, damping = measured_dynamics(name)
            ET.SubElement(body, "joint", name=name, type="hinge",
                          axis=via_joint.find("axis").attrib["xyz"], range=f"{fmt(lo)} {fmt(hi)}",
                          limited="true", damping=fmt(damping), frictionloss=fmt(friction), armature=fmt(armature(name)))
        link = links[link_name]
        inertial = link.find("inertial")
        inertia = inertial.find("inertia").attrib
        ET.SubElement(body, "inertial", pos=inertial.find("origin").attrib.get("xyz", "0 0 0"),
                      mass=inertial.find("mass").attrib["value"],
                      fullinertia=" ".join(inertia[k] for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        for i, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            color = visual.find("material/color")
            ET.SubElement(body, "geom", name=f"{link_name}_visual_{i}", type="mesh",
                          mesh=mesh_names[Path(mesh.attrib["filename"]).name],
                          pos=visual.find("origin").attrib.get("xyz", "0 0 0"),
                          euler=visual.find("origin").attrib.get("rpy", "0 0 0"),
                          rgba="0.7 0.7 0.7 1" if color is None else color.attrib.get("rgba", "0.7 0.7 0.7 1"),
                          contype="0", conaffinity="0", group="2", density="0")
        for i, collision in enumerate(link.findall("collision")):
            origin = collision.find("origin")
            common = dict(name=f"{link_name}_collision_{i}",
                          pos=origin.attrib.get("xyz", "0 0 0"), euler=origin.attrib.get("rpy", "0 0 0"),
                          rgba="0.4 0.6 0.8 0.35", density="0")
            box = collision.find("geometry/box")
            sphere = collision.find("geometry/sphere")
            if box is not None:
                half = [float(v) / 2 for v in box.attrib["size"].split()]
                ET.SubElement(body, "geom", type="box", size=" ".join(fmt(v) for v in half), **common)
            elif sphere is not None:
                ET.SubElement(body, "geom", type="sphere", size=sphere.attrib["radius"], **common)
            else:
                raise RuntimeError(f"Unsupported collision on {link_name}")
        for child_joint in children.get(link_name, []):
            add_link(body, child_joint.find("child").attrib["link"], child_joint)

    add_link(world, "base_link")
    actuators = ET.SubElement(model, "actuator")
    for joint in urdf.findall("joint"):
        name = joint.attrib["name"]
        effort = LIMITS[name][2]
        ET.SubElement(actuators, "motor", name=name + "_motor", joint=name,
                      gear="1", ctrllimited="true", ctrlrange=f"-{fmt(effort)} {fmt(effort)}",
                      forcelimited="true", forcerange=f"-{fmt(effort)} {fmt(effort)}")
    ET.indent(model, space="  ")
    ET.ElementTree(model).write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--collision-reference", type=Path, required=True)
    ap.add_argument("--meshes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    (out / "urdf").mkdir(parents=True, exist_ok=True)
    (out / "mjcf").mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.meshes, out / "meshes", dirs_exist_ok=True)
    urdf_path = out / "urdf" / "edu3_nqj13_trainable_fullbody.urdf"
    urdf = build_urdf(args.raw, args.collision_reference, urdf_path)
    mjcf_path = out / "mjcf" / "edu3_nqj13_trainable_fullbody.xml"
    make_mjcf(urdf, mjcf_path)
    manifest = {
        "asset": "edu3_nqj13_trainable_fullbody",
        "source_raw_urdf": str(args.raw),
        "source_raw_sha256": sha256(args.raw),
        "urdf_sha256": sha256(urdf_path),
        "mjcf_sha256": sha256(mjcf_path),
        "links": 22, "joints": 21, "mass_kg": 9.2699999944,
        "measured_dynamics_source": "Zeroth engineer regression, 2026-05-20/21; output-side units",
        "limits": {k: {"lower": v[0], "upper": v[1], "effort": v[2], "velocity": v[3],
                        "armature": armature(k),
                        "coulomb_friction_nm": measured_dynamics(k)[0],
                        "viscous_damping_nm_s_per_rad": measured_dynamics(k)[1]}
                   for k, v in LIMITS.items()},
        "link_map": LINK_MAP, "joint_map": JOINT_MAP,
    }
    (out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"urdf": str(urdf_path), "mjcf": str(mjcf_path), "manifest": str(out / 'asset_manifest.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
