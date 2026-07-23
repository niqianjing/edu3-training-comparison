#!/usr/bin/env python3
"""Strict provenance and cross-format checks for the EDU3 nqj13 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import build_trainable_edu3_nqj13 as build_contract
from build_trainable_edu3_nqj13 import ANKLE_LINK_OFFSETS, ANKLE_PATCH, AXIS_OVERRIDES, JOINT_MAP, LIMITS, LINK_MAP, armature


def vec(text: str) -> list[float]:
    return [float(x) for x in text.split()]


def close_vec(a: str, b: str, tol: float = 2e-5) -> bool:
    va, vb = vec(a), vec(b)
    return len(va) == len(vb) and max(abs(x-y) for x, y in zip(va, vb)) <= tol

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def collision_signature(link: ET.Element) -> list[tuple]:
    out = []
    for c in link.findall("collision"):
        origin = c.find("origin")
        box, sphere = c.find("geometry/box"), c.find("geometry/sphere")
        if box is not None:
            shape = ("box", tuple(vec(box.attrib["size"])))
        elif sphere is not None:
            shape = ("sphere", float(sphere.attrib["radius"]))
        else:
            shape = ("unsupported",)
        out.append((tuple(vec(origin.attrib.get("xyz", "0 0 0"))),
                    tuple(vec(origin.attrib.get("rpy", "0 0 0"))), shape))
    return out


def inertial_signature(link: ET.Element) -> list[float]:
    node = link.find("inertial")
    inertia = node.find("inertia").attrib
    return (vec(node.find("origin").attrib.get("xyz", "0 0 0"))
            + vec(node.find("origin").attrib.get("rpy", "0 0 0"))
            + [float(node.find("mass").attrib["value"])]
            + [float(inertia[k]) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    raw = ET.parse(args.raw).getroot()
    ref = ET.parse(args.reference).getroot()
    gen = ET.parse(args.package / "urdf" / "edu3_nqj13_trainable_fullbody.urdf").getroot()
    mj = ET.parse(args.package / "mjcf" / "edu3_nqj13_trainable_fullbody.xml").getroot()
    errors, notes = [], []

    raw_links = {x.attrib["name"]: x for x in raw.findall("link")}
    gen_links = {x.attrib["name"]: x for x in gen.findall("link")}
    ref_links = {x.attrib["name"]: x for x in ref.findall("link")}
    raw_joints = {x.attrib["name"]: x for x in raw.findall("joint")}
    gen_joints = {x.attrib["name"]: x for x in gen.findall("joint")}
    ref_joints = {x.attrib["name"]: x for x in ref.findall("joint")}

    if set(gen_links) != set(LINK_MAP.values()): errors.append("semantic link set mismatch")
    if set(gen_joints) != set(JOINT_MAP.values()): errors.append("semantic joint set mismatch")
    raw_mass = sum(float(x.attrib["value"]) for x in raw.findall(".//mass"))
    gen_mass = sum(float(x.attrib["value"]) for x in gen.findall(".//mass"))
    if abs(raw_mass-gen_mass) > 1e-12: errors.append(f"mass changed {raw_mass}->{gen_mass}")

    for old, new in LINK_MAP.items():
        ri, gi = inertial_signature(raw_links[old]), inertial_signature(gen_links[new])
        expected = list(ri)
        if old in ANKLE_LINK_OFFSETS:
            for i, delta in enumerate(ANKLE_LINK_OFFSETS[old]):
                expected[i] += delta
        if any(abs(a-b) > 2e-9 for a, b in zip(expected, gi)):
            errors.append(f"raw-derived inertial mismatch: {old}->{new}")
        raw_visuals = raw_links[old].findall("visual")
        gen_visuals = gen_links[new].findall("visual")
        if len(raw_visuals) != len(gen_visuals):
            errors.append(f"visual count mismatch: {old}->{new}")
        for rv, gv in zip(raw_visuals, gen_visuals):
            raw_xyz = vec(rv.find("origin").attrib.get("xyz", "0 0 0"))
            if old in ANKLE_LINK_OFFSETS:
                for i, delta in enumerate(ANKLE_LINK_OFFSETS[old]):
                    raw_xyz[i] += delta
            if not close_vec(gv.find("origin").attrib.get("xyz", "0 0 0"), " ".join(map(str, raw_xyz)), 2e-9):
                errors.append(f"raw-derived visual xyz mismatch: {old}->{new}")
            if not close_vec(gv.find("origin").attrib.get("rpy", "0 0 0"), rv.find("origin").attrib.get("rpy", "0 0 0"), 1e-12):
                errors.append(f"raw visual rpy mismatch: {old}->{new}")
            rm, gm = rv.find("geometry/mesh"), gv.find("geometry/mesh")
            if rm is None or gm is None:
                errors.append(f"visual mesh missing: {old}->{new}")
            elif Path(rm.attrib["filename"]).name != Path(gm.attrib["filename"]).name:
                errors.append(f"visual mesh identity mismatch: {old}->{new}")
        refi = inertial_signature(ref_links[new])
        if any(abs(a-b) > 2e-9 for a, b in zip(gi, refi)):
            errors.append(f"reference inertial mismatch: {new}")
        if collision_signature(gen_links[new]) != collision_signature(ref_links[new]):
            errors.append(f"collision proxy mismatch: {new}")

    for old, new in JOINT_MAP.items():
        rj, gj, vj = raw_joints[old], gen_joints[new], ref_joints[new]
        if gj.find("parent").attrib["link"] != LINK_MAP[rj.find("parent").attrib["link"]]:
            errors.append(f"parent mismatch: {new}")
        if gj.find("child").attrib["link"] != LINK_MAP[rj.find("child").attrib["link"]]:
            errors.append(f"child mismatch: {new}")
        expected_origin = ANKLE_PATCH[old][0] if old in ANKLE_PATCH else rj.find("origin").attrib["xyz"]
        expected_axis = ANKLE_PATCH[old][1] if old in ANKLE_PATCH else AXIS_OVERRIDES.get(old, rj.find("axis").attrib["xyz"])
        if not close_vec(gj.find("origin").attrib["xyz"], expected_origin, 1e-10): errors.append(f"origin mismatch: {new}")
        if not close_vec(gj.find("axis").attrib["xyz"], expected_axis, 1e-10): errors.append(f"axis mismatch: {new}")
        if not close_vec(gj.find("origin").attrib["xyz"], vj.find("origin").attrib["xyz"]): errors.append(f"reference origin mismatch: {new}")
        if not close_vec(gj.find("axis").attrib["xyz"], vj.find("axis").attrib["xyz"]): errors.append(f"reference axis mismatch: {new}")
        lo, hi, effort, velocity = LIMITS[new]
        lim = gj.find("limit").attrib
        actual = tuple(float(lim[k]) for k in ("lower", "upper", "effort", "velocity"))
        if any(abs(a-b) > 1e-8 for a, b in zip(actual, (lo, hi, effort, velocity))): errors.append(f"limit contract mismatch: {new}")
        dyn = gj.find("dynamics").attrib
        friction, damping = build_contract.measured_dynamics(new)
        if abs(float(dyn["friction"])-friction) > 1e-9 or abs(float(dyn["damping"])-damping) > 1e-9: errors.append(f"measured dynamics mismatch: {new}")
        ref_lim = vj.find("limit").attrib
        if abs(float(ref_lim["lower"])-lo) > 1e-8 or abs(float(ref_lim["upper"])-hi) > 1e-8:
            errors.append(f"reference ROM mismatch: {new}")

    mj_joints = {x.attrib["name"]: x for x in mj.findall(".//joint") if "name" in x.attrib}
    motors = {x.attrib["joint"]: x for x in list(mj.find("actuator"))}
    if set(mj_joints) != set(gen_joints): errors.append("MJCF joint name set mismatch")
    if set(motors) != set(gen_joints): errors.append("MJCF actuator set mismatch")
    for name, joint in mj_joints.items():
        lo, hi, effort, _ = LIMITS[name]
        if not close_vec(joint.attrib["range"], f"{lo} {hi}", 1e-8): errors.append(f"MJCF range mismatch: {name}")
        friction, damping = build_contract.measured_dynamics(name)
        if abs(float(joint.attrib["frictionloss"])-friction) > 1e-9 or abs(float(joint.attrib["damping"])-damping) > 1e-9: errors.append(f"MJCF measured dynamics mismatch: {name}")
        if abs(float(joint.attrib["armature"])-armature(name)) > 1e-9: errors.append(f"MJCF armature mismatch: {name}")
        cr = vec(motors[name].attrib["ctrlrange"])
        if max(abs(cr[0]+effort), abs(cr[1]-effort)) > 1e-8: errors.append(f"MJCF effort mismatch: {name}")

    source_mesh_dir = args.raw.parent.parent / "meshes"
    expected_meshes = {Path(x.attrib["filename"]).name for x in raw.findall(".//visual//mesh")}
    packaged_meshes = {p.name for p in (args.package / "meshes").glob("*.STL")}
    if expected_meshes != packaged_meshes:
        errors.append(f"mesh file set mismatch: expected={sorted(expected_meshes)} packaged={sorted(packaged_meshes)}")
    for name in expected_meshes:
        if not (source_mesh_dir / name).exists() or not (args.package / "meshes" / name).exists():
            errors.append(f"mesh missing: {name}")
        elif sha256(source_mesh_dir / name) != sha256(args.package / "meshes" / name):
            errors.append(f"mesh byte hash mismatch: {name}")

    notes.extend([
        f"Raw source: {args.raw}",
        f"Raw/Generated mass: {raw_mass:.10f}/{gen_mass:.10f} kg",
        "Reference is used only for semantic validation and collision-proxy comparison.",
        "Measured dynamics: 25Nm group 0.51Nm/0.0432Nm*s/rad; 10Nm group 0.146Nm/0.0306Nm*s/rad.",
        "Intentional CAD-to-training transform: coincident ankle pitch/roll universal-joint abstraction.",
    ])
    status = "PASS" if not errors else "FAIL"
    report = [f"# EDU3 nqj13 trainable asset verification: {status}", "", "## Evidence", ""]
    report += [f"- {x}" for x in notes]
    report += ["", "## Errors", ""] + ([f"- {x}" for x in errors] if errors else ["- None"])
    args.report.write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps({"status": status, "errors": errors, "report": str(args.report)}, ensure_ascii=False))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
