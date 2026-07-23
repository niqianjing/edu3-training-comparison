#!/usr/bin/env python3
"""Generate and verify the EDU3 candidate MJCF from the single-source contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


CONTRACT = Path("/home/zero/edu3_reference_mimic_v1/contract/edu3_reference_contract_v1.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(value: float) -> str:
    return f"{value:.9g}"


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source_info = contract["provenance"]["source_mjcf"]
    source = Path(source_info["path"])
    if sha256(source) != source_info["sha256"]:
        raise RuntimeError("Source MJCF hash mismatch")

    output = Path(contract["provenance"]["generated_mjcf_path"])
    tree = ET.parse(source)
    root = tree.getroot()
    expected = contract["robot"]["joints"]
    option = root.find("option")
    if option is None:
        raise RuntimeError("MJCF has no option element")
    option.set("timestep", fmt(contract["control"]["mujoco_physics_dt_s"]))

    seen_joints = set()
    for element in root.iter("joint"):
        name = element.attrib.get("name")
        if name not in expected:
            continue
        cfg = expected[name]
        element.set("range", f"{fmt(cfg['lower_rad'])} {fmt(cfg['upper_rad'])}")
        element.set("damping", fmt(cfg["viscous_damping_nm_s_per_rad"]))
        element.set("frictionloss", fmt(cfg["coulomb_friction_nm"]))
        element.set("armature", fmt(cfg["armature_kg_m2"]))
        seen_joints.add(name)

    seen_motors = set()
    for motor in root.iter("motor"):
        name = motor.attrib.get("joint")
        if name not in expected:
            continue
        effort = float(expected[name]["peak_effort_nm"])
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", f"-{fmt(effort)} {fmt(effort)}")
        motor.set("forcelimited", "true")
        motor.set("forcerange", f"-{fmt(effort)} {fmt(effort)}")
        seen_motors.add(name)

    if seen_joints != set(expected) or seen_motors != set(expected):
        raise RuntimeError(
            f"MJCF coverage mismatch: joints_missing={sorted(set(expected)-seen_joints)} "
            f"motors_missing={sorted(set(expected)-seen_motors)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    # Compile-time readback from the generated XML.
    check = ET.parse(output).getroot()
    for element in check.iter("joint"):
        name = element.attrib.get("name")
        if name not in expected:
            continue
        cfg = expected[name]
        actual_range = [float(v) for v in element.attrib["range"].split()]
        wanted_range = [cfg["lower_rad"], cfg["upper_rad"]]
        if max(abs(a - b) for a, b in zip(actual_range, wanted_range)) > 1e-8:
            raise RuntimeError(f"Range mismatch for {name}")
        for key, attr in (
            ("viscous_damping_nm_s_per_rad", "damping"),
            ("coulomb_friction_nm", "frictionloss"),
            ("armature_kg_m2", "armature"),
        ):
            if abs(float(element.attrib[attr]) - float(cfg[key])) > 1e-9:
                raise RuntimeError(f"{attr} mismatch for {name}")

    report = {
        "status": "PASS",
        "contract_path": str(CONTRACT),
        "contract_sha256": sha256(CONTRACT),
        "source_mjcf_sha256": sha256(source),
        "generated_mjcf_path": str(output),
        "generated_mjcf_sha256": sha256(output),
        "joint_count": len(expected),
        "motor_count": len(seen_motors),
    }
    report_path = CONTRACT.parents[1] / "provenance" / "generated_mjcf_readback_v1.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
