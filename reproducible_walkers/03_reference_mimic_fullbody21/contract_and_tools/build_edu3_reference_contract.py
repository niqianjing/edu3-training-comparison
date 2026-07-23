#!/usr/bin/env python3
"""Build the versioned EDU3 reference-walking single-source contract.

The official asset manifest remains provenance.  This file materializes the
candidate 25/10 Nm product contract that both Isaac Lab and MuJoCo must read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path("/home/zero/edu3_reference_mimic_v1")
OFFICIAL = ROOT / "assets" / "edu3_official"
PROJECT = ROOT / "project" / "mimic_real"
CONTRACT_PATH = ROOT / "contract" / "edu3_reference_contract_v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def joint_order_from_urdf(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    ]


def group_for_joint(name: str) -> str:
    if "thigh_pitch" in name or "knee" in name:
        return "module_25_high"
    if "thigh_roll" in name:
        return "module_25_high"
    return "module_10_low"


def pd_for_joint(name: str) -> tuple[float, float]:
    if "thigh_pitch" in name or "knee" in name:
        return 100.0, 3.0
    if "thigh_roll" in name or "thigh_yaw" in name or name == "torso_joint":
        return 80.0, 2.5
    if "ankle" in name:
        return 40.0, 1.5
    return 30.0, 1.2


def main() -> None:
    manifest_path = OFFICIAL / "asset_manifest.json"
    urdf_path = OFFICIAL / "urdf" / "edu3_nqj13_trainable_fullbody.urdf"
    source_mjcf_path = OFFICIAL / "mjcf" / "edu3_nqj13_trainable_fullbody.xml"
    usd_path = OFFICIAL / "usd" / "edu3_nqj13_trainable_fullbody.usd"
    motion_path = PROJECT / "data" / "edu3_walk_from_xiaohai_v1.json"
    retarget_report = ROOT / "provenance" / "retarget_report_v1.json"
    # Keep the generated MJCF next to the isolated source copy so its existing
    # relative `../meshes` path remains portable inside the asset package.
    generated_mjcf = OFFICIAL / "mjcf" / "edu3_reference_25_10_v1.xml"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = joint_order_from_urdf(urdf_path)
    limits = manifest["limits"]
    if set(order) != set(limits):
        raise RuntimeError("URDF joint set and official manifest joint set differ")

    modules = {
        "module_25_high": {
            "peak_effort_nm": 25.0,
            "continuous_effort_nm": 7.0,
            "armature_kg_m2": 0.02649,
            "coulomb_friction_nm": 0.51,
            "viscous_damping_nm_s_per_rad": 0.0432,
        },
        "module_10_low": {
            "peak_effort_nm": 10.0,
            "continuous_effort_nm": 3.75,
            "armature_kg_m2": 0.0067367,
            "coulomb_friction_nm": 0.146,
            "viscous_damping_nm_s_per_rad": 0.0306,
        },
    }

    joints = {}
    for name in order:
        source = limits[name]
        module_name = group_for_joint(name)
        module = modules[module_name]
        stiffness, damping = pd_for_joint(name)
        joints[name] = {
            "lower_rad": float(source["lower"]),
            "upper_rad": float(source["upper"]),
            "velocity_limit_rad_s": float(source["velocity"]),
            "module": module_name,
            "peak_effort_nm": module["peak_effort_nm"],
            "continuous_effort_nm": module["continuous_effort_nm"],
            "armature_kg_m2": module["armature_kg_m2"],
            "coulomb_friction_nm": module["coulomb_friction_nm"],
            "viscous_damping_nm_s_per_rad": module["viscous_damping_nm_s_per_rad"],
            "stiffness_nm_per_rad": stiffness,
            "drive_damping_nm_s_per_rad": damping,
        }

    contract = {
        "schema": "edu3.reference-walking.contract.v1",
        "version": "EDU3-REFERENCE-25-10-V1",
        "status": "preflight",
        "provenance": {
            "official_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "urdf": {"path": str(urdf_path), "sha256": sha256(urdf_path)},
            "source_mjcf": {"path": str(source_mjcf_path), "sha256": sha256(source_mjcf_path)},
            "usd": {"path": str(usd_path), "sha256": sha256(usd_path)},
            "motion": {"path": str(motion_path), "sha256": sha256(motion_path)},
            "retarget_report": {"path": str(retarget_report), "sha256": sha256(retarget_report)},
            "generated_mjcf_path": str(generated_mjcf),
        },
        "robot": {
            "joint_count": 21,
            "joint_order": order,
            "mass_kg": float(manifest["mass_kg"]),
            "root_link": "base_link",
            "feet_body_patterns": [".*_ankle_roll_link"],
            "joints": joints,
            "modules": modules,
            "continuous_rating_note": (
                "7.0/3.75 Nm are provisional thermal-screen values inherited from the "
                "innovation-line module contract; they are not a passed student-hardware rating gate"
            ),
        },
        "control": {
            "isaac_physics_dt_s": 0.005,
            "mujoco_physics_dt_s": 0.001,
            "decimation": 4,
            "policy_period_s": 0.020,
            "action_scale_rad": 0.25,
            "target_mapping": "reference_plus_scaled_action_then_full_joint_limit_clamp",
            "history_uses": "actual_effective_action_after_delay_and_target_clamp",
            "actor_history_frames": 10,
            "actor_features_per_frame": 75,
            "actor_input_dim": 750,
            "actor_output_dim": 21,
            "environment_action_delay_policy_steps": [0, 5],
            "actuator_delay_physics_steps": [0, 0],
            "physx_legacy_joint_friction_required": 0.0,
            "explicit_si_friction_required": True,
            "effort_limit_implementation": "explicit_actuator_clip; PhysX drive max force intentionally high",
        },
        "training_randomization": {
            "observation_noise": {
                "enabled": True,
                "angular_velocity": 0.2,
                "projected_gravity": 0.05,
                "joint_position": 0.02,
                "joint_velocity": 1.5,
            },
            "reset_joint_position_offset_rad": [-0.1, 0.1],
            "reset_joint_velocity_rad_s": [0.0, 0.0],
            "reset_root_z_offset_m": [0.02, 0.02],
            "contact_material": {
                "static_friction": [0.6, 1.0],
                "dynamic_friction": [0.4, 0.8],
                "restitution": [0.0, 0.005],
                "buckets": 64,
            },
            "base_mass_additive_kg": [-1.0, 1.0],
            "push": {"enabled": True, "interval_s": 1.0, "velocity_xy_m_s": [-0.5, 0.5]},
        },
        "evaluation": {
            "duration_s": 8.1,
            "automatic_reset": False,
            "randomization": False,
            "required_pulse_times_ms": [5, 20],
            "required_first_step_fields": [
                "observation", "raw_action", "raw_target", "executed_target",
                "effective_action", "last_action"
            ],
        },
    }

    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE={CONTRACT_PATH}")
    print(f"SHA256={sha256(CONTRACT_PATH)}")
    print(f"JOINTS={len(order)} INPUT={contract['control']['actor_input_dim']} OUTPUT={contract['control']['actor_output_dim']}")


if __name__ == "__main__":
    main()
