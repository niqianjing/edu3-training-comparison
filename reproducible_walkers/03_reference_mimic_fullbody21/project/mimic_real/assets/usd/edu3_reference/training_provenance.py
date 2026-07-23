"""Training fingerprint and compiled-runtime gate for EDU3 reference walking."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import sys

from mimic_real.assets.usd.edu3_reference.contract import CONTRACT, CONTRACT_PATH


ROOT = Path("/home/zero/edu3_reference_mimic_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_row(tensor) -> list[float]:
    value = tensor[0] if getattr(tensor, "ndim", 0) > 1 else tensor
    return [float(item) for item in value.detach().cpu().tolist()]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def code_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or "logs" in path.parts:
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def runtime_actuators(robot) -> dict:
    result = {}
    for name, actuator in robot.actuators.items():
        result[name] = {
            "joint_names": list(actuator.joint_names),
            "effort_limit_nm": tensor_row(actuator.effort_limit),
            "velocity_limit_rad_s": tensor_row(actuator.velocity_limit),
            "stiffness_nm_per_rad": tensor_row(actuator.stiffness),
            "drive_damping_nm_s_per_rad": tensor_row(actuator.damping),
            "armature_kg_m2": tensor_row(actuator.armature),
            "legacy_friction": tensor_row(actuator.friction),
            "dynamic_friction": tensor_row(actuator.dynamic_friction),
            "viscous_friction": tensor_row(actuator.viscous_friction),
            "coulomb_friction_nm": float(actuator.cfg.coulomb_friction_nm),
            "explicit_viscous_damping_nm_s_per_rad": float(actuator.cfg.viscous_damping_nm_s_per_rad),
            "delay_physics_steps": [int(actuator.cfg.min_delay), int(actuator.cfg.max_delay)],
        }
    return result


def write_training_provenance(env, log_dir: str) -> Path:
    output = Path(log_dir) / "provenance"
    output.mkdir(parents=True, exist_ok=True)
    robot = env.robot
    physx = robot.root_physx_view
    names = list(robot.joint_names)
    contract_joints = CONTRACT["robot"]["joints"]
    if set(names) != set(contract_joints):
        raise RuntimeError("Runtime joint set differs from the single-source contract")

    position_limits = robot.data.joint_pos_limits[0].detach().cpu().tolist()
    velocity_limits = tensor_row(physx.get_dof_max_velocities())
    armatures = tensor_row(physx.get_dof_armatures())
    old_friction = tensor_row(physx.get_dof_friction_coefficients())
    friction_properties = physx.get_dof_friction_properties()[0]
    static_friction = [float(value) for value in friction_properties[:, 0].detach().cpu().tolist()]
    dynamic_friction = [float(value) for value in friction_properties[:, 1].detach().cpu().tolist()]
    viscous_friction = [float(value) for value in friction_properties[:, 2].detach().cpu().tolist()]
    runtime_groups = runtime_actuators(robot)

    per_joint_actuator = {}
    for group_name, group in runtime_groups.items():
        for local_index, joint_name in enumerate(group["joint_names"]):
            per_joint_actuator[joint_name] = {
                "group": group_name,
                "effort_limit_nm": group["effort_limit_nm"][local_index],
                "velocity_limit_rad_s": group["velocity_limit_rad_s"][local_index],
                "stiffness_nm_per_rad": group["stiffness_nm_per_rad"][local_index],
                "drive_damping_nm_s_per_rad": group["drive_damping_nm_s_per_rad"][local_index],
                "armature_kg_m2": group["armature_kg_m2"][local_index],
                "coulomb_friction_nm": group["coulomb_friction_nm"],
                "viscous_damping_nm_s_per_rad": group["explicit_viscous_damping_nm_s_per_rad"],
            }

    failures = []
    tolerance = 1.0e-5
    for index, joint_name in enumerate(names):
        wanted = contract_joints[joint_name]
        actual = per_joint_actuator[joint_name]
        checks = {
            "lower_rad": position_limits[index][0],
            "upper_rad": position_limits[index][1],
            "velocity_limit_rad_s": velocity_limits[index],
            "armature_kg_m2": armatures[index],
            "peak_effort_nm": actual["effort_limit_nm"],
            "stiffness_nm_per_rad": actual["stiffness_nm_per_rad"],
            "drive_damping_nm_s_per_rad": actual["drive_damping_nm_s_per_rad"],
            "coulomb_friction_nm": actual["coulomb_friction_nm"],
            "viscous_damping_nm_s_per_rad": actual["viscous_damping_nm_s_per_rad"],
        }
        for key, runtime_value in checks.items():
            if abs(float(runtime_value) - float(wanted[key])) > tolerance:
                failures.append(f"{joint_name}.{key}: runtime={runtime_value} expected={wanted[key]}")

    if max(map(abs, old_friction + static_friction + dynamic_friction + viscous_friction), default=0.0) > 1e-12:
        failures.append("One or more legacy PhysX joint-friction fields are non-zero")

    mass_matrix = physx.get_masses().detach().cpu()
    masses = [float(value) for value in mass_matrix[0].tolist()]
    total_masses = [float(value) for value in mass_matrix.sum(dim=1).tolist()]
    nominal_mass = float(CONTRACT["robot"]["mass_kg"])
    mass_offset = CONTRACT["training_randomization"]["base_mass_additive_kg"]
    allowed_mass = [nominal_mass + float(mass_offset[0]), nominal_mass + float(mass_offset[1])]
    if min(total_masses) < allowed_mass[0] - 1.0e-4 or max(total_masses) > allowed_mass[1] + 1.0e-4:
        failures.append(
            f"randomized total mass outside contract: actual={min(total_masses)}..{max(total_masses)} "
            f"allowed={allowed_mass[0]}..{allowed_mass[1]}"
        )

    readback = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "joint_names_runtime_order": names,
        "position_limits_rad": position_limits,
        "velocity_limits_rad_s": velocity_limits,
        "physx_drive_max_force_sentinel_nm": tensor_row(physx.get_dof_max_forces()),
        "armature_kg_m2": armatures,
        "legacy_joint_friction": old_friction,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        "viscous_friction": viscous_friction,
        "body_names": list(robot.body_names),
        "body_masses_kg": masses,
        "nominal_total_mass_kg": nominal_mass,
        "allowed_randomized_total_mass_kg": allowed_mass,
        "randomized_total_mass_kg": {
            "first_environment": total_masses[0],
            "minimum": min(total_masses),
            "maximum": max(total_masses),
            "mean": sum(total_masses) / len(total_masses),
        },
        "body_coms": physx.get_coms()[0].detach().cpu().tolist(),
        "body_inertias": physx.get_inertias()[0].detach().cpu().tolist(),
        "explicit_actuators": runtime_groups,
    }
    readback_path = output / "compiled_runtime_readback.json"
    readback_path.write_text(json.dumps(readback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("Compiled runtime gate failed: " + "; ".join(failures[:8]))

    evidence_paths = [
        ROOT / "provenance" / "retarget_report_v1.json",
        ROOT / "provenance" / "target_mapping_probe.json",
        ROOT / "provenance" / "generated_mjcf_readback_v1.json",
        ROOT / "provenance" / "pulse_isaac_v1.json",
        ROOT / "provenance" / "pulse_mujoco_v1.json",
        ROOT / "provenance" / "pulse_compare_v1.json",
    ]
    copied = {}
    shutil.copy2(CONTRACT_PATH, output / CONTRACT_PATH.name)
    generated_mjcf = Path(CONTRACT["provenance"]["generated_mjcf_path"])
    shutil.copy2(generated_mjcf, output / generated_mjcf.name)
    for path in evidence_paths:
        if not path.exists():
            raise RuntimeError(f"Required preflight evidence missing: {path}")
        shutil.copy2(path, output / path.name)
        copied[path.name] = sha256(path)

    fingerprint = {
        "status": "PASS",
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "contract_version": CONTRACT["version"],
        "contract_sha256": sha256(CONTRACT_PATH),
        "generated_mjcf_sha256": sha256(generated_mjcf),
        "compiled_runtime_readback_sha256": sha256(readback_path),
        "evidence_sha256": copied,
        "project_python_tree_sha256": code_tree_sha256(ROOT / "project" / "mimic_real"),
        "versions": {
            "python": sys.version,
            "torch": package_version("torch"),
            "isaaclab": package_version("isaaclab"),
            "isaacsim": package_version("isaacsim"),
            "mujoco": package_version("mujoco"),
        },
    }
    fingerprint_path = output / "training_fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "EDU3_COMPILED_RUNTIME_GATE=PASS "
        f"contract_sha256={fingerprint['contract_sha256']} "
        f"runtime_sha256={fingerprint['compiled_runtime_readback_sha256']} "
        f"code_sha256={fingerprint['project_python_tree_sha256']}",
        flush=True,
    )
    return fingerprint_path


__all__ = ["write_training_provenance"]
