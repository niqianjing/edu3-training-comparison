# -*- coding: utf-8 -*-
"""Validate the external EDU3 live-readback evidence at the actuator boundary.

The package uses an explicit delayed-PD actuator.  PhysX joint max-force is
left effectively unlimited, while the actuator clips motor effort to the
50/40/20/10 Nm contract.  This validator makes that boundary explicit.
"""

import argparse
import json
import math
import os


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def expected_effort(name):
    if "ankle" in name:
        return 40.0
    if "knee" in name or "thigh_pitch" in name:
        return 50.0
    if "thigh_roll" in name or "thigh_yaw" in name or "torso" in name:
        return 20.0
    return 10.0


def expected_velocity(name):
    return 12.0 if "ankle" in name else 24.0


def expected_armature(name):
    return 0.02649 if ("thigh_pitch" in name or "knee" in name) else 0.0067367


with open(args.input, encoding="utf-8") as handle:
    raw = json.load(handle)

errors = []
names = raw.get("joint_names", [])
if len(names) != 21 or len(set(names)) != 21:
    errors.append(f"joint order invalid: count={len(names)}, unique={len(set(names))}")

actual_effort = {}
actual_armature = {}
for group_name, group in raw.get("actuators", {}).items():
    group_names = group.get("joint_names", [])
    efforts = group.get("effort_limit", [])
    armatures = group.get("armature", [])
    if not (len(group_names) == len(efforts) == len(armatures)):
        errors.append(f"actuator group length mismatch: {group_name}")
        continue
    for name, effort, armature in zip(group_names, efforts, armatures):
        if name in actual_effort:
            errors.append(f"duplicate actuator assignment: {name}")
        actual_effort[name] = float(effort)
        actual_armature[name] = float(armature)

for index, name in enumerate(names):
    effort = actual_effort.get(name)
    if effort is None or abs(effort - expected_effort(name)) > 1.0e-5:
        errors.append(
            f"actuator effort {name}: runtime={effort}, expected={expected_effort(name)}"
        )
    velocity = float(raw.get("joint_vel_limits", [math.nan] * len(names))[index])
    if abs(velocity - expected_velocity(name)) > 1.0e-5:
        errors.append(
            f"velocity {name}: runtime={velocity}, expected={expected_velocity(name)}"
        )
    armature = actual_armature.get(name)
    if armature is None or abs(armature - expected_armature(name)) > 1.0e-5:
        errors.append(
            f"armature {name}: runtime={armature}, expected={expected_armature(name)}"
        )

physx = raw.get("physx_runtime", {})
legacy_friction = physx.get("get_dof_friction_coefficients")
if not isinstance(legacy_friction, list) or len(legacy_friction) != 21:
    errors.append("legacy PhysX friction readback missing")
elif max(abs(float(value)) for value in legacy_friction) > 1.0e-8:
    errors.append("legacy PhysX friction is not all zero")

max_forces = physx.get("get_dof_max_forces")
physx_unlimited = (
    isinstance(max_forces, list)
    and len(max_forces) == 21
    and min(float(value) for value in max_forces) >= 1.0e8
)
if not physx_unlimited:
    errors.append("unexpected PhysX constraint max-force contract")

mass = raw.get("runtime_total_mass_kg")
if mass is None or abs(float(mass) - 9.27) > 0.02:
    errors.append(f"runtime total mass invalid: {mass}")
if not raw.get("zero_action_20_step_finite"):
    errors.append("20-step zero-action finite probe failed")

final = {
    "status": "PASS" if not errors else "FAIL",
    "errors": errors,
    "task": raw.get("task"),
    "device": raw.get("device"),
    "num_envs": raw.get("num_envs"),
    "joint_names": names,
    "joint_position_limits_runtime": raw.get("joint_pos_limits"),
    "joint_velocity_limits_runtime": raw.get("joint_vel_limits"),
    "motor_effort_contract_boundary": "MeasuredFrictionDelayedPDActuator.effort_limit",
    "actuator_effort_limits_by_joint": actual_effort,
    "actuator_armatures_by_joint": actual_armature,
    "physx_constraint_max_force_unlimited": physx_unlimited,
    "physx_constraint_max_force_runtime": max_forces,
    "legacy_physx_friction_all_zero": not errors
    and max(abs(float(value)) for value in legacy_friction) <= 1.0e-8,
    "explicit_si_friction_model": raw.get("explicit_si_friction_model"),
    "runtime_total_mass_kg": mass,
    "zero_action_20_step_finite": raw.get("zero_action_20_step_finite"),
    "raw_probe_file": os.path.abspath(args.input),
    "raw_probe_parser_errors_superseded": raw.get("errors", []),
    "note": (
        "The raw probe compared motor limits to PhysX constraint max-force and therefore "
        "reported a false failure. This final gate validates the actual actuator clipping "
        "boundary and separately records the intentionally unlimited PhysX constraint field."
    ),
}

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w", encoding="utf-8") as handle:
    json.dump(final, handle, ensure_ascii=False, indent=2)
    handle.flush()
    os.fsync(handle.fileno())

print("FINAL_RUNTIME_GATE=" + final["status"])
print("FINAL_RUNTIME_GATE_OUTPUT=" + os.path.abspath(args.output))
if errors:
    print("FINAL_RUNTIME_GATE_ERRORS=" + json.dumps(errors, ensure_ascii=False))
raise SystemExit(0 if not errors else 2)
