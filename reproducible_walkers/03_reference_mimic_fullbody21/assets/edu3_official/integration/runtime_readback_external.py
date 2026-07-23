# -*- coding: utf-8 -*-
"""Read back the external EDU3 probe contract from the live PhysX articulation."""

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Edu3-Flat-External-20260721")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--output",
    default="/home/zero/external_edu3_flat_20260721/provenance/runtime_readback_probe.json",
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch

import robolab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, "/home/zero/external_edu3_flat_20260721/integration")
from external_compat_patch import install as install_external_compat

install_external_compat()
import edu3_nqj13_flat_external  # noqa: F401,E402


def values_1d(tensor, digits=7):
    data = tensor.detach().cpu()
    if data.ndim > 1:
        data = data[0]
    return [round(float(x), digits) for x in data.flatten().tolist()]


def values_2d(tensor, digits=7):
    data = tensor.detach().cpu()
    if data.ndim > 2:
        data = data[0]
    return [[round(float(x), digits) for x in row] for row in data.tolist()]


def get_view(view, method):
    fn = getattr(view, method, None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception as exc:  # keep the remaining gate evidence available
        return {"error": f"{type(exc).__name__}: {exc}"}


def serialise_view(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu()
        if data.ndim >= 2:
            data = data[0]
        return data.tolist()
    return value


expected_effort = {
    "hip_pitch": 50.0,
    "hip_roll": 20.0,
    "hip_yaw": 20.0,
    "knee": 50.0,
    "ankle": 40.0,
    "torso": 20.0,
    "arm": 10.0,
}


def expected_effort_for(name):
    if "ankle" in name:
        return expected_effort["ankle"]
    if "knee" in name:
        return expected_effort["knee"]
    if "hip_pitch" in name:
        return expected_effort["hip_pitch"]
    if "hip_roll" in name:
        return expected_effort["hip_roll"]
    if "hip_yaw" in name:
        return expected_effort["hip_yaw"]
    if "waist" in name or "torso" in name:
        return expected_effort["torso"]
    return expected_effort["arm"]


report = {"status": "FAIL", "errors": [], "warnings": []}
env = None
try:
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    unwrapped = env.unwrapped
    robot = getattr(unwrapped, "robot", None) or unwrapped.scene["robot"]
    names = list(robot.data.joint_names)
    effort = values_1d(robot.data.joint_effort_limits)
    velocity = values_1d(robot.data.joint_vel_limits)
    hard_pos = values_2d(robot.data.joint_pos_limits)
    soft_pos = values_2d(robot.data.soft_joint_pos_limits)
    default_pos = values_1d(robot.data.default_joint_pos)
    expected = [expected_effort_for(name) for name in names]

    if len(names) != 21:
        report["errors"].append(f"joint_count={len(names)}, expected=21")
    for name, actual, wanted in zip(names, effort, expected):
        if abs(actual - wanted) > 1e-5:
            report["errors"].append(f"effort {name}: runtime={actual}, expected={wanted}")

    view = robot.root_physx_view
    view_methods = [
        "get_dof_limits",
        "get_dof_max_forces",
        "get_dof_max_velocities",
        "get_dof_armatures",
        "get_dof_friction_coefficients",
        "get_dof_dynamic_friction_coefficients",
        "get_dof_viscous_friction_coefficients",
        "get_masses",
        "get_coms",
        "get_inertias",
    ]
    live = {}
    for method in view_methods:
        live[method] = serialise_view(get_view(view, method))

    old_friction = live.get("get_dof_friction_coefficients")
    if isinstance(old_friction, list):
        flat = torch.as_tensor(old_friction).flatten()
        if torch.max(torch.abs(flat)).item() > 1e-8:
            report["errors"].append(
                f"legacy PhysX friction is not zero: max_abs={torch.max(torch.abs(flat)).item()}"
            )
    else:
        report["warnings"].append("legacy friction getter unavailable; rely on explicit startup gate")

    # Short no-command stability probe: this is not policy evaluation.
    finite = True
    for _ in range(20):
        action = torch.zeros((args.num_envs, env.action_space.shape[-1]), device=unwrapped.device)
        result = env.step(action)
        obs = result[0]
        if isinstance(obs, dict):
            tensors = [v for v in obs.values() if isinstance(v, torch.Tensor)]
        else:
            tensors = [obs] if isinstance(obs, torch.Tensor) else []
        if any(not torch.isfinite(t).all().item() for t in tensors):
            finite = False
            break
    if not finite:
        report["errors"].append("non-finite observation during 20-step zero-action probe")

    masses = live.get("get_masses")
    total_mass = None
    if isinstance(masses, list):
        total_mass = float(torch.as_tensor(masses).sum().item())
        if abs(total_mass - 9.27) > 0.02:
            report["errors"].append(f"runtime total_mass={total_mass:.6f}, expected=9.27")

    actuators = {}
    for group_name, group in robot.actuators.items():
        actuators[group_name] = {
            "class": type(group).__name__,
            "joint_names": list(getattr(group, "joint_names", [])),
            "effort_limit": serialise_view(getattr(group, "effort_limit", None)),
            "velocity_limit": serialise_view(getattr(group, "velocity_limit", None)),
            "stiffness": serialise_view(getattr(group, "stiffness", None)),
            "damping": serialise_view(getattr(group, "damping", None)),
            "armature": serialise_view(getattr(group, "armature", None)),
        }

    report.update(
        {
            "task": args.task,
            "device": str(unwrapped.device),
            "num_envs": args.num_envs,
            "joint_names": names,
            "default_joint_pos": default_pos,
            "joint_pos_limits": hard_pos,
            "soft_joint_pos_limits": soft_pos,
            "joint_vel_limits": velocity,
            "joint_effort_limits": effort,
            "expected_effort_limits": expected,
            "physx_runtime": live,
            "runtime_total_mass_kg": total_mass,
            "actuators": actuators,
            "explicit_si_friction_model": "enabled by EDU3 package startup configuration",
            "zero_action_20_step_finite": finite,
        }
    )
    report["status"] = "PASS" if not report["errors"] else "FAIL"
except Exception as exc:
    report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["traceback"] = traceback.format_exc()
finally:
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    print("RUNTIME_READBACK_STATUS=" + report["status"])
    print("RUNTIME_READBACK_OUTPUT=" + args.output)
    if report["errors"]:
        print("RUNTIME_READBACK_ERRORS=" + json.dumps(report["errors"], ensure_ascii=False))
    if env is not None:
        env.close()
    app.close()

if report["status"] != "PASS":
    raise SystemExit(2)
