"""Independent MuJoCo closed-loop examination for EDU3 reference walking.

The policy is frozen.  This script reproduces the 750-input history, the
reference-plus-action target mapping, joint-limit clamp, explicit motor PD,
motor torque clipping, and measured SI friction used by the Isaac training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--runtime-readback", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--duration", type=float, default=8.1)
    ap.add_argument("--diagnostic-velocity-clamp", action="store_true")
    ap.add_argument("--diagnostic-no-contact", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.contract, encoding="utf-8") as handle:
        contract = json.load(handle)
    with open(args.runtime_readback, encoding="utf-8") as handle:
        runtime = json.load(handle)
    with open(args.motion, encoding="utf-8") as handle:
        motion = json.load(handle)

    expected_motion_sha = contract["provenance"]["motion"]["sha256"]
    if sha256(args.motion) != expected_motion_sha:
        raise RuntimeError("Motion SHA256 does not match the frozen contract")
    policy_order = list(runtime["joint_names_runtime_order"])
    contract_order = list(contract["robot"]["joint_order"])
    if set(policy_order) != set(contract_order) or len(policy_order) != 21:
        raise RuntimeError("Runtime/contract joint order mismatch")

    control = contract["control"]
    physics_dt = float(control["mujoco_physics_dt_s"])
    control_dt = float(control["policy_period_s"])
    control_stride = int(round(control_dt / physics_dt))
    if control_stride * physics_dt != control_dt:
        raise RuntimeError("Control period is not an integer MuJoCo step count")
    action_scale = float(control["action_scale_rad"])
    history_frames = int(control["actor_history_frames"])
    if history_frames != 10 or int(control["actor_input_dim"]) != 750:
        raise RuntimeError("Actor history contract mismatch")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = physics_dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)

    actuator_joint_ids = model.actuator_trnid[:, 0].astype(int)
    mj_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(jid)) for jid in actuator_joint_ids]
    if set(mj_names) != set(contract_order):
        raise RuntimeError(f"MJCF joint-name mismatch: {mj_names}")
    qadr = model.jnt_qposadr[actuator_joint_ids]
    dadr = model.jnt_dofadr[actuator_joint_ids]
    policy_to_mj = np.asarray([policy_order.index(name) for name in mj_names])
    mj_for_policy = np.asarray([mj_names.index(name) for name in policy_order])

    joint_cfg_mj = [contract["robot"]["joints"][name] for name in mj_names]
    lower = np.asarray([j["lower_rad"] for j in joint_cfg_mj], dtype=np.float64)
    upper = np.asarray([j["upper_rad"] for j in joint_cfg_mj], dtype=np.float64)
    peak = np.asarray([j["peak_effort_nm"] for j in joint_cfg_mj], dtype=np.float64)
    velocity_limit = np.asarray([j["velocity_limit_rad_s"] for j in joint_cfg_mj], dtype=np.float64)
    continuous = np.asarray([j["continuous_effort_nm"] for j in joint_cfg_mj], dtype=np.float64)
    kp = np.asarray([j["stiffness_nm_per_rad"] for j in joint_cfg_mj], dtype=np.float64)
    kd = np.asarray([j["drive_damping_nm_s_per_rad"] for j in joint_cfg_mj], dtype=np.float64)
    coulomb = np.asarray([j["coulomb_friction_nm"] for j in joint_cfg_mj], dtype=np.float64)
    viscous = np.asarray([j["viscous_damping_nm_s_per_rad"] for j in joint_cfg_mj], dtype=np.float64)

    # The explicit actuator owns SI friction.  Remove the MJCF passive copy so
    # it cannot become a second hidden brake.  Armature remains unchanged.
    model.dof_frictionloss[dadr] = 0.0
    model.dof_damping[dadr] = 0.0
    data.ctrl[:] = 0.0

    # Match MotionLoader.sync_fps() used during training. Reading the same
    # JSON is not enough: training converts the source 30 Hz motion to the
    # 50 Hz policy clock before it reaches the observation/reward code.
    root_trans_raw = np.asarray(motion["root_trans"], dtype=np.float64)
    root_wxyz_raw = np.asarray(motion["root_wxyz"], dtype=np.float64)
    motion_dof_raw = np.asarray(motion["dof_pos"], dtype=np.float64)
    old_fps = float(motion["fps"])
    new_fps = 1.0 / control_dt
    old_time = (len(motion_dof_raw) - 1) / old_fps
    frame_count = int(math.floor(old_time * new_fps))
    sample_time = np.arange(frame_count, dtype=np.float64) / new_fps
    source_pos = sample_time * old_fps
    lo = np.floor(source_pos).astype(np.int64)
    hi = np.ceil(source_pos).astype(np.int64)
    blend = (source_pos - lo).reshape(-1, 1)
    root_trans = root_trans_raw[lo] * (1.0 - blend) + root_trans_raw[hi] * blend
    motion_dof = motion_dof_raw[lo] * (1.0 - blend) + motion_dof_raw[hi] * blend
    root_wxyz = root_wxyz_raw[lo] * (1.0 - blend) + root_wxyz_raw[hi] * blend
    root_wxyz /= np.linalg.norm(root_wxyz, axis=1, keepdims=True)
    source_names = list(motion["data_joint_names"])
    motion_to_mj = np.asarray([source_names.index(name) for name in mj_names])
    frame_count = len(motion_dof)

    data.qpos[:3] = root_trans[0]
    if args.diagnostic_no_contact:
        data.qpos[2] += 1.0
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
    data.qpos[3:7] = root_wxyz[0]
    data.qpos[qadr] = motion_dof[0, motion_to_mj]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(args.policy, map_location="cpu").eval()
    history = np.zeros((history_frames, 75), dtype=np.float32)
    last_action_policy = np.zeros(21, dtype=np.float32)
    target_mj = data.qpos[qadr].copy()

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, contract["robot"]["root_link"])
    if base_id < 0:
        raise RuntimeError("base_link is missing from MJCF")
    foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_body_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_body_names
    ])
    if np.any(foot_body_ids < 0):
        raise RuntimeError(f"Foot body missing from MJCF: {foot_body_names}")
    capture_names = list(motion.get("target_link_names", []))
    capture_ref = np.asarray(motion.get("target_link_pos", []), dtype=np.float64)
    capture_pairs = []
    for index, name in enumerate(capture_names):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            capture_pairs.append((index, body_id))

    total_control_steps = int(round(args.duration / control_dt))
    if frame_count != total_control_steps + 1:
        raise RuntimeError(
            f"Resampled reference has {frame_count} frames; expected "
            f"{total_control_steps + 1} for {args.duration}s at {new_fps}Hz"
        )

    # Training computes capture points from the resampled robot state. Build
    # the same reference in this engine rather than indexing raw 30 Hz points.
    capture_ref = np.zeros((frame_count, len(capture_pairs), 3), dtype=np.float64)
    ref_data = mujoco.MjData(model)
    for frame in range(frame_count):
        ref_data.qpos[:3] = root_trans[frame]
        ref_data.qpos[3:7] = root_wxyz[frame]
        ref_data.qpos[qadr] = motion_dof[frame, motion_to_mj]
        ref_data.qvel[:] = 0.0
        mujoco.mj_forward(model, ref_data)
        for pair_index, (_, body_id) in enumerate(capture_pairs):
            capture_ref[frame, pair_index] = ref_data.xpos[body_id]
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.55
    camera.azimuth = 135.0
    camera.elevation = -10.0
    writer = imageio.get_writer(out_dir / f"{args.name}_MuJoCo.mp4", fps=50, codec="libx264", quality=8, pixelformat="yuv420p")

    logs = {key: [] for key in (
        "obs", "raw_action_policy", "reference_target_mj", "raw_target_mj",
        "executed_target_mj", "effective_action_policy", "q_mj", "dq_mj",
        "motor_torque_mj", "passive_friction_torque_mj", "net_torque_mj",
        "base_pos", "tilt_deg", "capture_error", "foot_contact_resultant", "foot_pos_w",
    )}
    traces = []
    first_fall = None
    first_capture = None

    def reference_mj(control_step: int) -> np.ndarray:
        return motion_dof[min(control_step, frame_count - 1), motion_to_mj]

    try:
        for control_step in range(total_control_steps):
            phase = control_step / total_control_steps
            q_mj = data.qpos[qadr].copy()
            dq_mj = data.qvel[dadr].copy()
            q_policy = q_mj[mj_for_policy]
            dq_policy = dq_mj[mj_for_policy]

            local_velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, local_velocity, 1)
            # Free-root angular velocity used by the Isaac observation contract.
            # mj_objectVelocity returned a non-equivalent body velocity here and
            # made the closed loop diverge from the second policy cycle.
            base_ang_vel = data.qvel[3:6].copy()
            quat_wxyz = data.qpos[3:7].copy()
            rotation = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
            projected_gravity = rotation.inv().apply(np.array([0.0, 0.0, -1.0]))
            a = 2.0 * math.pi * phase
            current_obs = np.concatenate([
                base_ang_vel,
                projected_gravity,
                q_policy,
                dq_policy,
                last_action_policy,
                [math.sin(a), math.cos(a), math.sin(2*a), math.cos(2*a), math.sin(4*a), math.cos(4*a)],
            ]).astype(np.float32)
            if control_step == 0:
                history[:] = current_obs
            else:
                history[:-1] = history[1:]
                history[-1] = current_obs
            policy_obs = history.reshape(1, -1)
            with torch.inference_mode():
                policy_out = policy(torch.from_numpy(policy_obs))
                if isinstance(policy_out, tuple):
                    policy_out = policy_out[0]
                raw_action_policy = policy_out.detach().cpu().numpy()[0]

            clipped_policy = np.clip(raw_action_policy, -100.0, 100.0)
            raw_action_mj = clipped_policy[policy_to_mj]
            reference = reference_mj(control_step)
            raw_target = reference + raw_action_mj * action_scale
            executed_target = np.clip(raw_target, lower, upper)
            effective_action_mj = (executed_target - reference) / action_scale
            effective_action_policy = effective_action_mj[mj_for_policy].astype(np.float32)
            last_action_input = last_action_policy.copy()
            last_action_policy = effective_action_policy
            target_mj = executed_target

            first_motor = first_passive = first_net = None
            for _ in range(control_stride):
                q_now = data.qpos[qadr]
                dq_now = data.qvel[dadr]
                motor = np.clip(kp * (target_mj - q_now) - kd * dq_now, -peak, peak)
                passive = -(coulomb * np.tanh(dq_now / 0.01) + viscous * dq_now)
                net = motor + passive
                if first_motor is None:
                    first_motor, first_passive, first_net = motor.copy(), passive.copy(), net.copy()
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[dadr] = net
                mujoco.mj_step(model, data)
                if args.diagnostic_velocity_clamp:
                    data.qvel[dadr] = np.clip(data.qvel[dadr], -velocity_limit, velocity_limit)
                logs["motor_torque_mj"].append(motor.copy())
                logs["passive_friction_torque_mj"].append(passive.copy())
                logs["net_torque_mj"].append(net.copy())

            root_pos = data.qpos[:3].copy()
            quat_wxyz = data.qpos[3:7].copy()
            rotation = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
            up = rotation.apply(np.array([0.0, 0.0, 1.0]))
            tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
            capture_error = 0.0
            if capture_pairs and capture_ref.size:
                frame = min(control_step, frame_count - 1)
                for pair_index, (_, body_id) in enumerate(capture_pairs):
                    capture_error += float(np.sum((capture_ref[frame, pair_index] - data.xpos[body_id]) ** 2))
            foot_contact_resultant = np.zeros((2, 3), dtype=np.float64)
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                body1 = model.geom_bodyid[contact.geom1]
                body2 = model.geom_bodyid[contact.geom2]
                if body1 not in foot_body_ids and body2 not in foot_body_ids:
                    continue
                contact_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_index, contact_force)
                force_world = contact.frame.reshape(3, 3).T @ contact_force[:3]
                for foot_index, foot_body_id in enumerate(foot_body_ids):
                    if body2 == foot_body_id:
                        foot_contact_resultant[foot_index] += force_world
                    elif body1 == foot_body_id:
                        foot_contact_resultant[foot_index] -= force_world
            t = (control_step + 1) * control_dt
            if first_fall is None and (root_pos[2] < 0.20 or tilt > 60.0):
                first_fall = t
            if first_capture is None and capture_error > 0.5:
                first_capture = t

            control_values = {
                "obs": policy_obs[0],
                "raw_action_policy": raw_action_policy,
                "reference_target_mj": reference,
                "raw_target_mj": raw_target,
                "executed_target_mj": executed_target,
                "effective_action_policy": effective_action_policy,
                "q_mj": data.qpos[qadr].copy(),
                "dq_mj": data.qvel[dadr].copy(),
                "base_pos": root_pos,
                "tilt_deg": np.asarray([tilt]),
                "capture_error": np.asarray([capture_error]),
                "foot_contact_resultant": foot_contact_resultant,
                "foot_pos_w": data.xpos[foot_body_ids].copy(),
            }
            for key, value in control_values.items():
                logs[key].append(np.asarray(value).copy())

            if control_step < 2:
                traces.append({
                    "step": control_step,
                    "observation": policy_obs[0].tolist(),
                    "last_action_input": last_action_input.tolist(),
                    "raw_policy_action": raw_action_policy.tolist(),
                    "reference_target_rad_mujoco_order": reference.tolist(),
                    "raw_target_rad_mujoco_order": raw_target.tolist(),
                    "clamped_target_rad_mujoco_order": executed_target.tolist(),
                    "effective_action_policy_order": effective_action_policy.tolist(),
                    "raw_target_oob_count": int(np.sum((raw_target < lower) | (raw_target > upper))),
                    "executed_target_oob_count": int(np.sum((executed_target < lower - 1e-7) | (executed_target > upper + 1e-7))),
                    "motor_torque_nm_mujoco_order": first_motor.tolist(),
                    "passive_friction_torque_nm_mujoco_order": first_passive.tolist(),
                    "net_torque_nm_mujoco_order": first_net.tolist(),
                    "foot_contact_resultant": foot_contact_resultant.tolist(),
                })

            camera.lookat[:] = root_pos
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()

    arrays = {key: np.asarray(value) for key, value in logs.items()}
    raw_oob = (arrays["raw_target_mj"] < lower) | (arrays["raw_target_mj"] > upper)
    exec_oob = (arrays["executed_target_mj"] < lower - 1e-7) | (arrays["executed_target_mj"] > upper + 1e-7)
    motor = arrays["motor_torque_mj"]
    net = arrays["net_torque_mj"]
    net_rms = np.sqrt(np.mean(net ** 2, axis=0))
    base = arrays["base_pos"]
    metrics = {
        "engine": "MuJoCo",
        "policy": str(Path(args.policy).resolve()),
        "contract_sha256": sha256(args.contract),
        "motion_sha256": sha256(args.motion),
        "duration_s": args.duration,
        "physics_dt_s": physics_dt,
        "policy_period_s": control_dt,
        "actor_input": 750,
        "actor_output": 21,
        "joint_names_policy_order": policy_order,
        "joint_names_mujoco_order": mj_names,
        "fell": first_fall is not None,
        "fall_time_s": first_fall,
        "first_capture_threshold_s": first_capture,
        "minimum_base_height_m": float(np.min(base[:, 2])),
        "maximum_tilt_deg": float(np.max(arrays["tilt_deg"])),
        "final_displacement_m": (base[-1] - base[0]).tolist(),
        "raw_target_oob_total": int(raw_oob.sum()),
        "executed_target_oob_total": int(exec_oob.sum()),
        "target_edge_rate_mujoco_order": np.mean((np.abs(arrays["executed_target_mj"] - lower) < 1e-3) | (np.abs(arrays["executed_target_mj"] - upper) < 1e-3), axis=0).tolist(),
        "net_torque_rms_nm_mujoco_order": net_rms.tolist(),
        "net_torque_abs_peak_nm_mujoco_order": np.max(np.abs(net), axis=0).tolist(),
        "motor_peak_saturation_rate_mujoco_order": np.mean(np.abs(motor) >= peak * 0.999, axis=0).tolist(),
        "continuous_rating_ratio_mujoco_order": (net_rms / continuous).tolist(),
        "actuator_model": "explicit_PD_motor_clip_then_explicit_SI_Coulomb_and_viscous_friction",
        "passive_MJCF_joint_friction_and_damping_disabled": True,
        "diagnostic_post_step_velocity_clamp": bool(args.diagnostic_velocity_clamp),
        "diagnostic_all_contacts_disabled": bool(args.diagnostic_no_contact),
        "automatic_reset": False,
        "randomization": False,
    }
    np.savez_compressed(out_dir / f"{args.name}_MuJoCo_dump.npz", **arrays)
    (out_dir / f"{args.name}_MuJoCo_first2.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{args.name}_MuJoCo_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EDU3_REFERENCE_MUJOCO_EVAL=PASS", flush=True)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
