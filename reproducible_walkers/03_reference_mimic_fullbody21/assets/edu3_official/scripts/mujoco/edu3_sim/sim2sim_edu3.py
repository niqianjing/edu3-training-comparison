# MuJoCo Sim2Sim for EDU3 nqj13 (Edu3-Flat).
# Physics: dt=0.005, decimation=4 (50 Hz policy, 200 Hz physics) — aligned with Isaac Lab.
# Recording camera matches roboparty mini_sim (front-facing).

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from actuator_backend import PhysicsPDBackend
from edu3_deploy_config import (
    DEFAULT_POLICY_PATH,
    LIN_VEL_X_MAX,
    LIN_VEL_X_MIN,
    MJCF_PATH,
    STEP_CYCLE_PERIOD,
    URDF_PATH,
    build_policy_obs,
    default_joint_pos_rad,
    pd_arrays,
)
from joint_order import ISAAC_JOINT_NAMES, assert_isaac_joint_order

DEFAULT_FRAME_STACK = 10
NUM_SINGLE_OBS = 74
NUM_ACTIONS = 21

CMD_LIN_VEL_X = (LIN_VEL_X_MIN, LIN_VEL_X_MAX)

# Front-facing camera (same as roboparty mini_sim).
CAM_DISTANCE = 1.6
CAM_AZIMUTH = 180.0
CAM_ELEVATION = -12.0
CAM_LOOKAT_Z = 0.22


def reset_policy_state(policy: torch.jit.ScriptModule) -> None:
    if hasattr(policy, "hidden_state"):
        policy.hidden_state.zero_()
    if hasattr(policy, "cell_state"):
        policy.cell_state.zero_()


def infer_policy_input_dim(policy: torch.nn.Module) -> int | None:
    try:
        for mod in policy.modules():
            if isinstance(mod, torch.nn.Linear):
                return int(mod.in_features)
    except Exception:
        return None
    return None


class ObsHistoryStack:
    """IsaacLab CircularBuffer-compatible actor obs history (oldest → newest, flat)."""

    def __init__(self, frame_stack: int, obs_dim: int) -> None:
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")
        self.frame_stack = int(frame_stack)
        self.obs_dim = int(obs_dim)
        self._buf: np.ndarray | None = None

    def reset(self) -> None:
        self._buf = None

    def append(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        if self._buf is None:
            self._buf = np.tile(x, (self.frame_stack, 1))
        else:
            self._buf = np.concatenate((self._buf[1:], x.reshape(1, -1)), axis=0)
        return self._buf.reshape(-1).astype(np.float32)


def get_obs(data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quat_mj = data.qpos[3:7]
    quat_scipy = np.array([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]], dtype=np.double)
    rot = R.from_quat(quat_scipy)
    v_body = rot.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega_body = data.qvel[3:6].astype(np.double)
    gvec = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return v_body, omega_body, gvec


def load_edu3_model(*, offscreen_width: int = 1920, offscreen_height: int = 1080) -> mujoco.MjModel:
    """Load package MJCF, add checker ground (mini_sim style), hide collision hulls from view."""
    import tempfile

    xml = MJCF_PATH.read_text(encoding="utf-8")
    # Inject skybox + checker ground textures into existing <asset>.
    inject = (
        '  <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" '
        'width="512" height="512"/>\n'
        '    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" '
        'width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>\n'
        '    <material name="matplane" reflectance="0.3" texture="texplane" '
        'texrepeat="1 1" texuniform="true"/>\n'
    )
    xml = xml.replace("<asset>\n", "<asset>\n" + inject, 1)
    xml = xml.replace(
        '<geom name="floor" type="plane" size="0 0 0.1" rgba="0.25 0.25 0.25 1" />',
        '<geom name="floor" type="plane" size="0 0 1" pos="0 0 0" material="matplane" '
        'contype="1" conaffinity="1" condim="3" friction="1 0.005 0.0001" group="1"/>\n'
        '    <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" '
        'pos="0 0 4" dir="0 0 -1"/>\n'
        '    <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" '
        'pos="0 0 5" dir="0 0 -1" castshadow="false"/>',
    )

    # Write beside original MJCF so meshdir="../meshes" still resolves.
    fd, temp_path = tempfile.mkstemp(prefix="_sim2sim_", suffix=".xml", dir=str(MJCF_PATH.parent))
    os.close(fd)
    try:
        Path(temp_path).write_text(xml, encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass

    model.opt.timestep = 0.005
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(offscreen_width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(offscreen_height))

    # Visual meshes = group 2; collision hulls ("膨胀箱") → group 3 (hidden in render option).
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name == "floor":
            model.geom_group[i] = 1
        elif "_collision_" in name:
            model.geom_group[i] = 3
    return model


def make_render_option() -> mujoco.MjvOption:
    """Show visual meshes + ground; hide collision expansion hulls."""
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1  # floor
    opt.geomgroup[2] = 1  # visual meshes
    return opt


def make_front_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    cam.lookat[:] = [0.0, 0.0, CAM_LOOKAT_Z]
    return cam


def _follow_base_camera(cam: mujoco.MjvCamera, data: mujoco.MjData) -> None:
    cam.lookat[0] = float(data.qpos[0])
    cam.lookat[1] = float(data.qpos[1])
    cam.lookat[2] = CAM_LOOKAT_Z


def run_mujoco(
    policy: torch.jit.ScriptModule | None,
    cfg,
    *,
    headless: bool = False,
    record_video: bool = True,
    cmd_vx: float = 0.2,
    cmd_vy: float = 0.0,
    cmd_dyaw: float = 0.0,
    video_path: str = "simulation.mp4",
    video_width: int = 1920,
    video_height: int = 1080,
) -> None:
    print(f"Loading MJCF from: {MJCF_PATH}")
    joint_names = assert_isaac_joint_order(URDF_PATH)
    print(f"Using {len(joint_names)} joints in Isaac Lab / PhysX DOF order.")

    model = load_edu3_model(offscreen_width=video_width, offscreen_height=video_height)
    data = mujoco.MjData(model)
    print(f"MuJoCo revolute joints count: {model.nv - 6}")

    qpos_indices = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
    )
    qvel_indices = np.array(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names]
    )
    # Map Isaac-order torque → MuJoCo actuator index (motors named "<joint>_motor").
    act_indices = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
            for n in joint_names
        ]
    )
    if np.any(act_indices < 0):
        missing = [n for n, i in zip(joint_names, act_indices) if i < 0]
        raise RuntimeError(f"Missing MuJoCo actuators for joints: {missing}")

    def reset_state() -> None:
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = [0.0, 0.0, cfg.robot_config.initial_height]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[qpos_indices] = cfg.robot_config.default_pos
        mujoco.mj_forward(model, data)

    reset_state()
    cam_template = make_front_camera()
    scene_option = make_render_option()

    if headless and not record_video:
        renderer = out = cam = viewer = None
        print("Headless metrics-only (no video).")
    elif headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(model, width=video_width, height=video_height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cam = make_front_camera()
        _follow_base_camera(cam, data)
        fps = 1.0 / (cfg.sim_config.dt * cfg.sim_config.decimation)
        out = cv2.VideoWriter(video_path, fourcc, fps, (video_width, video_height))
        if not out.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at {video_path}")
        viewer = None
        print(f"Headless video: {video_width}x{video_height} @ {fps:.0f} Hz → {video_path}")
    else:
        import mujoco_viewer

        viewer = mujoco_viewer.MujocoViewer(model, data, mode="window", width=1920, height=1080)
        viewer.cam.distance = cam_template.distance
        viewer.cam.azimuth = cam_template.azimuth
        viewer.cam.elevation = cam_template.elevation
        _follow_base_camera(viewer.cam, data)
        # Hide collision hulls; show visuals + floor.
        viewer.vopt.geomgroup[:] = 0
        viewer.vopt.geomgroup[1] = 1
        viewer.vopt.geomgroup[2] = 1
        renderer = out = cam = None
        print("\n>>> Press [R] in the viewer to reset robot and policy state <<<\n")

    kp, kd, effort = pd_arrays()
    actuator_backend = PhysicsPDBackend(kp, kd, effort)
    # 1 control-step delay (~20 ms); Isaac delay range is 2–8 physics steps.
    target_history: list[np.ndarray] = [cfg.robot_config.default_pos.copy() for _ in range(2)]

    action = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    delayed_target_pos = cfg.robot_config.default_pos.copy()

    frame_stack = int(getattr(cfg.robot_config, "frame_stack", DEFAULT_FRAME_STACK))
    num_single_obs = int(getattr(cfg.robot_config, "num_single_obs", NUM_SINGLE_OBS))
    obs_history = ObsHistoryStack(frame_stack, num_single_obs)
    expected_policy_dim = frame_stack * num_single_obs
    if policy is not None:
        inferred = infer_policy_input_dim(policy)
        if inferred is not None and inferred != expected_policy_dim:
            raise RuntimeError(
                f"Policy expects obs dim {inferred}, but MuJoCo stack is "
                f"{frame_stack}×{num_single_obs}={expected_policy_dim}. "
                f"Pass --frame_stack {max(1, inferred // num_single_obs)}."
            )

    log_interval_policy_steps = max(1, int(round(1.0 / (cfg.sim_config.dt * cfg.sim_config.decimation))))
    log_vx_samples: list[float] = []
    log_omega_samples: list[np.ndarray] = []
    log_z_samples: list[float] = []
    log_err_samples: list[float] = []
    log_action_max_samples: list[float] = []
    hud_vx: list[float] = []
    hud_wz: list[float] = []

    count_lowlevel = 0
    policy_step_count = 0
    step = 0
    max_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)

    if policy is not None:
        reset_policy_state(policy)

    print(
        f"Running simulation (policy={'on' if policy is not None else 'off'}) "
        f"cmd=({cmd_vx:.3f}, {cmd_vy:.3f}, {cmd_dyaw:.3f}) "
        f"obs={frame_stack}×{num_single_obs}={expected_policy_dim} ..."
    )

    import glfw

    while (headless and step < max_steps) or (viewer is not None and viewer.is_alive):
        if viewer is not None and glfw.get_key(viewer.window, glfw.KEY_R) == glfw.PRESS:
            print("[Reset] robot + policy")
            reset_state()
            target_history = [cfg.robot_config.default_pos.copy() for _ in range(2)]
            action.fill(0.0)
            delayed_target_pos = cfg.robot_config.default_pos.copy()
            obs_history.reset()
            policy_step_count = 0
            count_lowlevel = 0
            hud_vx.clear()
            hud_wz.clear()
            if policy is not None:
                reset_policy_state(policy)

        q = data.qpos[qpos_indices].copy()
        dq = data.qvel[qvel_indices].copy()

        if count_lowlevel % cfg.sim_config.decimation == 0:
            v_body, omega_body, gvec = get_obs(data)
            phase = (
                policy_step_count * cfg.sim_config.dt * cfg.sim_config.decimation
            ) / cfg.robot_config.gait_phase_period % 1.0
            phase_angle = phase * 2.0 * np.pi

            if policy is not None:
                single_obs = build_policy_obs(
                    omega_body,
                    gvec,
                    np.array([cmd_vx, cmd_vy, cmd_dyaw], dtype=np.float64),
                    phase_angle,
                    q,
                    dq,
                    action,
                    cfg.robot_config.default_pos,
                )
                policy_obs = obs_history.append(single_obs)
                with torch.inference_mode():
                    raw_action = policy(
                        torch.tensor(policy_obs.reshape(1, -1), dtype=torch.float32)
                    )[0].detach().numpy()
                action[:] = np.clip(raw_action, -cfg.robot_config.clip_actions, cfg.robot_config.clip_actions)

            target_pos = action * cfg.robot_config.action_scale + cfg.robot_config.default_pos
            target_history.append(target_pos.copy())
            if len(target_history) > 2:
                target_history.pop(0)
            delayed_target_pos = target_history[0].copy()

            policy_step_count += 1
            log_vx_samples.append(float(v_body[0]))
            log_omega_samples.append(omega_body.copy())
            log_z_samples.append(float(data.qpos[2]))
            log_err_samples.append(abs(cmd_vx - v_body[0]))
            log_action_max_samples.append(float(np.max(np.abs(action))))
            hud_vx.append(float(v_body[0]))
            hud_wz.append(float(omega_body[2]))

            if policy_step_count % log_interval_policy_steps == 0:
                omega_avg = np.mean(log_omega_samples, axis=0)
                print(
                    f"[t={step * cfg.sim_config.dt:.1f}s] "
                    f"cmd_vx={cmd_vx:+.3f} "
                    f"vx_avg={np.mean(log_vx_samples):+.3f}m/s "
                    f"err_avg={np.mean(log_err_samples):.3f} "
                    f"cmd_wz={cmd_dyaw:+.3f} "
                    f"wz_avg={omega_avg[2]:+.3f}rad/s "
                    f"z_avg={np.mean(log_z_samples):.3f}m "
                    f"|action|_max_avg={np.mean(log_action_max_samples):.3f}"
                )
                log_vx_samples.clear()
                log_omega_samples.clear()
                log_z_samples.clear()
                log_err_samples.clear()
                log_action_max_samples.clear()

            if headless and renderer is not None and out is not None and cam is not None:
                _follow_base_camera(cam, data)
                renderer.update_scene(data, camera=cam, scene_option=scene_option)
                frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                vx_avg = float(np.mean(hud_vx)) if hud_vx else float(v_body[0])
                wz_avg = float(np.mean(hud_wz)) if hud_wz else float(omega_body[2])
                cv2.putText(
                    frame,
                    f"cmd_vx={cmd_vx:+.3f} cmd_wz={cmd_dyaw:+.3f} | "
                    f"vx={v_body[0]:+.3f} wz={omega_body[2]:+.3f} z={data.qpos[2]:.3f}",
                    (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"avg: vx={vx_avg:+.3f} wz={wz_avg:+.3f}",
                    (16, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (180, 255, 180),
                    2,
                    cv2.LINE_AA,
                )
                out.write(frame)
            elif viewer is not None:
                _follow_base_camera(viewer.cam, data)
                viewer.render()

        act_out = actuator_backend.step(delayed_target_pos, q, dq)
        data.ctrl[:] = 0.0
        data.ctrl[act_indices] = act_out.tau_applied
        mujoco.mj_step(model, data)
        count_lowlevel += 1
        step += 1

    if log_vx_samples:
        omega_avg = np.mean(log_omega_samples, axis=0)
        print(
            f"[t={step * cfg.sim_config.dt:.1f}s] (partial) "
            f"cmd_vx={cmd_vx:+.3f} "
            f"vx_avg={np.mean(log_vx_samples):+.3f}m/s "
            f"err_avg={np.mean(log_err_samples):.3f} "
            f"wz_avg={omega_avg[2]:+.3f}rad/s "
            f"z_avg={np.mean(log_z_samples):.3f}m"
        )

    if out is not None:
        out.release()
        print(f"Video saved to: {video_path}")
    elif viewer is not None:
        viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuJoCo Sim2Sim for EDU3 nqj13 (Edu3-Flat).")
    parser.add_argument(
        "--load_model",
        type=str,
        default=str(DEFAULT_POLICY_PATH),
        help="Exported JIT policy.pt path.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI and save video.")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration in seconds.")
    parser.add_argument("--vx", type=float, default=0.2, help="Forward velocity command (m/s).")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s).")
    parser.add_argument("--dyaw", type=float, default=0.0, help="Yaw rate command (rad/s).")
    parser.add_argument(
        "--gait-period",
        type=float,
        default=STEP_CYCLE_PERIOD,
        help=f"Gait phase period in seconds (default {STEP_CYCLE_PERIOD}).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=str(_THIS_DIR / "simulation.mp4"),
        help="Headless video output path.",
    )
    parser.add_argument("--video_width", type=int, default=1920)
    parser.add_argument("--video_height", type=int, default=1080)
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=DEFAULT_FRAME_STACK,
        help=f"Actor obs history length (default {DEFAULT_FRAME_STACK} → {DEFAULT_FRAME_STACK * NUM_SINGLE_OBS}).",
    )
    parser.add_argument("--no_policy", action="store_true", help="PD hold test only (action=0).")
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="With --headless, skip MuJoCo Renderer / mp4 (metrics only).",
    )
    cli = parser.parse_args()

    if cli.vx < CMD_LIN_VEL_X[0] or cli.vx > CMD_LIN_VEL_X[1]:
        print(f"WARNING: vx={cli.vx} outside training range {CMD_LIN_VEL_X}")

    class Sim2simCfg:
        class sim_config:
            sim_duration = cli.duration
            dt = 0.005
            decimation = 4

        class robot_config:
            default_pos = default_joint_pos_rad()
            action_scale = 0.25
            clip_actions = 1.0
            initial_height = 0.40
            gait_phase_period = cli.gait_period
            num_actions = NUM_ACTIONS
            num_single_obs = NUM_SINGLE_OBS
            frame_stack = cli.frame_stack

    policy = None if cli.no_policy else torch.jit.load(cli.load_model, map_location="cpu")
    if policy is None and not cli.no_policy:
        parser.error("--load_model is required unless --no_policy is set.")

    run_mujoco(
        policy,
        Sim2simCfg(),
        headless=cli.headless,
        record_video=not cli.no_video,
        cmd_vx=cli.vx,
        cmd_vy=cli.vy,
        cmd_dyaw=cli.dyaw,
        video_path=cli.video_path,
        video_width=cli.video_width,
        video_height=cli.video_height,
    )
