"""Shared EDU3 deploy constants for MuJoCo sim2sim (matches Edu3-Flat training)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from joint_order import ISAAC_JOINT_NAMES

# From edu3_nqj13_flat/gait_params.py
STEP_CYCLE_PERIOD = 0.5
GAIT_PHASE_OFFSET = 0.5
GAIT_STANCE_DUTY = 0.615
LIN_VEL_X_MIN = -0.4
LIN_VEL_X_MAX = 0.4

def _resolve_asset_root() -> Path:
    """MJCF/URDF live in the edu3 asset package (not in this repro slice alone)."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3],  # .../edu3_nqj13_trainable_fullbody_v1 when installed under asset/scripts/...
        here.parents[4] / "01_asset_task" / "edu3_nqj13_trainable_fullbody_v1",
        Path("/home/joyin/edu3_nqj13_trainable_fullbody_v1_FINAL/edu3_nqj13_trainable_fullbody_v1"),
    ]
    for cand in candidates:
        if (cand / "mjcf" / "edu3_nqj13_trainable_fullbody.xml").is_file():
            return cand
    raise FileNotFoundError(
        "EDU3 MJCF not found. Install sim2sim under "
        "<asset>/scripts/mujoco/edu3_sim/ or place the asset package at "
        "01_asset_task/edu3_nqj13_trainable_fullbody_v1/"
    )


ASSET_ROOT = _resolve_asset_root()
MJCF_PATH = ASSET_ROOT / "mjcf" / "edu3_nqj13_trainable_fullbody.xml"
URDF_PATH = ASSET_ROOT / "urdf" / "edu3_nqj13_trainable_fullbody.urdf"

# Prefer companion checkpoint in the repro bundle; override with --load_model.
_BUNDLE_POLICY = Path(__file__).resolve().parents[4] / "04_checkpoint" / "exported" / "policy.pt"
_ASSET_POLICY = ASSET_ROOT / "scripts" / "mujoco" / "edu3_sim" / "policy.pt"
DEFAULT_POLICY_PATH = next(
    (
        p
        for p in (
            _BUNDLE_POLICY,
            _ASSET_POLICY,
            Path(
                "/home/joyin/roboparty_train/robolab/logs/rsl_rl/"
                "edu3_flat_phase_rl-20260720/2026-07-20_11-19-54/exported/policy.pt"
            ),
        )
        if p.is_file()
    ),
    _BUNDLE_POLICY,
)

# From EDU3_NQJ13_TRAINABLE_CFG.init_state.joint_pos (unspecified joints = 0).
DEFAULT_JOINT_POS_RAD: dict[str, float] = {
    "left_thigh_pitch_joint": -0.10,
    "right_thigh_pitch_joint": 0.10,
    "left_knee_joint": 0.30,
    "right_knee_joint": -0.30,
    "left_ankle_pitch_joint": -0.20,
    "right_ankle_pitch_joint": 0.20,
    "left_arm_pitch_joint": -0.15,
    "right_arm_pitch_joint": 0.15,
    "left_arm_roll_joint": 0.05,
    "right_arm_roll_joint": -0.05,
    "left_elbow_pitch_joint": 0.60,
    "right_elbow_pitch_joint": -0.60,
}

# Per-actuator PD + effort (MeasuredFrictionDelayedPDActuatorCfg groups).
# MJCF already encodes coulomb/viscous via frictionloss/damping — do not double-apply.
JOINT_PD: dict[str, tuple[float, float, float]] = {}
for _n in ("left_thigh_pitch_joint", "right_thigh_pitch_joint", "left_knee_joint", "right_knee_joint"):
    JOINT_PD[_n] = (100.0, 3.0, 50.0)
for _n in ("left_thigh_yaw_joint", "right_thigh_yaw_joint", "left_thigh_roll_joint", "right_thigh_roll_joint", "torso_joint"):
    JOINT_PD[_n] = (80.0, 2.5, 20.0)
for _n in (
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
):
    JOINT_PD[_n] = (40.0, 1.5, 40.0)
for _n in (
    "left_arm_pitch_joint",
    "right_arm_pitch_joint",
    "left_arm_roll_joint",
    "right_arm_roll_joint",
    "left_arm_yaw_joint",
    "right_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
):
    JOINT_PD[_n] = (30.0, 1.2, 10.0)


def default_joint_pos_rad() -> np.ndarray:
    return np.array(
        [float(DEFAULT_JOINT_POS_RAD.get(name, 0.0)) for name in ISAAC_JOINT_NAMES],
        dtype=np.float64,
    )


def pd_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kp = np.array([JOINT_PD[n][0] for n in ISAAC_JOINT_NAMES], dtype=np.float64)
    kd = np.array([JOINT_PD[n][1] for n in ISAAC_JOINT_NAMES], dtype=np.float64)
    effort = np.array([JOINT_PD[n][2] for n in ISAAC_JOINT_NAMES], dtype=np.float64)
    return kp, kd, effort


def build_policy_obs(
    omega_body: np.ndarray,
    projected_gravity: np.ndarray,
    command: np.ndarray,
    phase_angle: float,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    default_pos: np.ndarray,
    *,
    clip_obs: float = 100.0,
) -> np.ndarray:
    """Build 74-dim actor observation matching Edu3-Flat / BaseEnv."""
    obs = np.zeros(74, dtype=np.float32)
    obs[0:3] = omega_body
    obs[3:6] = projected_gravity
    obs[6:9] = command
    obs[9] = math.sin(phase_angle)
    obs[10] = math.cos(phase_angle)
    obs[11:32] = joint_pos - default_pos
    obs[32:53] = joint_vel
    obs[53:74] = last_action
    return np.clip(obs, -clip_obs, clip_obs)
