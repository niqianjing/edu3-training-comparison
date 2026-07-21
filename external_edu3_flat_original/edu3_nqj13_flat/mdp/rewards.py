"""EDU3-specific MDP helpers (no WBC / joint-reference tracking).

Phase gait contact/height terms live in ``robolab.tasks.direct.base.mdp``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from robolab.tasks.direct.base.base_env import BaseEnv


def _walking_mask(env: BaseEnv) -> torch.Tensor:
    cmd = env.command_generator.command
    cmd_active = float(getattr(env.cfg.robot, "cmd_active_threshold", 0.01))
    return (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > cmd_active


def _knee_flexion_rad(env: BaseEnv, asset, joint_ids: torch.Tensor) -> torch.Tensor:
    """EDU3: left knee flexes +, right knee flexes −; return positive flexion (rad)."""
    q = asset.data.joint_pos[:, joint_ids]
    cache = getattr(env, "_edu3_knee_sign", None)
    if cache is None or cache.shape[0] != joint_ids.numel():
        signs = []
        for jid in joint_ids.tolist():
            name = asset.data.joint_names[int(jid)]
            signs.append(1.0 if "left" in name else -1.0)
        cache = torch.tensor(signs, device=q.device, dtype=q.dtype)
        env._edu3_knee_sign = cache
    return q * cache.unsqueeze(0)


def arm_default_when_standing(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_arm_pitch_joint"]),
) -> torch.Tensor:
    """L1 arm-pitch deviation from default when velocity command is near zero."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(q - q0), dim=1)

    standing = ~_walking_mask(env)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return penalty * standing.float() * upright


def hip_yaw_excess_l2(
    env: BaseEnv,
    deadzone_rad: float = math.radians(2.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_thigh_yaw.*"]),
) -> torch.Tensor:
    """Penalize thigh_yaw only beyond a small deadzone (both legs)."""
    asset = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    excess = torch.clamp(torch.abs(deviation) - deadzone_rad, min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def swing_hip_yaw_l2(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_thigh_yaw.*"]),
) -> torch.Tensor:
    """Extra yaw penalty on the swing leg — blocks inward-sweep clearance cheats."""
    asset = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    duty = float(env.cfg.robot.gait_phase_duty)
    is_swing = env.leg_phase >= duty
    # leg_phase columns are [left, right]; assume joint_ids ordered L then R via name sort — match by name.
    joint_ids = asset_cfg.joint_ids
    if not torch.is_tensor(joint_ids):
        joint_ids = torch.as_tensor(joint_ids, device=deviation.device)
    left_ids = []
    right_ids = []
    for local_i, jid in enumerate(joint_ids.tolist()):
        name = asset.data.joint_names[int(jid)]
        if "left" in name:
            left_ids.append(local_i)
        else:
            right_ids.append(local_i)
    penal = torch.zeros(env.num_envs, device=deviation.device)
    if left_ids:
        penal = penal + torch.square(deviation[:, left_ids]).sum(dim=1) * is_swing[:, 0].float()
    if right_ids:
        penal = penal + torch.square(deviation[:, right_ids]).sum(dim=1) * is_swing[:, 1].float()
    return penal * _walking_mask(env).float()


def swing_knee_flexion(
    env: BaseEnv,
    target_flex_rad: float = 0.55,
    std: float = 0.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
) -> torch.Tensor:
    """Reward swing-leg knee flexion near a human-like target (both legs)."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if not torch.is_tensor(joint_ids):
        joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    flex = _knee_flexion_rad(env, asset, joint_ids)

    duty = float(env.cfg.robot.gait_phase_duty)
    is_swing = env.leg_phase >= duty
    # Map knee columns to L/R via joint names.
    left_cols = []
    right_cols = []
    for local_i, jid in enumerate(joint_ids.tolist()):
        name = asset.data.joint_names[int(jid)]
        if "left" in name:
            left_cols.append(local_i)
        else:
            right_cols.append(local_i)

    rew = torch.zeros(env.num_envs, device=flex.device)
    n = torch.zeros(env.num_envs, device=flex.device)
    if left_cols:
        err2 = torch.square(flex[:, left_cols[0]] - target_flex_rad)
        term = torch.exp(-err2 / (std * std)) * is_swing[:, 0].float()
        rew = rew + term
        n = n + is_swing[:, 0].float()
    if right_cols:
        err2 = torch.square(flex[:, right_cols[0]] - target_flex_rad)
        term = torch.exp(-err2 / (std * std)) * is_swing[:, 1].float()
        rew = rew + term
        n = n + is_swing[:, 1].float()
    rew = rew / n.clamp(min=1.0)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return rew * _walking_mask(env).float() * upright


def base_pitch_l2(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base pitch both ways (forward and backward lean)."""
    asset = env.scene[asset_cfg.name]
    gx = asset.data.projected_gravity_b[:, 0]
    return torch.square(gx)
