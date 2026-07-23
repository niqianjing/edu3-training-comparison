"""Isolated compatibility bridge for the received Edu3-Flat task."""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _walking_mask(env, threshold: float) -> torch.Tensor:
    cmd = env.command_generator.command
    return (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > threshold


def _standing_mask(env, threshold: float = 0.01) -> torch.Tensor:
    cmd = env.command_generator.command
    return (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < threshold


def _upright(env) -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def _leg_phase(env) -> torch.Tensor:
    if hasattr(env, "leg_phase"):
        return env.leg_phase
    period = float(env.cfg.robot.gait_phase_period)
    offset = float(env.cfg.robot.gait_phase_offset)
    phase = (env.episode_length_buf.float() * env.step_dt) % period / period
    return torch.stack((phase, (phase + offset) % 1.0), dim=-1)


def _contacts(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1).amax(dim=1) > 1.0


def lin_vel_x_stall_penalty(env, min_ratio: float, cmd_threshold: float,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_x = env.command_generator.command[:, 0]
    vel_x = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )[:, 0]
    cmd_mag = torch.abs(cmd_x)
    directed_speed = vel_x * torch.sign(cmd_x)
    deficit = torch.relu(min_ratio * cmd_mag - directed_speed) / cmd_mag.clamp(min=cmd_threshold)
    return deficit * (cmd_mag > cmd_threshold).float() * _upright(env)


def lin_vel_x_overspeed_penalty(env, max_ratio: float, cmd_threshold: float,
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_x = env.command_generator.command[:, 0]
    vel_x = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )[:, 0]
    cmd_mag = torch.abs(cmd_x)
    directed_speed = vel_x * torch.sign(cmd_x)
    excess = torch.relu(directed_speed - max_ratio * cmd_mag) / cmd_mag.clamp(min=cmd_threshold)
    return excess * (cmd_mag > cmd_threshold).float() * _upright(env)


def feet_phase_contact_xnor(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contacts = _contacts(env, sensor_cfg)
    stance = _leg_phase(env) < float(env.cfg.robot.gait_phase_duty)
    return (contacts == stance).float().sum(dim=1) * _walking_mask(
        env, float(env.cfg.robot.cmd_active_threshold)
    ).float() * _upright(env)


def feet_swing_contact_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contacts = _contacts(env, sensor_cfg)
    swing = _leg_phase(env) >= float(env.cfg.robot.gait_phase_duty)
    return (contacts & swing).float().sum(dim=1) * _walking_mask(
        env, float(env.cfg.robot.cmd_active_threshold)
    ).float() * _upright(env)


def feet_swing_height_penalty(env, sensor_cfg1: SceneEntityCfg, sensor_cfg2: SceneEntityCfg,
                              target_height: float, ankle_height: float) -> torch.Tensor:
    height = torch.stack(
        [
            env.scene[sensor_cfg1.name].data.pos_w[:, 2]
            - env.scene[sensor_cfg1.name].data.ray_hits_w[..., 2].mean(dim=-1),
            env.scene[sensor_cfg2.name].data.pos_w[:, 2]
            - env.scene[sensor_cfg2.name].data.ray_hits_w[..., 2].mean(dim=-1),
        ], dim=-1,
    )
    clearance = torch.clamp(height - ankle_height, min=0.0)
    clearance = torch.nan_to_num(clearance, nan=target_height, posinf=target_height, neginf=0.0)
    swing = _leg_phase(env) >= float(env.cfg.robot.gait_phase_duty)
    deficit = torch.relu(target_height - clearance)
    return torch.square(deficit).mul(swing.float()).sum(dim=1) * _walking_mask(
        env, float(env.cfg.robot.cmd_active_threshold)
    ).float() * _upright(env)


def feet_contact_no_vel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    contacts = _contacts(env, sensor_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    speed2 = torch.sum(torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]), dim=-1)
    return (speed2 * contacts.float()).sum(dim=1)


def hip_pos_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(error), dim=1) * _upright(env)


def single_foot_stance_without_cmd(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contacts = _contacts(env, sensor_cfg)
    return (contacts.sum(dim=1) == 1).float() * _standing_mask(env).float() * _upright(env)


def stand_still(env, pos_cfg: SceneEntityCfg, vel_cfg: SceneEntityCfg, pos_weight: float = 1.0,
                vel_weight: float = 1.0, body_vel_threshold: float = 0.5) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1) + torch.abs(
        asset.data.root_ang_vel_b[:, 2]
    )
    pos_cost = pos_weight * torch.sum(
        torch.abs(asset.data.joint_pos[:, pos_cfg.joint_ids] - asset.data.default_joint_pos[:, pos_cfg.joint_ids]), dim=1
    )
    vel_cost = vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]), dim=1)
    enabled = _standing_mask(env) & (body_vel <= body_vel_threshold)
    return (pos_cost + vel_cost) * enabled.float() * _upright(env)


def install() -> None:
    from robolab.tasks.direct.base import mdp
    functions = {
        "lin_vel_x_stall_penalty": lin_vel_x_stall_penalty,
        "lin_vel_x_overspeed_penalty": lin_vel_x_overspeed_penalty,
        "feet_phase_contact_xnor": feet_phase_contact_xnor,
        "feet_swing_contact_penalty": feet_swing_contact_penalty,
        "feet_swing_height_penalty": feet_swing_height_penalty,
        "feet_contact_no_vel": feet_contact_no_vel,
        "hip_pos_l2": hip_pos_l2,
        "single_foot_stance_without_cmd": single_foot_stance_without_cmd,
        "stand_still": stand_still,
    }
    for name, function in functions.items():
        setattr(mdp, name, function)
    print("EXTERNAL_EDU3_COMPAT_V1=INSTALLED functions=" + ",".join(functions), flush=True)
