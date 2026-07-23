# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from robolab.envs.base.base_env import BaseEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, :] - env.action_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )

def action_smoothness_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -3, :] - 2*env.action_buffer.buffer[:, -2, :] + env.action_buffer.buffer[:, -1, :] 
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    return torch.sum(is_contact, dim=1)


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_terminated


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    is_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_mode_time = torch.where(is_contact, contact_time, air_time)
    single_stance = torch.sum(is_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, min=0.0, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_phase_contact_biped(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    period: float = 0.6,
    duty_factor: float = 0.6,
) -> torch.Tensor:
    """Reward an alternating biped contact pattern from a simple gait clock."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt / period, 1.0)
    left_stance = phase < duty_factor
    right_stance = torch.remainder(phase + 0.5, 1.0) < duty_factor
    desired_contacts = torch.stack([left_stance, right_stance], dim=1)

    reward = (contacts == desired_contacts).float().mean(dim=1)
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_swing_height_phased(
    env: BaseEnv,
    sensor_cfg1: SceneEntityCfg,
    sensor_cfg2: SceneEntityCfg,
    period: float = 0.6,
    duty_factor: float = 0.6,
    target_height: float = 0.06,
    std: float = 0.01,
    ankle_height: float = 0.04,
) -> torch.Tensor:
    """Reward swing-foot clearance that follows a phase-guided parabolic profile."""
    phase = torch.remainder(env.episode_length_buf.float() * env.step_dt / period, 1.0)
    phase_left = phase
    phase_right = torch.remainder(phase + 0.5, 1.0)

    swing_denom = max(1.0 - duty_factor, 1e-6)
    swing_progress_left = torch.clamp((phase_left - duty_factor) / swing_denom, 0.0, 1.0)
    swing_progress_right = torch.clamp((phase_right - duty_factor) / swing_denom, 0.0, 1.0)
    is_swing_left = phase_left >= duty_factor
    is_swing_right = phase_right >= duty_factor

    target_h_left = target_height * torch.sin(torch.pi * swing_progress_left)
    target_h_right = target_height * torch.sin(torch.pi * swing_progress_right)

    feet_height = torch.stack(
        [
            env.scene[sensor.name].data.pos_w[:, 2]
            - env.scene[sensor.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    actual_h = torch.clamp(feet_height - ankle_height, min=0.0, max=1.0)
    actual_h = torch.nan_to_num(actual_h, nan=0.0, posinf=1.0, neginf=0.0)

    error_left = torch.square(actual_h[:, 0] - target_h_left) * is_swing_left.float()
    error_right = torch.square(actual_h[:, 1] - target_h_right) * is_swing_right.float()
    reward = torch.exp(-error_left / std) + torch.exp(-error_right / std)

    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _cmd_magnitude(env: BaseEnv) -> torch.Tensor:
    return torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(
        env.command_generator.command[:, 2]
    )


def _cmd_active_threshold(env: BaseEnv) -> float:
    return float(getattr(env.cfg.robot, "cmd_active_threshold", 0.01))


def _gait_active_mask(env: BaseEnv) -> torch.Tensor:
    has_command = _cmd_magnitude(env) > _cmd_active_threshold(env)
    is_upright = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return has_command * is_upright


def _overspeed_curriculum_scale(env: BaseEnv) -> float:
    """Ramp overspeed penalty 0→1 over training iters to avoid early lean-back collapse."""
    start = int(getattr(env.cfg.robot, "overspeed_curriculum_start_iter", 2000))
    end = int(getattr(env.cfg.robot, "overspeed_curriculum_end_iter", 6000))
    spi = int(getattr(env.cfg.robot, "steps_per_iteration", 24))
    spi = max(spi, 1)
    it = float(env.common_step_counter) / float(spi)
    if end <= start:
        return 1.0
    return float(min(1.0, max(0.0, (it - start) / (end - start))))


def _feet_contact_mask(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )


def _feet_clearance_heights(
    env: BaseEnv,
    sensor_cfg1: SceneEntityCfg,
    sensor_cfg2: SceneEntityCfg,
    ankle_height: float,
) -> torch.Tensor:
    feet_height = torch.stack(
        [
            env.scene[sensor.name].data.pos_w[:, 2]
            - env.scene[sensor.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    feet_height = torch.clamp(feet_height - ankle_height, min=0.0, max=1.0)
    return torch.nan_to_num(feet_height, nan=0.0, posinf=1.0, neginf=0.0)


def feet_phase_contact_xnor(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    stance_duty: float | None = None,
) -> torch.Tensor:
    """Reward contact states that match the open-loop gait clock (XNOR / equality)."""
    contacts = _feet_contact_mask(env, sensor_cfg)
    duty = env.cfg.robot.gait_phase_duty if stance_duty is None else stance_duty
    is_stance = env.leg_phase < duty
    # Both feet must match the gait clock; partial credit enabled persistent double support.
    reward = (contacts == is_stance).all(dim=1).float()
    return reward * _gait_active_mask(env)


def feet_swing_contact_penalty(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    stance_duty: float | None = None,
) -> torch.Tensor:
    """Penalize feet that remain in contact during their swing phase."""
    contacts = _feet_contact_mask(env, sensor_cfg)
    duty = env.cfg.robot.gait_phase_duty if stance_duty is None else stance_duty
    is_swing = env.leg_phase >= duty
    bad_contact = (contacts & is_swing).float().sum(dim=1)
    return bad_contact * _gait_active_mask(env)


def feet_swing_height_penalty(
    env: BaseEnv,
    sensor_cfg1: SceneEntityCfg,
    sensor_cfg2: SceneEntityCfg,
    target_height: float = 0.05,
    ankle_height: float = 0.04,
) -> torch.Tensor:
    """Penalize swing-phase feet that do not reach the target clearance height."""
    duty = env.cfg.robot.gait_phase_duty
    is_swing = env.leg_phase >= duty
    actual_h = _feet_clearance_heights(env, sensor_cfg1, sensor_cfg2, ankle_height)
    pos_error = torch.square(actual_h - target_height) * is_swing.float()
    return pos_error.sum(dim=1) * _gait_active_mask(env)


def lin_vel_x_stall_penalty(env: BaseEnv, min_ratio: float = 0.5, cmd_threshold: float = 0.005) -> torch.Tensor:
    """Penalize insufficient velocity in the commanded x direction (forward or backward)."""
    command_x = env.command_generator.command[:, 0]
    root_lin_vel_x = env.robot.data.root_lin_vel_b[:, 0]
    cmd_abs = torch.abs(command_x)
    has_cmd = cmd_abs > cmd_threshold
    vel_in_cmd_dir = torch.sign(command_x) * root_lin_vel_x
    stall = (vel_in_cmd_dir < min_ratio * cmd_abs).float()
    return stall * has_cmd.float() * _gait_active_mask(env)


def lin_vel_x_overspeed_penalty(
    env: BaseEnv, max_ratio: float = 1.0, cmd_threshold: float = 0.005
) -> torch.Tensor:
    """Penalize excess speed beyond ``max_ratio * |cmd|`` in the commanded x direction.

    Returned term is linear excess (m/s). Strength is ramped by
    ``overspeed_curriculum_*`` so early training does not force lean-back braking.
    """
    command_x = env.command_generator.command[:, 0]
    root_lin_vel_x = env.robot.data.root_lin_vel_b[:, 0]
    cmd_abs = torch.abs(command_x)
    has_cmd = cmd_abs > cmd_threshold
    vel_in_cmd_dir = torch.sign(command_x) * root_lin_vel_x
    excess = torch.clamp(vel_in_cmd_dir - max_ratio * cmd_abs, min=0.0)
    scale = _overspeed_curriculum_scale(env)
    return excess * has_cmd.float() * _gait_active_mask(env) * scale


def lin_vel_xy_standing_l2(env: BaseEnv, cmd_threshold: float | None = None) -> torch.Tensor:
    """Penalize base planar velocity when the command is near zero."""
    thresh = _cmd_active_threshold(env) if cmd_threshold is None else float(cmd_threshold)
    standing = _cmd_magnitude(env) <= thresh
    v_xy = env.robot.data.root_lin_vel_b[:, :2]
    upright = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return torch.sum(torch.square(v_xy), dim=1) * standing.float() * upright


def feet_contact_no_vel(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear velocity of feet that are in contact."""
    contacts = _feet_contact_mask(env, sensor_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :3]
    penalize = torch.square(feet_vel) * contacts.unsqueeze(-1)
    return penalize.sum(dim=(1, 2)) * _gait_active_mask(env)


def hip_pos_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize hip yaw/roll deviation from default pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(deviation), dim=1)


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = torch.sum(torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=2), dim=1)
    reward = (reward - threshold).clamp(min=0.0, max=max_reward)
    return reward


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]    
    body_orientation = torch.stack(
        [
            math_utils.quat_apply_inverse(
                asset.data.body_quat_w[:, body_id, :], asset.data.GRAVITY_VEC_W
            )
            for body_id in asset_cfg.body_ids
            if body_id is not None
        ],
        dim=-1,
    )
    return torch.sum(torch.sum(torch.square(body_orientation[:, :2, :]), dim=1), dim=-1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 3 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def body_distance_y(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min: float = 0.2, max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0.)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_contact_without_cmd(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward double-support contact when the velocity command is near zero."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    reward = (torch.sum(contacts, dim=-1) == 2).float()
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def single_foot_stance_without_cmd(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize exactly-one-foot contact while the command is near zero.

    Double-support standing is rewarded by ``feet_contact_without_cmd``; single-leg
    balancing at cmd≈0 is a common exploit (especially with sticky contacts) and
    is not gated by gait-phase terms (those are off when not walking).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    n_contact = torch.sum(contacts.float(), dim=-1)
    single = (n_contact == 1).float()
    no_cmd = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < _cmd_active_threshold(env)
    return single * no_cmd.float()


def undesired_foothold(env: BaseEnv, sensor_cfg: SceneEntityCfg, sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None, ankle_height: float = 0.035) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    undesired_contacts = torch.stack(
        [
            torch.sum(
                (env.scene[sensor.name].data.pos_w[:, 2].unsqueeze(1)
                - env.scene[sensor.name].data.ray_hits_w[..., 2]
                - ankle_height) > 0.01,
                dim=-1
            ) / float(env.scene[sensor.name].data.ray_hits_w.shape[1])
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    reward = torch.where(contacts, undesired_contacts, 0.0)
    return reward.sum(dim=1)

def upward(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = -asset.data.projected_gravity_b[:, 2]
    return reward


def stand_still(
    env: BaseEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
    body_vel_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    cmd = _cmd_magnitude(env)
    cmd_thresh = _cmd_active_threshold(env)
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_reward = pos_weight * torch.sum(torch.abs
        (asset.data.joint_pos[:, pos_cfg.joint_ids] - asset.data.default_joint_pos[:, pos_cfg.joint_ids]), dim=1
    )
    vel_reward = vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]), dim=1)
    # body_vel_threshold: if the robot is commanded to stand but keeps wobbling
    # above this speed, the stand-still penalty used to turn OFF — that let a
    # single-leg balance "escape" the default-pose regularizer. Mini uses a
    # lower gate so cmd≈0 still pays stand_still while rocking on one foot.
    reward = torch.where(
        torch.logical_or(cmd > cmd_thresh, body_vel > body_vel_threshold),
        0.0,
        pos_reward + vel_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None, ankle_height: float = 0.035, threshold: float = 0.05):
    """
    Calculates reward based on the clearance of the swing leg from the ground during movement.
    Encourages appropriate lift of the feet during the swing phase of the gait.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = torch.stack(
        [
            env.scene[sensor.name].data.pos_w[:, 2]
            - env.scene[sensor.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    feet_height = torch.clamp(feet_height - ankle_height, min=0.0, max=1.0)
    feet_height = torch.nan_to_num(feet_height, nan=1.0, posinf=1.0, neginf=0)
    # Compute single_stance mask
    single_stance = contacts.sum(dim=1) == 1
    # feet height should be closed to target feet height at the peak
    rew_pos = feet_height > threshold
    reward = torch.where(torch.logical_and(~contacts, single_stance.unsqueeze(-1)), rew_pos.float(), 0.0).sum(dim=1)
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.01
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_deviation_interrupt(env: BaseEnv, asset_cfg1: SceneEntityCfg, asset_cfg2: SceneEntityCfg, weight1: float, weight2: float) -> torch.Tensor:
    """Penalize joint deviation during interruption."""
    # extract the used quantities (to enable type-hinting)
    asset1: Articulation = env.scene[asset_cfg1.name]
    asset2: Articulation = env.scene[asset_cfg2.name]
    angle1 = asset1.data.joint_pos[:, asset_cfg1.joint_ids] - asset1.data.default_joint_pos[:, asset_cfg1.joint_ids]
    angle2 = asset2.data.joint_pos[:, asset_cfg2.joint_ids] - asset2.data.default_joint_pos[:, asset_cfg2.joint_ids]
    reward = weight1 * torch.sum(torch.abs(angle1), dim=1) + weight2 * torch.sum(torch.abs(angle2), dim=1)
    reward *= ~env.interrupt_mask
    return reward

def stand_still_interrupt(
    env: BaseEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    interrupt_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    cmd = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_joint_ids = list(set(pos_cfg.joint_ids) - set(interrupt_cfg.joint_ids))
    vel_joint_ids = list(set(vel_cfg.joint_ids) - set(interrupt_cfg.joint_ids))
    pos_reward = torch.where(env.interrupt_mask, 
                             pos_weight * torch.sum(torch.abs(asset.data.joint_pos[:, pos_joint_ids] - asset.data.default_joint_pos[:, pos_joint_ids]), dim=1), 
                             pos_weight * torch.sum(torch.abs(asset.data.joint_pos[:, pos_cfg.joint_ids] - asset.data.default_joint_pos[:, pos_cfg.joint_ids]), dim=1))
    vel_reward = torch.where(env.interrupt_mask, 
                             vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_joint_ids]), dim=1), 
                             vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]), dim=1))
    reward = torch.where(
        torch.logical_or(cmd > 0.01, body_vel > 0.5),
        0.0,
        pos_reward + vel_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def action_penalty_interrupt(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize action magnitude during interruption."""
    reward = torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    reward *= env.interrupt_mask
    return reward