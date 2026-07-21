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

import numpy as np
import torch
from collections.abc import Sequence
from isaaclab.envs import DirectRLEnv
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils

from .base_config import BaseEnvCfg


class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg
    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.reward_manager = RewardManager(self.cfg.reward, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene_context.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        self.left_feet_scanner_cfg = SceneEntityCfg("left_feet_scanner")
        self.right_feet_scanner_cfg = SceneEntityCfg("right_feet_scanner")

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self._reset_idx(env_ids)

    def init_buffers(self):
        self.extras = {}

        self.episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = CircularBuffer(
            max_len=self.cfg.robot.action_history_length, batch_size=self.num_envs, device=self.device
        )
        self.action_buffer.append(torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_left = torch.zeros(self.num_envs, device=self.device)
        self.phase_right = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)
        # Unwrapped gait-clock accumulator (cycles) and whether the clock is
        # currently parked at a double-support anchor (phase 0 or 0.5).
        self._gait_phase_accum = torch.zeros(self.num_envs, device=self.device)
        self._gait_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.init_obs_buffer()

    def _recompute_phase_outputs(self) -> None:
        offset = self.cfg.robot.gait_phase_offset
        self.phase = torch.remainder(self._gait_phase_accum, 1.0)
        self.phase_left = self.phase
        self.phase_right = torch.remainder(self.phase + offset, 1.0)
        self.leg_phase = torch.stack([self.phase_left, self.phase_right], dim=1)

    def _update_gait_phase(self) -> None:
        """Advance the open-loop gait clock by one control step.

        The clock free-runs while a velocity command is active. Once the
        command drops to (near) zero it keeps advancing only until the next
        double-support instant (phase 0 or 0.5) — so an in-flight step
        finishes naturally instead of freezing a swing leg mid-air — and
        then locks there until a walk command resumes, at which point it
        continues seamlessly from that same double-support instant.
        """
        if self.cfg.robot.gait_phase_period <= 0.0:
            return
        period = self.cfg.robot.gait_phase_period
        increment = self.step_dt / period

        command = self.command_generator.command
        cmd_active = float(getattr(self.cfg.robot, "cmd_active_threshold", 0.01))
        walking = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > cmd_active
        advancing = walking | (~self._gait_locked)

        old_phase = torch.remainder(self._gait_phase_accum, 1.0)
        self._gait_phase_accum = torch.where(
            advancing, self._gait_phase_accum + increment, self._gait_phase_accum
        )
        new_phase = torch.remainder(self._gait_phase_accum, 1.0)

        # A double-support anchor (0 or 0.5) was just crossed this step.
        crossed_anchor = (new_phase < old_phase) | ((old_phase < 0.5) & (new_phase >= 0.5))
        newly_locked = (~walking) & crossed_anchor
        self._gait_locked = torch.where(
            walking, torch.zeros_like(self._gait_locked), self._gait_locked | newly_locked
        )

        self._recompute_phase_outputs()

    def _reset_gait_phase(self, env_ids: Sequence[int]) -> None:
        """Reset the gait clock for the given envs to a parked double-support state.

        Starting locked at phase 0 (rather than free-running) means a fresh
        episode only starts stepping once the first command actually calls
        for it — pure-standing episodes never take a "free" half-step before
        the stand-still logic can lock them.
        """
        if self.cfg.robot.gait_phase_period <= 0.0:
            return
        self._gait_phase_accum[env_ids] = 0.0
        self._gait_locked[env_ids] = True
        self._recompute_phase_outputs()

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            obs_offset = 0
            noise_vec[obs_offset : obs_offset + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            obs_offset += 3
            noise_vec[obs_offset : obs_offset + 3] = (
                noise_scales.projected_gravity * self.obs_scales.projected_gravity
            )
            obs_offset += 3
            noise_vec[obs_offset : obs_offset + 3] = 0.0
            obs_offset += 3
            if self.cfg.robot.gait_phase_period > 0.0:
                noise_vec[obs_offset : obs_offset + 2] = 0.0
                obs_offset += 2
            noise_vec[obs_offset : obs_offset + self.num_actions] = (
                noise_scales.joint_pos * self.obs_scales.joint_pos
            )
            obs_offset += self.num_actions
            noise_vec[obs_offset : obs_offset + self.num_actions] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            obs_offset += self.num_actions
            noise_vec[obs_offset : obs_offset + self.num_actions] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene_context.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
                height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
                height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
                height_scan *= self.obs_scales.height_scan
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer.buffer[:, -1, :]
        actor_obs_terms = [
            ang_vel * self.obs_scales.ang_vel,
            projected_gravity * self.obs_scales.projected_gravity,
            command * self.obs_scales.commands,
        ]
        if self.cfg.robot.gait_phase_period > 0.0:
            phase_angle = self.phase.unsqueeze(-1) * (2.0 * torch.pi)
            actor_obs_terms.append(torch.cat([torch.sin(phase_angle), torch.cos(phase_angle)], dim=-1))
        actor_obs_terms.extend(
            [
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
            ]
        )
        current_actor_obs = torch.cat(
            actor_obs_terms,
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
        feet_contact_force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :]
        feet_air_time = self.contact_sensor.data.current_air_time[:, self.feet_cfg.body_ids]
        feet_height = torch.stack(
        [
            self.scene[sensor_cfg.name].data.pos_w[:, 2]
            - self.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [self.left_feet_scanner_cfg, self.right_feet_scanner_cfg]
            if sensor_cfg is not None
        ],
        dim=-1,
        )
        feet_height = torch.clamp(feet_height - 0.04, min=0.0, max=1.0)
        feet_height = torch.nan_to_num(feet_height, nan=1.0, posinf=1.0, neginf=0)
        joint_torque = robot.data.applied_torque
        joint_acc = robot.data.joint_acc
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact.float(), feet_contact_force.flatten(1), feet_air_time.flatten(1), feet_height.flatten(1), joint_acc, joint_torque], dim=-1
        )
        
        return current_actor_obs, current_critic_obs


    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)

        self._pre_physics_step(actions)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self._update_gait_phase()
        self.common_step_counter += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        
        # ── Debug extras (lean set; keep only decision-useful signals) ──
        # Keep these in a local dict because _reset_idx() rewrites extras["log"].
        raw_actions = self.action_buffer.buffer[:, -1, :]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, :, self.feet_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        is_contact = contact_forces > 1.0

        feet_height = torch.stack(
            [
                self.scene[sensor_cfg.name].data.pos_w[:, 2]
                - self.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
                for sensor_cfg in [self.left_feet_scanner_cfg, self.right_feet_scanner_cfg]
                if sensor_cfg is not None
            ],
            dim=-1,
        )
        actual_h = torch.clamp(feet_height - 0.04, min=0.0, max=1.0)
        actual_h = torch.nan_to_num(actual_h, nan=0.0, posinf=1.0, neginf=0.0)

        is_swing = self.leg_phase >= self.cfg.robot.gait_phase_duty
        command_x = self.command_generator.command[:, 0]
        root_lin_vel_x = self.robot.data.root_lin_vel_b[:, 0]
        lin_vel_x_error = torch.abs(command_x - root_lin_vel_x)
        is_upright = (-self.robot.data.projected_gravity_b[:, 2]) > 0.85
        cmd_active = float(getattr(self.cfg.robot, "cmd_active_threshold", 0.01))
        has_vel_cmd = torch.abs(command_x) > cmd_active
        lin_vel_x_min, lin_vel_x_max = self.cfg.commands.ranges.lin_vel_x
        max_cmd_x = max(abs(float(lin_vel_x_min)), abs(float(lin_vel_x_max)), 1e-6)
        success_err = getattr(self.cfg.robot, "lin_vel_success_error", None)
        if success_err is None:
            success_err = 0.5 * max_cmd_x
        else:
            success_err = float(success_err)
        lin_vel_x_success = (
            (lin_vel_x_error < success_err) & is_upright & ~self.reset_terminated & has_vel_cmd
        )
        n_walking = has_vel_cmd.float().sum().clamp(min=1.0)
        lin_vel_x_error_walking = (lin_vel_x_error * has_vel_cmd.float()).sum() / n_walking
        lin_vel_x_success_rate = lin_vel_x_success.float().sum() / n_walking

        gait_active = has_vel_cmd & is_upright
        if getattr(self.cfg.robot, "debug_gait_metrics_strict", False):
            gait_active_b = gait_active.unsqueeze(-1)
            swing_mask = is_swing & gait_active_b
            air_ok = ~(is_swing & is_contact)
            n_air = swing_mask.float().sum().clamp(min=1.0)
            feet_air_time_success_rate = (air_ok & swing_mask).float().sum() / n_air

            duty = self.cfg.robot.gait_phase_duty
            swing_denom = max(1.0 - duty, 1e-6)
            swing_progress = torch.clamp((self.leg_phase - duty) / swing_denom, 0.0, 1.0)
            in_peak_swing = is_swing & (swing_progress >= 0.4) & (swing_progress <= 0.6)
            height_min = float(getattr(self.cfg.robot, "debug_feet_height_min_m", 0.018))
            height_ok = actual_h >= height_min
            peak_mask = in_peak_swing & gait_active_b
            n_peak = peak_mask.float().sum().clamp(min=1.0)
            feet_height_success_rate = (height_ok & peak_mask).float().sum() / n_peak
        else:
            air_time_success = (~(is_swing & is_contact)).all(dim=1)
            feet_air_time_success_rate = air_time_success.float().mean()
            in_mid_swing = (self.leg_phase >= 0.7) & (self.leg_phase <= 0.9)
            height_ok = actual_h > 0.015
            height_success = (~in_mid_swing | height_ok).all(dim=1)
            feet_height_success_rate = height_success.float().mean()

        action_saturation_rate = (raw_actions.abs() > 0.95 * self.clip_actions).float().mean()

        # Continuous foot load (Contact ≠ Support). Prefer vertical GRF when available.
        feet_ids = self.feet_cfg.body_ids
        grf_z = self.contact_sensor.data.net_forces_w[:, feet_ids, 2].abs()
        grf_mag = contact_forces  # history-max ||F||, same indexing as is_contact
        # Use |Fz| for load share; fall back to magnitude if z-channel is uninformative.
        grf = torch.where(grf_z.sum(dim=-1, keepdim=True) > 1.0, grf_z, grf_mag)
        grf_sum = grf.sum(dim=-1).clamp(min=1e-3)
        grf_load_frac_left = grf[:, 0] / grf_sum
        # Walking-only averages so standing double-support does not pull frac toward 0.5.
        n_walk = has_vel_cmd.float().sum().clamp(min=1.0)
        walk_f = has_vel_cmd.float()
        grf_left_walk = (grf[:, 0] * walk_f).sum() / n_walk
        grf_right_walk = (grf[:, 1] * walk_f).sum() / n_walk
        grf_load_frac_left_walk = (grf_load_frac_left * walk_f).sum() / n_walk
        contact_left = is_contact[:, 0].float().mean()
        contact_right = is_contact[:, 1].float().mean()
        stance_diff = (contact_left - contact_right).abs()
        joint_reference_rms = getattr(self, "_joint_reference_rms", None)

        # Less critical first; gait / load / tracking watch-list last (console prints in this order).
        debug_log = {
            "Debug/base_height": self.robot.data.root_pos_w[:, 2].mean().item(),
            "Debug/projected_gravity_z": (-self.robot.data.projected_gravity_b[:, 2]).mean().item(),
            "Debug/reset_contact_rate": self._termination_contact_buf.float().mean().item(),
            "Debug/reset_terminated_rate": self.reset_terminated.float().mean().item(),
            "Debug/action_saturation_rate": action_saturation_rate.item(),
            "Debug/root_lin_vel_x": root_lin_vel_x.mean().item(),
            "Debug/lin_vel_x_error_walking": lin_vel_x_error_walking.item(),
            # --- watch: contact vs support / gait / tracking ---
            "Debug/feet_contact_left": contact_left.item(),
            "Debug/feet_contact_right": contact_right.item(),
            "Debug/stance_diff": stance_diff.item(),
            "Debug/grf_left": grf_left_walk.item(),
            "Debug/grf_right": grf_right_walk.item(),
            "Debug/grf_load_frac_left": grf_load_frac_left_walk.item(),
            "Debug/feet_air_time_success_rate": feet_air_time_success_rate.item(),
            "Debug/feet_height_success_rate": feet_height_success_rate.item(),
            "Debug/lin_vel_x_success_rate": lin_vel_x_success_rate.item(),
        }
        if joint_reference_rms is not None:
            for category, rms in joint_reference_rms.items():
                debug_log[f"Debug/joint_reference_rms_{category}"] = rms.mean().item()
        # for body_name, contact_idx in zip(
        #     self.termination_contact_cfg.body_names,
        #     range(termination_contacts.shape[1]),
        # ):
        #     debug_name = body_name.replace(".*", "").replace("*", "").replace(".", "").replace("/", "_")
        #     debug_log[f"Debug/term_contact/{debug_name}_rate"] = termination_contacts[:, contact_idx].float().mean().item()
        #     debug_log[f"Debug/term_contact/{debug_name}_force"] = termination_contact_forces[:, contact_idx].mean().item()
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
        self.extras.setdefault("log", {})
        self.extras["log"].update(debug_log)

        self.obs_buf = self._get_observations()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.cfg.scene_context.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def _setup_scene(self):
        self.robot: Articulation = self.scene["robot"]
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _pre_physics_step(self, actions: torch.Tensor):
        self.action_buffer.append(actions)
        self.actions = actions.clone()
        self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)
        self.actions = self.actions * self.action_scale + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        if self.cfg.scene_context.height_scanner.enable_height_scan:
            height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
            height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
            height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
            height_scan *= self.obs_scales.height_scan
            current_critic_obs = torch.cat([current_critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            if self.cfg.scene_context.height_scanner.enable_height_scan_actor:
                current_actor_obs = torch.cat([current_actor_obs, height_scan], dim=-1)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        observations = {"policy": actor_obs, "critic":critic_obs}
        return observations
    
    def _get_rewards(self):
        return self.reward_manager.compute(dt=self.step_dt)
    
    def _get_dones(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._termination_contact_buf = torch.zeros_like(terminated_buf)
        self._termination_orientation_buf = torch.zeros_like(terminated_buf)
        self._termination_height_buf = torch.zeros_like(terminated_buf)
        if self.cfg.robot.terminate_contacts_body_names is not None:
            self._termination_contact_buf = torch.any(
                torch.max(
                    torch.norm(
                        net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0,
                dim=1,
            )
            terminated_buf |= self._termination_contact_buf
        if self.cfg.robot.terminate_base_orientation is not None:
            self._termination_orientation_buf = torch.acos(
                torch.clamp(-self.robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
            ).abs() > self.cfg.robot.terminate_base_orientation
            terminated_buf |= self._termination_orientation_buf
        if self.cfg.robot.terminate_base_height is not None:
            self._termination_height_buf = self.robot.data.root_pos_w[:, 2] < self.cfg.robot.terminate_base_height
            terminated_buf |= self._termination_height_buf
        time_out_buf = self.episode_length_buf >= self.episode_length
        return terminated_buf, time_out_buf

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if len(env_ids) == 0:
            return
        
        if self.cfg.scene_context.height_scanner.enable_height_scan:
            self.height_scanner.reset(env_ids)

        self.extras["log"] = dict()
        if self.cfg.scene_context.terrain_generator is not None:
            if self.cfg.scene_context.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self._sim_step_counter // self.cfg.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.reset_time_outs

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._reset_gait_phase(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
