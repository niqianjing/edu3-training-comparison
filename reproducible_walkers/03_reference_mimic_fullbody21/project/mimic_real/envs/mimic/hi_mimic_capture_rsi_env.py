import torch
import isaaclab.utils.math as math_utils

from .hi_mimic_env import BaseEnv


class HIMimicCaptureRSIEnv(BaseEnv):
    """Capture-only task with physically synchronized random reference-state resets."""

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.extras["log"] = dict()
        self.scene.reset(env_ids)

        # One random reference frame drives both the clock and the physical state.
        max_step = int(self.max_episode_length)
        random_steps = torch.randint(
            low=0,
            high=max_step,
            size=(len(env_ids),),
            device=self.device,
            dtype=self.episode_length_buf.dtype,
        )
        self.episode_length_buf[env_ids] = random_steps
        self.phase[env_ids] = self.episode_length_buf[env_ids] / self.max_episode_length

        # Joint pose and velocity come from the exact same reference phase.
        ref_joint_pos = self.motion_loader.get_dof_pos_batch(self.phase[env_ids])
        ref_joint_pos += math_utils.sample_uniform(
            *self.cfg.domain_rand.reset_robot_joints.params["position_range"],
            ref_joint_pos.shape,
            self.device,
        )
        ref_joint_vel = self.motion_loader.get_dof_vel_batch(self.phase[env_ids])
        ref_joint_vel += math_utils.sample_uniform(
            *self.cfg.domain_rand.reset_robot_joints.params["velocity_range"],
            ref_joint_vel.shape,
            self.device,
        )
        self.robot.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, env_ids=env_ids)

        # Root pose and velocity are synchronized to that same phase.
        ref_positions = (
            self.motion_loader.get_root_trans_batch(self.phase[env_ids])
            + self.scene.env_origins[env_ids]
        ).to(torch.float32)
        ref_orientations = self.motion_loader.get_root_rot_batch(self.phase[env_ids]).to(torch.float32)

        pose_range = self.cfg.domain_rand.reset_robot_base.params["pose_range"]
        range_list = [
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        ref_positions += rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        ref_orientations = math_utils.quat_mul(ref_orientations, orientations_delta)

        ref_velocities = self.robot.data.default_root_state[env_ids, 7:13].clone()
        ref_velocities[:, 0:3] = self.motion_loader.get_root_vel_batch(self.phase[env_ids])
        ref_velocities[:, 3:6] = self.motion_loader.get_root_omega_batch(self.phase[env_ids])

        velocity_range = self.cfg.domain_rand.reset_robot_base.params["velocity_range"]
        range_list = [
            velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        ref_velocities += rand_samples

        self.robot.write_root_pose_to_sim(
            torch.cat([ref_positions, ref_orientations], dim=-1), env_ids=env_ids
        )
        self.robot.write_root_velocity_to_sim(ref_velocities, env_ids=env_ids)

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        # CircularBuffer repeats the first post-reset observation across history,
        # so all ten frames begin from the same synchronized physical/reference state.
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.last_action[env_ids] = 0.0
        self.action_buffer.reset(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()