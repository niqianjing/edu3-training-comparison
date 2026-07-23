"""EDU3 body driven by Xiaohai's footstep controller with SI motor losses."""

from __future__ import annotations

import torch

from gym.envs.hi_12.hi_controller import HiController
from gym.utils.math import exp_avg_filter


class Edu3HiController(HiController):
    """Keep the Xiaohai task/rewards and replace only the motor physics."""

    def _init_buffers(self):
        super()._init_buffers()
        if bool(getattr(self.cfg.control, "enforce_right_left_feet", False)):
            right_id = self.rigid_body_idx["r_ankle_roll_link"]
            left_id = self.rigid_body_idx["l_ankle_roll_link"]
            self.feet_ids = torch.tensor(
                [right_id, left_id], dtype=torch.long, device=self.device
            )
            print("EDU3_FEET_SEMANTIC_ORDER_PASS", ["right", "left"], self.feet_ids.tolist())
        roll25 = bool(self.cfg.control.roll25)
        tc = []
        viscous = []
        continuous = []
        for name in self.dof_names:
            is_25 = ("hip_pitch" in name or "calf" in name or (roll25 and "hip_roll" in name))
            tc.append(0.51 if is_25 else 0.146)
            viscous.append(0.0432 if is_25 else 0.0306)
            continuous.append(7.0 if is_25 else 3.75)
        self.edu3_coulomb = torch.tensor(tc, dtype=torch.float, device=self.device).unsqueeze(0)
        self.edu3_viscous = torch.tensor(viscous, dtype=torch.float, device=self.device).unsqueeze(0)
        self.edu3_continuous = torch.tensor(continuous, dtype=torch.float, device=self.device).unsqueeze(0)
        self.edu3_motor_torques = torch.zeros_like(self.torques)
        self.edu3_friction_torques = torch.zeros_like(self.torques)
        self.edu3_net_torques = torch.zeros_like(self.torques)
        print("EDU3_EXPLICIT_SI_FRICTION_ENABLED")
        print("edu3_coulomb_Nm=", tc)
        print("edu3_viscous_Nms_per_rad=", viscous)
        print("edu3_continuous_Nm=", continuous)

    def _compute_torques(self):
        if self.cfg.control.exp_avg_decay:
            self.dof_pos_avg = exp_avg_filter(
                self.dof_pos_target, self.dof_pos_avg, self.cfg.control.exp_avg_decay
            )
            target = self.dof_pos_avg
        else:
            target = self.dof_pos_target

        motor = self.p_gains * (
            target * self.cfg.control.actuation_scale + self.default_dof_pos - self.dof_pos
        ) - self.d_gains * self.dof_vel
        motor = torch.clip(motor, -self.torque_limits, self.torque_limits)
        friction = self.edu3_coulomb * torch.tanh(
            self.dof_vel / float(self.cfg.control.coulomb_vel_eps)
        ) + self.edu3_viscous * self.dof_vel
        net = torch.clip(motor - friction, -self.torque_limits, self.torque_limits)
        self.edu3_motor_torques.copy_(motor)
        self.edu3_friction_torques.copy_(friction)
        self.edu3_net_torques.copy_(net)
        return net
