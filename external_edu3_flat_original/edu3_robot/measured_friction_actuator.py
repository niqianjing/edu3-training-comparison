"""Isaac Lab actuator with measured output-side Coulomb and viscous friction.

The PhysX joint ``friction`` property is unitless and must stay zero for EDU3.
This explicit model applies the measured quantities in torque units after the
motor command delay and motor-effort clipping.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class MeasuredFrictionDelayedPDActuator(DelayedPDActuator):
    """Delayed PD motor plus explicit passive friction torque."""

    cfg: "MeasuredFrictionDelayedPDActuatorCfg"

    def __init__(self, cfg: "MeasuredFrictionDelayedPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.coulomb_friction_nm < 0.0 or cfg.viscous_damping_nm_s_per_rad < 0.0:
            raise ValueError("Measured friction and damping must be non-negative")
        if cfg.sign_smoothing_velocity <= 0.0:
            raise ValueError("sign_smoothing_velocity must be positive")
        print(
            "EDU3_MEASURED_FRICTION_MODEL=PASS "
            f"joints={','.join(self.joint_names)} "
            f"coulomb_nm={cfg.coulomb_friction_nm:.6g} "
            f"viscous_nm_s_per_rad={cfg.viscous_damping_nm_s_per_rad:.6g} "
            f"sign_smoothing_velocity={cfg.sign_smoothing_velocity:.6g}",
            flush=True,
        )

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        motor_effort = control_action.joint_efforts
        eps = max(float(self.cfg.sign_smoothing_velocity), 1.0e-6)
        resisting_effort = (
            float(self.cfg.coulomb_friction_nm) * torch.tanh(joint_vel / eps)
            + float(self.cfg.viscous_damping_nm_s_per_rad) * joint_vel
        )
        self.motor_effort = motor_effort
        self.passive_friction_effort = -resisting_effort
        self.applied_effort = motor_effort + self.passive_friction_effort
        control_action.joint_efforts = self.applied_effort
        return control_action


@configclass
class MeasuredFrictionDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration with output-side friction in explicit SI units."""

    class_type: type = MeasuredFrictionDelayedPDActuator
    coulomb_friction_nm: float = MISSING
    viscous_damping_nm_s_per_rad: float = MISSING
    sign_smoothing_velocity: float = 0.01

