"""Numpy PD + effort clamp for EDU3 MuJoCo sim2sim.

Passive Coulomb/viscous friction is already in the MJCF joint frictionloss/damping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActuatorStepResult:
    tau_applied: np.ndarray
    tau_raw: np.ndarray


class PhysicsPDBackend:
    def __init__(self, kp: np.ndarray, kd: np.ndarray, effort_limit: np.ndarray) -> None:
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.effort_limit = np.asarray(effort_limit, dtype=np.float64)

    def step(self, cmd_pos: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray) -> ActuatorStepResult:
        tau_raw = self.kp * (cmd_pos - joint_pos) - self.kd * joint_vel
        tau_applied = np.clip(tau_raw, -self.effort_limit, self.effort_limit)
        return ActuatorStepResult(tau_applied=tau_applied, tau_raw=tau_raw)
