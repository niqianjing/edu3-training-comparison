from isaaclab.utils import configclass

from .hi_mimic_et_phase0_config import (
    HIMimicETPhase0AgentCfg,
    HIMimicETPhase0EnvCfg,
)


@configclass
class HIMimicETBasePhase0EnvCfg(HIMimicETPhase0EnvCfg):
    """Phase-0 recovery: retain capture termination and terminate only on base contact."""

    def __post_init__(self):
        super().__post_init__()
        self.terminate.terminate_contacts = True
        self.terminate.terminate_capture_points_far = True
        self.terminate.terminate_contacts_body_names = ["base_link"]


@configclass
class HIMimicETBasePhase0AgentCfg(HIMimicETPhase0AgentCfg):
    experiment_name = "hi_mimic_et_base_phase0"
    wandb_project = "hi_mimic_et_base_phase0"
