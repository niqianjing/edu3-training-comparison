from isaaclab.utils import configclass

from .hi_mimic_et_phase0_config import (
    HIMimicETPhase0AgentCfg,
    HIMimicETPhase0EnvCfg,
)


@configclass
class HIMimicETContactDiagEnvCfg(HIMimicETPhase0EnvCfg):
    """Phase-0 diagnostic: terminate only on forbidden-body contact."""

    def __post_init__(self):
        super().__post_init__()
        self.terminate.terminate_contacts = True
        self.terminate.terminate_capture_points_far = False


@configclass
class HIMimicETContactDiagAgentCfg(HIMimicETPhase0AgentCfg):
    experiment_name = "hi_mimic_et_contact_diag"
    wandb_project = "hi_mimic_et_contact_diag"


@configclass
class HIMimicETCaptureDiagEnvCfg(HIMimicETPhase0EnvCfg):
    """Phase-0 diagnostic: terminate only on capture-point distance."""

    def __post_init__(self):
        super().__post_init__()
        self.terminate.terminate_contacts = False
        self.terminate.terminate_capture_points_far = True


@configclass
class HIMimicETCaptureDiagAgentCfg(HIMimicETPhase0AgentCfg):
    experiment_name = "hi_mimic_et_capture_diag"
    wandb_project = "hi_mimic_et_capture_diag"
