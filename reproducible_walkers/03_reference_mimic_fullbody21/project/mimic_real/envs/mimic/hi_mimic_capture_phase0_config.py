from isaaclab.utils import configclass

from .hi_mimic_et_phase0_config import (
    HIMimicETPhase0AgentCfg,
    HIMimicETPhase0EnvCfg,
)


@configclass
class HIMimicCapturePhase0EnvCfg(HIMimicETPhase0EnvCfg):
    """Recovery task: phase-0 alignment with capture-point termination only."""

    def __post_init__(self):
        super().__post_init__()
        self.terminate.terminate_contacts = False
        self.terminate.terminate_capture_points_far = True


@configclass
class HIMimicCapturePhase0AgentCfg(HIMimicETPhase0AgentCfg):
    experiment_name = "hi_mimic_capture_phase0"
    wandb_project = "hi_mimic_capture_phase0"
