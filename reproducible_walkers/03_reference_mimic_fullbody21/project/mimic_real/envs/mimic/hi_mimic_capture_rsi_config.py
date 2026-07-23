from isaaclab.utils import configclass

from .hi_mimic_capture_phase0_config import (
    HIMimicCapturePhase0AgentCfg,
    HIMimicCapturePhase0EnvCfg,
)


@configclass
class HIMimicCaptureRSIEnvCfg(HIMimicCapturePhase0EnvCfg):
    """Single-variable recovery: capture termination plus synchronized random RSI."""


@configclass
class HIMimicCaptureRSIAgentCfg(HIMimicCapturePhase0AgentCfg):
    experiment_name = "hi_mimic_capture_rsi"
    wandb_project = "hi_mimic_capture_rsi"