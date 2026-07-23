from isaaclab.utils import configclass

from mimic_real.data import MOTION_DATA_DIR
from mimic_real.envs.mimic.hi_mimic_capture_rsi_config import (
    HIMimicCaptureRSIAgentCfg,
    HIMimicCaptureRSIEnvCfg,
)


@configclass
class HIMimicCaptureRSIV2EnvCfg(HIMimicCaptureRSIEnvCfg):
    """Xiaohai R2 contract with only the contact-aware walk reference replaced."""

    def __post_init__(self):
        super().__post_init__()
        self.motion_data.motion_file_path = MOTION_DATA_DIR + "/hi/walk_v2_clearance.json"


@configclass
class HIMimicCaptureRSIV2AgentCfg(HIMimicCaptureRSIAgentCfg):
    experiment_name = "hi_mimic_capture_rsi_v2_clearance"
    wandb_project = "hi_mimic_capture_rsi_v2_clearance"
