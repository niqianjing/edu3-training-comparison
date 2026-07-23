from isaaclab.utils import configclass

from .hi_mimic_config import HIMimicAgentCfg, HIMimicEnvCfg


@configclass
class HIMimicETOnlyEnvCfg(HIMimicEnvCfg):
    """Original Xiaohai full-body task with early termination restored only."""

    def __post_init__(self):
        super().__post_init__()
        self.terminate.terminate_contacts = True
        self.terminate.terminate_capture_points_far = True


@configclass
class HIMimicETOnlyAgentCfg(HIMimicAgentCfg):
    experiment_name = "hi_mimic_et_only"
    wandb_project = "hi_mimic_et_only"
