from isaaclab.utils import configclass

from .hi_mimic_et_only_config import (
    HIMimicETOnlyAgentCfg,
    HIMimicETOnlyEnvCfg,
)


@configclass
class HIMimicETPhase0EnvCfg(HIMimicETOnlyEnvCfg):
    """ET task used with the isolated phase-0-aligned training entrypoint."""


@configclass
class HIMimicETPhase0AgentCfg(HIMimicETOnlyAgentCfg):
    experiment_name = "hi_mimic_et_phase0"
    wandb_project = "hi_mimic_et_phase0"
