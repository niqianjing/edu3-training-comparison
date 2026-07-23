# from mimic_real.envs.base.clean_env import BaseEnv
# from mimic_real.envs.base.base_env_config import BaseEnvCfg, BaseAgentCfg
# from mimic_real.utils.task_registry import task_registry

# from mimic_real.envs.hi_clean.hi_clean_config import HICleanAgentCfg, HICleanEnvCfg
# task_registry.register("hi_clean", BaseEnv, HICleanEnvCfg(), HICleanAgentCfg())

from mimic_real.envs.mimic.hi_mimic_env import BaseEnv
from mimic_real.utils.task_registry import task_registry
from mimic_real.envs.mimic.hi_mimic_config import HIMimicAgentCfg, HIMimicEnvCfg
task_registry.register("hi_mimic", BaseEnv, HIMimicEnvCfg(), HIMimicAgentCfg())
from mimic_real.envs.mimic.hi_mimic_et_only_config import HIMimicETOnlyAgentCfg, HIMimicETOnlyEnvCfg
task_registry.register("hi_mimic_et_only", BaseEnv, HIMimicETOnlyEnvCfg(), HIMimicETOnlyAgentCfg())
from mimic_real.envs.mimic.hi_mimic_et_phase0_config import HIMimicETPhase0AgentCfg, HIMimicETPhase0EnvCfg
task_registry.register("hi_mimic_et_phase0", BaseEnv, HIMimicETPhase0EnvCfg(), HIMimicETPhase0AgentCfg())
from mimic_real.envs.mimic.hi_mimic_et_diag_config import (
    HIMimicETCaptureDiagAgentCfg,
    HIMimicETCaptureDiagEnvCfg,
    HIMimicETContactDiagAgentCfg,
    HIMimicETContactDiagEnvCfg,
)
task_registry.register("hi_mimic_et_contact_diag", BaseEnv, HIMimicETContactDiagEnvCfg(), HIMimicETContactDiagAgentCfg())
task_registry.register("hi_mimic_et_capture_diag", BaseEnv, HIMimicETCaptureDiagEnvCfg(), HIMimicETCaptureDiagAgentCfg())
from mimic_real.envs.mimic.hi_mimic_et_base_phase0_config import HIMimicETBasePhase0AgentCfg, HIMimicETBasePhase0EnvCfg
task_registry.register("hi_mimic_et_base_phase0", BaseEnv, HIMimicETBasePhase0EnvCfg(), HIMimicETBasePhase0AgentCfg())
from mimic_real.envs.mimic.hi_mimic_diag_env import HIMimicDiagnosticEnv
task_registry.register("hi_mimic_et_probe_values", HIMimicDiagnosticEnv, HIMimicETBasePhase0EnvCfg(), HIMimicETBasePhase0AgentCfg())
from mimic_real.envs.mimic.hi_mimic_capture_phase0_config import HIMimicCapturePhase0AgentCfg, HIMimicCapturePhase0EnvCfg
task_registry.register("hi_mimic_capture_phase0", BaseEnv, HIMimicCapturePhase0EnvCfg(), HIMimicCapturePhase0AgentCfg())


from mimic_real.envs.mimic.hi_mimic_capture_rsi_env import HIMimicCaptureRSIEnv
from mimic_real.envs.mimic.edu3_reference_mimic_config import (
    EDU3ReferenceMimicAgentCfg,
    EDU3ReferenceMimicEnvCfg,
)
task_registry.register(
    "edu3_reference_mimic_r1",
    HIMimicCaptureRSIEnv,
    EDU3ReferenceMimicEnvCfg(),
    EDU3ReferenceMimicAgentCfg(),
)
