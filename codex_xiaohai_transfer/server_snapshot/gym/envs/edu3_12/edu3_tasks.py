"""Task registrations kept separate from the upstream Xiaohai package."""

from gym.envs.edu3_12.edu3_controller import Edu3HiController
from gym.envs.edu3_12.edu3_controller_config import (
    Edu3XiaohaiOriginalCfg,
    Edu3XiaohaiOriginalRunnerCfg,
    Edu3XiaohaiRoll25Cfg,
    Edu3XiaohaiRoll25RunnerCfg,
    Edu3XiaohaiOriginalFeetFixCfg,
    Edu3XiaohaiOriginalFeetFixRunnerCfg,
    Edu3XiaohaiRoll25FeetFixCfg,
    Edu3XiaohaiRoll25FeetFixRunnerCfg,
    Edu3XiaohaiTiming20msCfg,
    Edu3XiaohaiTiming20msRunnerCfg,
    Edu3XiaohaiTiming10msCfg,
    Edu3XiaohaiTiming10msRunnerCfg,
    Edu3XiaohaiRoll25Timing20msCfg,
    Edu3XiaohaiRoll25Timing20msSeed42RunnerCfg,
    Edu3XiaohaiRoll25Timing20msSeed43RunnerCfg,
)
from gym.utils.task_registry import task_registry


task_registry.register(
    "edu3_xiaohai_original",
    Edu3HiController,
    Edu3XiaohaiOriginalCfg,
    Edu3XiaohaiOriginalRunnerCfg,
)
task_registry.register(
    "edu3_xiaohai_roll25",
    Edu3HiController,
    Edu3XiaohaiRoll25Cfg,
    Edu3XiaohaiRoll25RunnerCfg,
)

task_registry.register(
    "edu3_xiaohai_original_feetfix",
    Edu3HiController,
    Edu3XiaohaiOriginalFeetFixCfg,
    Edu3XiaohaiOriginalFeetFixRunnerCfg,
)
task_registry.register(
    "edu3_xiaohai_roll25_feetfix",
    Edu3HiController,
    Edu3XiaohaiRoll25FeetFixCfg,
    Edu3XiaohaiRoll25FeetFixRunnerCfg,
)

task_registry.register(
    "edu3_xiaohai_timing20ms_r3",
    Edu3HiController,
    Edu3XiaohaiTiming20msCfg,
    Edu3XiaohaiTiming20msRunnerCfg,
)
task_registry.register(
    "edu3_xiaohai_timing10ms_r3",
    Edu3HiController,
    Edu3XiaohaiTiming10msCfg,
    Edu3XiaohaiTiming10msRunnerCfg,
)

task_registry.register(
    "edu3_xiaohai_roll25_timing20ms_r4_seed42",
    Edu3HiController,
    Edu3XiaohaiRoll25Timing20msCfg,
    Edu3XiaohaiRoll25Timing20msSeed42RunnerCfg,
)
task_registry.register(
    "edu3_xiaohai_roll25_timing20ms_r4_seed43",
    Edu3HiController,
    Edu3XiaohaiRoll25Timing20msCfg,
    Edu3XiaohaiRoll25Timing20msSeed43RunnerCfg,
)