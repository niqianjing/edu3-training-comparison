"""Edu3-Flat gym task (phase RL on edu3_nqj13 trainable asset)."""

import gymnasium as gym

gym.register(
    id="Edu3-Flat",
    entry_point="edu3_nqj13_flat.edu3_flat_env:Edu3FlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.edu3_flat_env_cfg:Edu3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.edu3_flat_agent_cfg:Edu3FlatAgentCfg",
    },
)
