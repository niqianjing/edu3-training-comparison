"""Edu3-Flat env: BaseEnv plus mandatory legacy PhysX friction gate."""

from __future__ import annotations

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

from robolab.tasks.direct.base.base_env import BaseEnv

from .robot_cfg import set_legacy_joint_friction_checked


class Edu3FlatEnv(BaseEnv):
    """Fail-closed EDU3 training env (friction gate required by asset README)."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        set_legacy_joint_friction_checked(
            self,
            None,
            SceneEntityCfg("robot", joint_names=".*"),
            value=0.0,
        )
