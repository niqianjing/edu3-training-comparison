"""Load EDU3 articulation cfg by file path (avoids package-name clashes)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ASSET_ROBOT = Path(__file__).resolve().parents[1] / "edu3_robot"


def _load_by_path(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_by_path("measured_friction_actuator", _ASSET_ROBOT / "measured_friction_actuator.py")
_load_by_path("edu3_legacy_friction_gate", _ASSET_ROBOT / "edu3_legacy_friction_gate.py")
_cfg_mod = _load_by_path("edu3_nqj13_trainable_cfg", _ASSET_ROBOT / "edu3_nqj13_trainable_cfg.py")

EDU3_NQJ13_TRAINABLE_CFG = _cfg_mod.EDU3_NQJ13_TRAINABLE_CFG
CONTACT_MATERIAL_CFG = _cfg_mod.CONTACT_MATERIAL_CFG
set_legacy_joint_friction_checked = sys.modules["edu3_legacy_friction_gate"].set_legacy_joint_friction_checked
