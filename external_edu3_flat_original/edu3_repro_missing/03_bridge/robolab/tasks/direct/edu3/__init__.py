"""Bridge: register Edu3-Flat from the edu3_nqj13 asset package.

Put the asset package on ``sys.path`` and import it so
``train.py --task=Edu3-Flat`` works inside robolab.

Path resolution order:
1. env ``EDU3_ASSET_ROOT`` if set
2. sibling layout: ``<bundle>/01_asset_task/edu3_nqj13_trainable_fullbody_v1``
   when this bridge lives under a repro bundle
3. fallback absolute path used on the original training machine (edit if needed)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()

def _resolve_asset_root() -> Path:
    env = os.environ.get("EDU3_ASSET_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # edu3_repro_missing/03_bridge/robolab/tasks/direct/edu3/__init__.py
    # -> bundle root = parents[5]
    try:
        bundle_candidate = _HERE.parents[5] / "01_asset_task" / "edu3_nqj13_trainable_fullbody_v1"
        if bundle_candidate.is_dir():
            return bundle_candidate
    except IndexError:
        pass
    # Common: asset package next to robolab checkout
    for cand in (
        Path("/home/joyin/edu3_nqj13_trainable_fullbody_v1_FINAL/edu3_nqj13_trainable_fullbody_v1"),
        Path.cwd() / "edu3_nqj13_trainable_fullbody_v1",
    ):
        if cand.is_dir():
            return cand
    raise ImportError(
        "EDU3 asset package not found. Set env EDU3_ASSET_ROOT to the "
        "edu3_nqj13_trainable_fullbody_v1 directory (contains edu3_nqj13_flat/)."
    )

_ASSET_ROOT = _resolve_asset_root()
_asset = str(_ASSET_ROOT)
if _asset not in sys.path:
    sys.path.insert(0, _asset)

import edu3_nqj13_flat  # noqa: E402,F401  — registers gym id Edu3-Flat
