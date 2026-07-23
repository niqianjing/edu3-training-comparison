"""Isolated bridge for the received Edu3-Flat package."""

from __future__ import annotations

import runpy
import sys


EXTERNAL_ROOT = "/home/zero/external_edu3_flat_20260721/integration"
TRAIN_ENTRY = "/home/zero/atom01_train/robolab/scripts/rsl_rl/train.py"

sys.path.insert(0, EXTERNAL_ROOT)
sys.path.insert(0, "/home/zero/atom01_train/robolab/scripts/rsl_rl")
import edu3_nqj13_flat_external  # noqa: F401, E402

sys.argv[0] = TRAIN_ENTRY
runpy.run_path(TRAIN_ENTRY, run_name="__main__")
