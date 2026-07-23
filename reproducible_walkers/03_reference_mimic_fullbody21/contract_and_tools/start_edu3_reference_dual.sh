#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zero/edu3_reference_mimic_v1/project
PYTHON=/home/zero/anaconda3/envs/isaaclab/bin/python
TRAIN=mimic_real/scripts/train_edu3_reference.py

cd "$ROOT"

nohup env PYTHONPATH=. "$PYTHON" "$TRAIN" \
  --task edu3_reference_mimic_r1 \
  --num_envs 4096 \
  --seed 42 \
  --device cuda:0 \
  --headless \
  --max_iterations 1001 \
  --run_name EDU3_XIAOHAI_REF_25_10_V1_seed42 \
  > /home/zero/edu3_reference_mimic_seed42.log 2>&1 &
echo $! > /home/zero/edu3_reference_mimic_seed42.pid

nohup env PYTHONPATH=. "$PYTHON" "$TRAIN" \
  --task edu3_reference_mimic_r1 \
  --num_envs 4096 \
  --seed 43 \
  --device cuda:1 \
  --headless \
  --max_iterations 1001 \
  --run_name EDU3_XIAOHAI_REF_25_10_V1_seed43 \
  > /home/zero/edu3_reference_mimic_seed43.log 2>&1 &
echo $! > /home/zero/edu3_reference_mimic_seed43.pid

sleep 8
echo "seed42_pid=$(cat /home/zero/edu3_reference_mimic_seed42.pid)"
echo "seed43_pid=$(cat /home/zero/edu3_reference_mimic_seed43.pid)"
ps -p "$(cat /home/zero/edu3_reference_mimic_seed42.pid)" -o pid=,stat=,cmd=
ps -p "$(cat /home/zero/edu3_reference_mimic_seed43.pid)" -o pid=,stat=,cmd=
