#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zero/external_edu3_flat_20260721
WORK=/home/zero/atom01_train/robolab
PY=/home/zero/anaconda3/envs/isaaclab/bin/python
TRAIN="$ROOT/integration/train_external_copy.py"
GATE="$ROOT/provenance/runtime_readback_probe_final.json"
PY_PATH="$WORK/scripts/rsl_rl:$ROOT/integration:$WORK"
TASK=Edu3-Flat-External-20260721
EXPERIMENT=edu3_external_original_probe_20260721

LOG42=/home/zero/external_edu3_probe_seed42.log
LOG43=/home/zero/external_edu3_probe_seed43.log
PID42=/home/zero/external_edu3_probe_seed42.pid
PID43=/home/zero/external_edu3_probe_seed43.pid

python3 - "$GATE" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("status") != "PASS":
    raise SystemExit("runtime readback gate is not PASS")
print("RUNTIME_READBACK_GATE=PASS")
PY

for pid_file in "$PID42" "$PID43"; do
    if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "Refusing duplicate launch: live PID in $pid_file" >&2
        exit 3
    fi
done

cd "$WORK"

nohup env PYTHONPATH="$PY_PATH" "$PY" "$TRAIN" \
    --task "$TASK" \
    --num_envs 4096 \
    --max_iterations 1001 \
    --seed 42 \
    --device cuda:0 \
    --headless \
    --experiment_name "$EXPERIMENT" \
    --run_name EXTERNAL_ORIGINAL_PROBE_seed42 \
    --logger tensorboard \
    >"$LOG42" 2>&1 &
echo $! >"$PID42"

nohup env PYTHONPATH="$PY_PATH" "$PY" "$TRAIN" \
    --task "$TASK" \
    --num_envs 4096 \
    --max_iterations 1001 \
    --seed 43 \
    --device cuda:1 \
    --headless \
    --experiment_name "$EXPERIMENT" \
    --run_name EXTERNAL_ORIGINAL_PROBE_seed43 \
    --logger tensorboard \
    >"$LOG43" 2>&1 &
echo $! >"$PID43"

sleep 6
echo "seed42_pid=$(cat "$PID42")"
ps -p "$(cat "$PID42")" -o pid=,stat=,etime=,cmd=
echo "seed43_pid=$(cat "$PID43")"
ps -p "$(cat "$PID43")" -o pid=,stat=,etime=,cmd=
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
