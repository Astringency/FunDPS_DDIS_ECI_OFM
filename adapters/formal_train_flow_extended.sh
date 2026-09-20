#!/usr/bin/env bash
set -euo pipefail
task_pde=$1
case "$task_pde" in darcy|nsnonbounded|burger) ;; *) exit 2;; esac
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_state=$task_out/jobs/flow_$task_pde
task_data=$task_out/data/compact/$task_pde
mkdir -p "$task_state" "$task_out/locks"
echo 'Waiting for checksum-verified training and validation data' > "$task_state/status"
while ! test -f "$task_data/verified.json"; do sleep 30; done
while true; do
    for task_gpu in 0 1 4 6 2 3 5 7; do
        task_free=$(nvidia-smi -i "$task_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
        if test "$task_free" -lt 40000; then continue; fi
        if ! awk '/MemAvailable:/ {exit !($2 > 64000 * 1024)}' /proc/meminfo; then continue; fi
        if ! awk '{exit !($1 < 110)}' /proc/loadavg; then continue; fi
        exec 9>"$task_out/locks/gpu_$task_gpu.lock"
        if flock -n 9; then
            task_free=$(nvidia-smi -i "$task_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
            if test "$task_free" -ge 40000; then break 2; fi
            flock -u 9
        fi
    done
    echo 'Waiting for a training GPU with at least 40,000 MiB free' > "$task_state/status"
    sleep 30
done
test ! -e "$task_state/started"
date -Iseconds > "$task_state/started"
echo "$task_gpu" > "$task_state/gpu_index"
git -C "$task_base/orchestration" rev-parse HEAD > "$task_state/orchestration_revision"
git -C "$task_base/official/OFM" rev-parse HEAD > "$task_state/official_revision"
cp "$task_data/verified.json" "$task_state/data_verification.json"
nvidia-smi > "$task_state/resources_at_start.txt"
free -h >> "$task_state/resources_at_start.txt"
export CUDA_VISIBLE_DEVICES=$task_gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
for task_batch in 1 100; do
    task_profile=$task_out/profiles/flow_${task_pde}_b${task_batch}
    echo "Profiling official OFM on $task_pde, batch $task_batch" > "$task_state/status"
    "$task_base/venv-flow/bin/python" -u "$task_base/orchestration/adapters/train_flow.py" \
        --ofm "$task_base/official/OFM" --train "$task_data/train.npy" \
        --output "$task_profile" --batch "$task_batch" --epochs 1 \
        --profile --limit "$((task_batch * 2))" > "$task_profile.log" 2>&1
    "$task_base/venv/bin/python" - "$task_profile" "$task_free" <<'PY'
from pathlib import Path
import json,math,re,sys
p=Path(sys.argv[1]); record=json.loads((p/'completed.json').read_text())
assert record['profile_only'] and record['completed_epochs']==1
assert record['peak_reserved_bytes'] < (int(sys.argv[2])-4000)*1024**2
losses=re.findall(r'tr @ epoch \d+/\d+ \| Loss (\S+)',p.with_suffix('.log').read_text())
assert losses and all(math.isfinite(float(x)) for x in losses)
p.with_suffix('.exit').write_text('0\n')
PY
done
echo 'Training official OFM prior with early stopping' > "$task_state/status"
"$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/early_stop.py" \
    --method flow --policy "$task_base/orchestration/configs/early_stopping.json" \
    --state "$task_state/early_stopping" --training-root "$task_out/training/flow/$task_pde" \
    --repo "$task_base/official/OFM" --validation "$task_data/validation.npy" -- \
    "$task_base/venv-flow/bin/python" -u "$task_base/orchestration/adapters/train_flow.py" \
    --ofm "$task_base/official/OFM" --train "$task_data/train.npy" \
    --validation "$task_data/validation.npy" --output "$task_out/training/flow/$task_pde" \
    --batch 100 --epochs 300
echo 'Training completed with validation-selected checkpoint' > "$task_state/status"
date -Iseconds > "$task_state/training_completed"
