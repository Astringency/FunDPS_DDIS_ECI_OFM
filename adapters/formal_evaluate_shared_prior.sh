#!/usr/bin/env bash
set -euo pipefail
task_prior=$1 task_method=$2 task_pde=$3 task_split=$4
case "$task_prior/$task_method" in fm4pde/eci|fm4pde/fm4pde|ofm/eci|ofm/ofm|ofm/fm4pde) ;; *) exit 2;; esac
case "$task_pde" in poisson|helmholtz|darcy|nsnonbounded|burger) ;; *) exit 2;; esac
case "$task_split" in id|smooth|rough) ;; *) exit 2;; esac
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_state=$task_out/jobs/shared_${task_prior}_${task_method}_${task_pde}_${task_split}
task_assets=$task_out/data/shared_prior_assets
task_result=$task_out/evaluation_shared_prior/$task_prior/$task_method/$task_pde
task_python=$task_base/venv-shared-prior/bin/python
mkdir -p "$task_state" "$task_result"
echo 'Waiting for runtime integration checks and verified input assets' > "$task_state/status"
while ! test -f "$task_out/setup/shared-prior-integration-ready.json" || ! test -f "$task_assets/verified.json"; do sleep 30; done
if test "$task_prior" = ofm; then
    task_dependency=$task_out/jobs/flow_$task_pde
    echo 'Waiting for OFM training and validation-selected checkpoint' > "$task_state/status"
    while ! test -f "$task_dependency/training_completed"; do
        test ! -f "$task_dependency/early_stopping/failure.json"
        sleep 30
    done
    task_checkpoint=$(readlink -f "$task_dependency/early_stopping/best_checkpoint")
    cp "$task_dependency/early_stopping/best.json" "$task_state/prior_selection.json"
else
    task_checkpoint=$task_assets/weights/$task_pde.pth
fi
test -f "$task_checkpoint"
# Newly queued OFM training gets first access to any released experiment slot.
while true; do
    task_pending=0
    for task_training_pde in poisson helmholtz darcy nsnonbounded burger; do
        task_training=$task_out/jobs/flow_$task_training_pde
        if ! test -f "$task_training/started" && ! test -f "$task_training/training_completed"; then task_pending=1; fi
    done
    if test "$task_pending" = 0; then break; fi
    echo 'Waiting for the priority OFM training queues to acquire GPUs' > "$task_state/status"
    sleep 30
done
while true; do
    for task_gpu in 0 1 5 7 4 6 2 3; do
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
    echo 'Waiting for a sampling GPU with at least 40,000 MiB free' > "$task_state/status"
    sleep 30
done
test ! -e "$task_state/started"
date -Iseconds > "$task_state/started"
echo "$task_gpu" > "$task_state/gpu_index"
git -C "$task_base/orchestration" rev-parse HEAD > "$task_state/orchestration_revision"
nvidia-smi > "$task_state/resources_at_start.txt"
free -h >> "$task_state/resources_at_start.txt"
export CUDA_VISIBLE_DEVICES=$task_gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
task_command=("$task_python" -u "$task_base/orchestration/adapters/evaluate_shared_prior.py"
    --prior "$task_prior" --method "$task_method" --pde "$task_pde" --split "$task_split"
    --checkpoint "$task_checkpoint" --assets "$task_assets" --source "$task_out/data/compact/$task_pde"
    --fm4pde "$task_base/official/FM4PDE-cbe627c" --ofm "$task_base/official/OFM" --eci "$task_base/official/ECI")
task_preflight=$task_result/preflight_$task_split
echo 'Profiling one case with the final selected weights and full sampling schedule' > "$task_state/status"
"${task_command[@]}" --profile --count 1 --output "$task_preflight" > "$task_state/preflight.log" 2>&1
"$task_python" "$task_base/orchestration/adapters/validate_shared_prior.py" --output "$task_preflight" > "$task_state/preflight_verification.json"
"$task_python" - "$task_preflight" "$task_free" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); summary=json.loads((p/'verified_summary.json').read_text())
assert summary['profile_only'] and summary['cases']==1 and summary['successes']==1
peak=max(json.loads((p/name).read_text())['peak_reserved_bytes'] for name in ('model_loaded.json','case_000.json'))
assert peak < (int(sys.argv[2])-4000)*1024**2
PY
echo 'Evaluating the 100 common cases' > "$task_state/status"
"${task_command[@]}" --count 100 --output "$task_result/$task_split" > "$task_state/sampling.log" 2>&1
"$task_python" "$task_base/orchestration/adapters/validate_shared_prior.py" --output "$task_result/$task_split" > "$task_state/verification.json"
echo 'All 100 cases evaluated and independently verified' > "$task_state/status"
date -Iseconds > "$task_state/evaluation_completed"
