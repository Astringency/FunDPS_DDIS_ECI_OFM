#!/usr/bin/env bash
# Wait for validation-selected official weights, then evaluate the common cases.
set -euo pipefail
method=$1
pde=$2
case "$method" in ddis|fundps|ofm|eci) ;; *) exit 2;; esac
case "$pde" in poisson|helmholtz) ;; *) exit 2;; esac
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_state=$task_out/jobs/evaluate_${method}_${pde}
task_result=$task_out/evaluation/$method/$pde
mkdir -p "$task_state" "$task_result"
test "$(cat "$task_out/profiles/${method}_sampling_full.exit")" = 0
test -f "$task_out/profiles/${method}_sampling_full/verified_summary.json"

wait_training() {
    local task_dependency=$task_out/jobs/${1}_${pde}
    while ! test -f "$task_dependency/training_completed"; do
        if test -f "$task_dependency/early_stopping/failure.json"; then
            echo "Training dependency failed: $task_dependency" >&2
            exit 1
        fi
        echo "$(date -Iseconds) Waiting for $1 $pde training and checkpoint selection" > "$task_state/status"
        sleep 30
    done
    test -f "$task_dependency/early_stopping/best_checkpoint"
}

task_training_method=$method
if test "$method" = ofm || test "$method" = eci; then task_training_method=flow; fi
wait_training "$task_training_method"
task_checkpoint=$(readlink -f "$task_out/jobs/${task_training_method}_${pde}/early_stopping/best_checkpoint")
cp "$task_out/jobs/${task_training_method}_${pde}/early_stopping/best.json" "$task_result/prior_selection.json"
task_surrogate=
if test "$method" = ddis; then
    wait_training surrogate
    task_surrogate=$(readlink -f "$task_out/jobs/surrogate_${pde}/early_stopping/best_checkpoint")
    cp "$task_out/jobs/surrogate_${pde}/early_stopping/best.json" "$task_result/surrogate_selection.json"
fi

# Prioritize a training job that has not acquired its advertised GPU yet.
reserved_for_training() {
    local task_gpu_to_check=$1 task_dir task_options
    for task_dir in "$task_out"/jobs/{ddis,fundps,flow,surrogate}_{poisson,helmholtz}; do
        if test -f "$task_dir/training_completed" || test -f "$task_dir/early_stopping/failure.json"; then continue; fi
        if test -f "$task_dir/started"; then
            task_options=$(cat "$task_dir/gpu_index")
        else
            task_options=$(sed -n 's/.*gpu_candidates=\([^ ]*\).*/\1/p' "$task_dir/queue_arguments.txt")
        fi
        case ",$task_options," in *",$task_gpu_to_check,"*) return 0;; esac
    done
    return 1
}
task_min_free=20000
if test "$method" = ofm; then task_min_free=36000; fi
while true; do
    for task_candidate in 4 6 5 7 2 3 0 1; do
        if reserved_for_training "$task_candidate"; then continue; fi
        task_free=$(nvidia-smi -i "$task_candidate" --query-gpu=memory.free --format=csv,noheader,nounits)
        if test "$task_free" -lt "$task_min_free"; then continue; fi
        if ! awk '/MemAvailable:/ {exit !($2 > 16000 * 1024)}' /proc/meminfo; then continue; fi
        if ! awk '{exit !($1 < 110)}' /proc/loadavg; then continue; fi
        exec 9>"$task_out/locks/gpu_${task_candidate}.lock"
        if flock -n 9; then
            task_free=$(nvidia-smi -i "$task_candidate" --query-gpu=memory.free --format=csv,noheader,nounits)
            if test "$task_free" -ge "$task_min_free"; then task_gpu=$task_candidate; break 2; fi
            flock -u 9
        fi
    done
    echo "$(date -Iseconds) Waiting for sampling GPU with ${task_min_free}MiB free" > "$task_state/status"
    sleep 30
done
test ! -f "$task_state/started"
date -Iseconds > "$task_state/started"
echo "$task_gpu" > "$task_state/gpu_index"
nvidia-smi > "$task_state/resources_at_start.txt"
free -h >> "$task_state/resources_at_start.txt"
git -C "$task_base/orchestration" rev-parse HEAD > "$task_state/orchestration_revision"
export CUDA_VISIBLE_DEVICES=$task_gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1

run_sampler() {
    local task_split=$1 task_count=$2 task_destination=$3
    if test "$method" = ddis || test "$method" = fundps; then
        local task_repo=DDIS
        if test "$method" = fundps; then task_repo=FunDPS; fi
        local task_extra=()
        if test "$method" = ddis; then task_extra=(--surrogate "$task_surrogate"); fi
        "$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/evaluate_diffusion.py" \
            --method "$method" --repo "$task_base/official/$task_repo" \
            --config "$task_base/orchestration/configs/evaluation/${method}_${pde}_${task_split}.yaml" \
            --checkpoint "$task_checkpoint" "${task_extra[@]}" --source "$task_out/data/compact/$pde" \
            --split "$task_split" --count "$task_count" --output "$task_destination"
    else
        "$task_base/venv-flow/bin/python" -u "$task_base/orchestration/adapters/evaluate_flow.py" \
            --method "$method" --ofm "$task_base/official/OFM" --eci "$task_base/official/ECI" \
            --checkpoint "$task_checkpoint" --source "$task_out/data/compact/$pde" \
            --split "$task_split" --count "$task_count" --output "$task_destination"
    fi
    "$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/validate_evaluation.py" \
        --source "$task_out/data/compact/$pde" --split "$task_split" \
        --count "$task_count" --output "$task_destination"
}

echo "$(date -Iseconds) Checking full sampler with selected trained weights" > "$task_state/status"
run_sampler id 1 "$task_result/preflight" > "$task_state/preflight.log" 2>&1
for task_split in id smooth rough; do
    echo "$(date -Iseconds) Evaluating $task_split first 100" > "$task_state/status"
    run_sampler "$task_split" 100 "$task_result/$task_split" > "$task_state/${task_split}.log" 2>&1
done
echo "$(date -Iseconds) All 300 cases evaluated and output records verified" > "$task_state/status"
date -Iseconds > "$task_state/evaluation_completed"
