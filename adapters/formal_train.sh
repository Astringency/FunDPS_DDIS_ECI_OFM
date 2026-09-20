#!/usr/bin/env bash
# Queue an official training entry point behind verified data and memory gates.
set -euo pipefail
method=$1
pde=$2
task_gpu_options=$3
task_min_free=$4
case "$pde" in poisson|helmholtz|darcy|nsnonbounded) ;; *) exit 2;; esac
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
if test "$method" != flow && test -f "$task_out/jobs/diffusion_training_cancelled.json"; then
    echo 'DDIS/FunDPS and auxiliary training cancelled by user.' >&2
    exit 2
fi
task_state=$task_out/jobs/${method}_${pde}
mkdir -p "$task_state" "$task_out/locks"
task_training_meta=$task_out/data/preprocessing/train/data/DiffPDE/${pde}_hf/metadata.json
task_validation_meta=$task_out/data/preprocessing/validation/data/DiffPDE/${pde}_test_hf/metadata.json
case "$method" in
    ddis) task_probe=ddis_b32; task_repo=DDIS;;
    fundps) task_probe=fundps_b32; task_repo=FunDPS;;
    flow) task_probe=flow_b100; task_repo=OFM;;
    surrogate) task_probe=surrogate_b32; task_repo=DDIS;;
    *) exit 2;;
esac
test "$(cat "$task_out/profiles/$task_probe.exit")" = 0
echo "$(date -Iseconds) Waiting for verified $pde data" > "$task_state/status"
while ! test -f "$task_training_meta"; do sleep 30; done
while ! test -f "$task_validation_meta"; do sleep 30; done
if test "$method" = surrogate; then
    while ! test -L "$task_out/data/surrogate_work/data/DiffPDE/${pde}_test_hf"; do sleep 30; done
fi
# New non-OFM queues must yield to all five OFM training tasks.
if test "$method" != flow; then
    for task_priority_pde in poisson helmholtz darcy nsnonbounded burger; do
        while ! test -f "$task_out/jobs/flow_$task_priority_pde/training_completed"; do
            echo 'Waiting for priority OFM training completion' > "$task_state/status"
            sleep 30
        done
    done
fi
# Locks serialize this experiment's jobs on each GPU. A candidate list allows
# another safe slot when the preferred GPU is occupied by unrelated jobs.
IFS=',' read -ra task_candidates <<< "$task_gpu_options"
while true; do
    task_ram=$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    for task_candidate in "${task_candidates[@]}"; do
        case "$task_candidate" in [0-7]) ;; *) exit 2;; esac
        task_free=$(nvidia-smi -i "$task_candidate" --query-gpu=memory.free --format=csv,noheader,nounits)
        if test "$task_free" -lt "$task_min_free" || test "$task_ram" -lt 64000; then continue; fi
        if ! awk '{exit !($1 < 110)}' /proc/loadavg; then continue; fi
        exec 9>"$task_out/locks/gpu_${task_candidate}.lock"
        if flock -n 9; then
            task_free=$(nvidia-smi -i "$task_candidate" --query-gpu=memory.free --format=csv,noheader,nounits)
            if test "$task_free" -ge "$task_min_free"; then
                task_gpu=$task_candidate
                break 2
            fi
            flock -u 9
        fi
    done
    echo "$(date -Iseconds) Waiting for an available GPU in [$task_gpu_options] with ${task_min_free}MiB free" > "$task_state/status"
    sleep 30
done
test ! -e "$task_state/started"
date -Iseconds > "$task_state/started"
echo "$task_gpu" > "$task_state/gpu_index"
git -C "$task_base/orchestration" rev-parse HEAD > "$task_state/orchestration_revision"
git -C "$task_base/official/$task_repo" rev-parse HEAD > "$task_state/official_revision"
nvidia-smi > "$task_state/resources_at_start.txt"
free -h >> "$task_state/resources_at_start.txt"
export CUDA_VISIBLE_DEVICES=$task_gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WANDB_MODE=offline WANDB_PROJECT=fm4pde_official_comparison
export WANDB_DIR=$task_state MPLBACKEND=Agg
export MASTER_PORT=$((29800 + task_gpu))
echo "$(date -Iseconds) Training" > "$task_state/status"
cd "$task_base/official/$task_repo"
task_training_root=$task_out/training/$method/$pde
case "$method" in
    ddis|fundps)
        task_config=$task_base/orchestration/configs/training/${method}_${pde}.yaml
        cp "$task_config" "$task_state/config.yaml"
        task_script=train.py
        if test "$method" = ddis; then task_script=scripts/train/train.py; fi
        task_command=("$task_base/venv/bin/python" -u "$task_script" -c "$task_state/config.yaml")
        ;;
    flow)
        task_command=("$task_base/venv-flow/bin/python" -u "$task_base/orchestration/adapters/train_flow.py" \
            --ofm "$task_base/official/OFM" --train "$task_out/data/compact/$pde/train.npy" \
            --validation "$task_out/data/compact/$pde/validation.npy" \
            --output "$task_out/training/flow/$pde" --batch 100 --epochs 300)
        ;;
    surrogate)
        task_config=$task_base/orchestration/configs/training/ddis_surrogate_${pde}.yaml
        cp "$task_config" "$task_state/config.yaml"
        task_training_root=$task_out/training/ddis_surrogate/$pde
        task_command=("$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/train_surrogate.py" \
            --repo "$task_base/official/DDIS" --config "$task_state/config.yaml" \
            --workdir "$task_out/data/surrogate_work" --seed 0)
        ;;
esac
"$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/early_stop.py" \
    --method "$method" --policy "$task_base/orchestration/configs/early_stopping_v2.json" \
    --state "$task_state/early_stopping" --training-root "$task_training_root" \
    --repo "$task_base/official/$task_repo" \
    --validation "$task_out/data/preprocessing/validation/data/DiffPDE/${pde}_test_hf" \
    -- "${task_command[@]}"
echo "$(date -Iseconds) Training completed with validation-selected checkpoint; evaluation pending" > "$task_state/status"
date -Iseconds > "$task_state/training_completed"
