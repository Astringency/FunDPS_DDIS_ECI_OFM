#!/usr/bin/env bash
# Provisional/resource weights only: compatibility profiles, never formal results.
set -euo pipefail
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_state=$task_out/jobs/profile_shared_prior_adapters
mkdir -p "$task_state"
test "$(cat "$task_out/setup/shared-prior-input-checks.exit")" = 0
test ! -e "$task_out/data/compact/darcy/verified.json"
exec 9>"$task_out/locks/gpu_1.lock"
flock -n 9
task_free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
test "$task_free" -ge 40000
awk '/MemAvailable:/ {exit !($2 > 64000 * 1024)}' /proc/meminfo
awk '{exit !($1 < 110)}' /proc/loadavg
nvidia-smi > "$task_state/resources_at_start.txt"
git -C "$task_base/orchestration" rev-parse HEAD > "$task_state/orchestration_revision"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
for task_spec in fm4pde:poisson fm4pde:burger eci:burger ofm:burger; do
    task_method=${task_spec%%:*}
    task_pde=${task_spec##*:}
    if test "$task_pde" = poisson; then
        task_checkpoint=$(readlink -f "$task_out/jobs/flow_poisson/early_stopping/best_checkpoint")
    else
        task_checkpoint=$task_out/profiles/flow_burger_b100/epoch_1.pt
    fi
    task_destination=$task_out/profiles/shared_ofm_${task_method}_${task_pde}_adapter
    echo "Profiling $task_spec with provisional/resource weights" > "$task_state/status"
    "$task_base/venv-shared-prior/bin/python" -u "$task_base/orchestration/adapters/evaluate_shared_prior.py" \
        --prior ofm --method "$task_method" --pde "$task_pde" --split id --profile --count 1 \
        --checkpoint "$task_checkpoint" --assets "$task_out/data/shared_prior_assets" \
        --source "$task_out/data/compact/$task_pde" --fm4pde "$task_base/official/FM4PDE-cbe627c" \
        --ofm "$task_base/official/OFM" --eci "$task_base/official/ECI" \
        --output "$task_destination" > "$task_destination.log" 2>&1
    "$task_base/venv-shared-prior/bin/python" "$task_base/orchestration/adapters/validate_shared_prior.py" \
        --output "$task_destination" > "$task_destination.verification.json"
    printf '0\n' > "$task_destination.exit"
done
echo 'Compatibility profiles completed; formal preflights still required with final selected weights' > "$task_state/status"
