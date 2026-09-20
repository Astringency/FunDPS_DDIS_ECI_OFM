#!/usr/bin/env bash
# Sample from a snapshot of a current validation-best prior without waiting for training to end.
# Usage: bash run_provisional_evaluation.sh PRIOR METHOD PDE SPLIT COUNT GPU OUTPUT_ROOT
set -euo pipefail

task_prior=$1 task_method=$2 task_pde=$3 task_split=$4 task_count=$5 task_gpu=$6 task_root=$7
case "$task_prior/$task_method" in
    ddis/ddis|fundps/fundps|fm4pde/eci|fm4pde/fm4pde|ofm/eci|ofm/ofm|ofm/fm4pde) ;;
    *) echo "Unsupported prior/method: $task_prior/$task_method" >&2; exit 2;;
esac
case "$task_pde" in poisson|helmholtz|darcy|nsnonbounded|burger) ;; *) exit 2;; esac
case "$task_split" in id|smooth|rough) ;; *) exit 2;; esac
case "$task_count" in 1|100) ;; *) echo 'COUNT must be 1 or 100' >&2; exit 2;; esac
case "$task_gpu" in 0|1|2|3|4|5|6|7) ;; *) exit 2;; esac
if test "$task_prior" = ddis || test "$task_prior" = fundps; then
    case "$task_pde" in poisson|helmholtz) ;; *) exit 2;; esac
fi

task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
case "$task_root" in "$task_out"/evaluation_provisional/*) ;; *) echo 'Use the main provisional result directory' >&2; exit 2;; esac
mkdir -p "$(dirname "$task_root")" "$task_out/locks"
mkdir "$task_root"
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$task_root/exit"' EXIT
printf '%s\n' 'Provisional result: current selected checkpoint; training may continue' > "$task_root/status"
printf '%s\n' "$task_prior $task_method $task_pde $task_split $task_count $task_gpu" > "$task_root/arguments.txt"
git -C "$task_base/orchestration" rev-parse HEAD > "$task_root/orchestration_revision"

# Multiple provisional runs on the same GPU must serialize. The 40 GiB free
# requirement retains headroom even when an existing trainer owns that GPU.
exec 9>"$task_out/locks/provisional_gpu_${task_gpu}.lock"
flock -n 9 || { echo 'Another provisional evaluation is using this GPU' >&2; exit 75; }
task_free=$(nvidia-smi -i "$task_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
test "$task_free" -ge 40000 || { echo "GPU $task_gpu has only $task_free MiB free" >&2; exit 75; }
awk '/MemAvailable:/ {exit !($2 > 64000 * 1024)}' /proc/meminfo
awk '{exit !($1 < 110)}' /proc/loadavg
nvidia-smi > "$task_root/resources_at_start.txt"
free -h >> "$task_root/resources_at_start.txt"
export CUDA_VISIBLE_DEVICES=$task_gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1

task_extra=()
if test "$task_prior" = ddis || test "$task_prior" = fundps; then
    task_selection=$task_out/jobs/${task_method}_${task_pde}/early_stopping
    task_checkpoint=$(readlink -f "$task_selection/best_checkpoint")
    cp "$task_selection/best.json" "$task_root/prior_selection.json"
    task_repo=DDIS
    if test "$task_method" = fundps; then task_repo=FunDPS; fi
    if test "$task_method" = ddis; then
        task_surrogate=$task_out/jobs/surrogate_${task_pde}/early_stopping
        task_extra=(--surrogate "$(readlink -f "$task_surrogate/best_checkpoint")")
        cp "$task_surrogate/best.json" "$task_root/surrogate_selection.json"
    fi
    task_command=("$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/evaluate_diffusion.py"
        --method "$task_method" --repo "$task_base/official/$task_repo"
        --config "$task_base/orchestration/configs/evaluation/${task_method}_${task_pde}_${task_split}.yaml"
        --checkpoint "$task_checkpoint" "${task_extra[@]}" --source "$task_out/data/compact/$task_pde"
        --split "$task_split" --count "$task_count" --output "$task_root/evaluation")
    task_verifier=("$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/validate_evaluation.py"
        --source "$task_out/data/compact/$task_pde" --split "$task_split"
        --count "$task_count" --output "$task_root/evaluation")
else
    task_assets=$task_out/data/shared_prior_assets
    test -f "$task_assets/verified.json"
    if test "$task_prior" = ofm; then
        task_selection=$task_out/jobs/flow_${task_pde}/early_stopping
        task_checkpoint=$(readlink -f "$task_selection/best_checkpoint")
        cp "$task_selection/best.json" "$task_root/prior_selection.json"
    else
        task_checkpoint=$task_assets/weights/$task_pde.pth
        cp "$task_assets/verified.json" "$task_root/assets_verification.json"
    fi
    task_command=("$task_base/venv-shared-prior/bin/python" -u
        "$task_base/orchestration/adapters/evaluate_shared_prior.py"
        --prior "$task_prior" --method "$task_method" --pde "$task_pde" --split "$task_split"
        --checkpoint "$task_checkpoint" --assets "$task_assets" --source "$task_out/data/compact/$task_pde"
        --fm4pde "$task_base/official/FM4PDE-cbe627c" --ofm "$task_base/official/OFM"
        --eci "$task_base/official/ECI" --count "$task_count" --output "$task_root/evaluation")
    task_verifier=("$task_base/venv-shared-prior/bin/python" -u
        "$task_base/orchestration/adapters/validate_shared_prior.py" --output "$task_root/evaluation")
fi
test -s "$task_checkpoint"
if test -f "$task_root/prior_selection.json"; then
    "$task_base/venv/bin/python" - "$task_root/prior_selection.json" "$task_checkpoint" <<'PY'
import json
from pathlib import Path
import sys
selected = json.loads(Path(sys.argv[1]).read_text())
assert Path(selected['checkpoint']).resolve(strict=True) == Path(sys.argv[2]).resolve(strict=True)
PY
fi
if test -f "$task_root/surrogate_selection.json"; then
    "$task_base/venv/bin/python" - "$task_root/surrogate_selection.json" "${task_extra[1]}" <<'PY'
import json
from pathlib import Path
import sys
selected = json.loads(Path(sys.argv[1]).read_text())
assert Path(selected['checkpoint']).resolve(strict=True) == Path(sys.argv[2]).resolve(strict=True)
PY
fi
printf '%s\n' "$task_checkpoint" > "$task_root/checkpoint_path.txt"
sha256sum "$task_checkpoint" > "$task_root/checkpoint.sha256"
if test "${#task_extra[@]}" -gt 0; then
    sha256sum "${task_extra[1]}" > "$task_root/surrogate.sha256"
fi
if test "$task_count" = 1; then task_command+=(--profile); fi
printf '%q ' "${task_command[@]}" > "$task_root/sampling_command.sh"
printf '\n' >> "$task_root/sampling_command.sh"
printf 'Sampling %s cases from %s on GPU %s\n' "$task_count" "$task_checkpoint" "$task_gpu" | tee "$task_root/status"
"${task_command[@]}" > "$task_root/sampling.log" 2>&1
"${task_verifier[@]}" > "$task_root/verification.json"
printf '%s\n' 'Provisional sampling completed and independently verified' | tee "$task_root/status"
