#!/usr/bin/env bash
set -euo pipefail
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_state=$task_out/jobs/plateau_priority
test "$(cat "$task_out/setup/plateau-priority-tests.exit")" = 0
mkdir -p "$task_state"
test ! -e "$task_state/started"
date -Iseconds > "$task_state/started"
git -C "$task_base/orchestration" rev-parse HEAD > "$task_state/orchestration_revision"
cp "$task_base/orchestration/configs/plateau_priority.json" "$task_state/policy.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set +e
"$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/plateau_priority.py" \
    --root "$task_out" --policy "$task_state/policy.json" \
    --orchestration "$task_base/orchestration" > "$task_out/jobs/plateau_priority.log" 2>&1
task_code=$?
printf '%s\n' "$task_code" > "$task_out/jobs/plateau_priority.exit"
exit "$task_code"
