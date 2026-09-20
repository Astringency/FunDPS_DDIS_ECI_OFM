#!/usr/bin/env bash
# Run in a local task tmux; retain partial files for resumable transport.
set -euo pipefail
task_pde=$1
case "$task_pde" in darcy|nsnonbounded|burger) ;; *) exit 2;; esac
task_local=/home/tat512/C01Python/DDIS_comparison_20260919
task_source=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ddis_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_stage=$task_local/transfers/extended/$task_pde
mkdir -p "$task_stage"
while ! ssh -o BatchMode=yes -o ConnectTimeout=10 server197 "test -f $task_source/export/$task_pde/manifest.json"; do
    task_exit=$(ssh -o BatchMode=yes -o ConnectTimeout=10 server197 "cat $task_source/setup/export_flow_$task_pde.exit 2>/dev/null || true")
    if test -n "$task_exit" && test "$task_exit" != 0; then exit 1; fi
    sleep 30
done
printf '%s\n' train.npy validation.npy id.npy smooth.npy rough.npy \
    train_ids.npy validation_ids.npy id_ids.npy smooth_ids.npy rough_ids.npy \
    normalizer.pt normalization.json manifest.json > "$task_stage/files.txt"
rsync -a --partial --append-verify --files-from="$task_stage/files.txt" \
    "server197:$task_source/export/$task_pde/" "$task_stage/"
ssh -S /tmp/ddis216-recovered-20260919 server216 "mkdir -p $task_out/data/compact/$task_pde"
rsync -a --partial --append-verify --files-from="$task_stage/files.txt" \
    -e 'ssh -S /tmp/ddis216-recovered-20260919' "$task_stage/" \
    "server216:$task_out/data/compact/$task_pde/"
ssh -S /tmp/ddis216-recovered-20260919 server216 \
    "$task_base/venv/bin/python $task_base/orchestration/adapters/verify_extended_flow_data.py --source $task_out/data/compact/$task_pde"
