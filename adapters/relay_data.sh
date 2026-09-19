#!/usr/bin/env bash
# Stream already verified exports through the workstation without staging a
# complete multi-GB file. A partial file is never published before SHA256 agrees.
set -euo pipefail
pde=$1
task_local=/home/tat512/C01Python/DDIS_comparison_20260919
task_source=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ddis_comparison_20260919/export/$pde
task_target=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919/data/compact/$pde
task_socket=/tmp/ddis216-recovered-20260919
case "$pde" in poisson|helmholtz) ;; *) exit 2;; esac
mkdir -p "$task_local/staging/compact/$pde"
rsync -a --exclude=train.npy --exclude=validation.npy --exclude='.*' \
    "server197:$task_source/" "$task_local/staging/compact/$pde/"
ssh -S "$task_socket" server216 "mkdir -p '$task_target'"
rsync -a --partial-dir=.rsync-partial -e "ssh -S $task_socket" \
    "$task_local/staging/compact/$pde/" "server216:$task_target/"
for task_split in train validation; do
    task_hash=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["outputs"][sys.argv[2]]["sha256"])' \
        "$task_local/staging/compact/$pde/manifest.json" "$task_split")
    task_final="$task_target/$task_split.npy"
    task_partial="$task_target/.$task_split.npy.partial"
    if ssh -S "$task_socket" server216 "test -f '$task_final'"; then
        ssh -S "$task_socket" server216 "printf '%s  %s\n' '$task_hash' '$task_final' | sha256sum --check"
        continue
    fi
    task_bytes=$(ssh -S "$task_socket" server216 "if test -f '$task_partial'; then stat -c %s '$task_partial'; else echo 0; fi")
    task_start=$((task_bytes + 1))
    echo "$(date -Iseconds) $pde/$task_split: streaming from byte $task_start"
    ssh server197 "tail -c +$task_start '$task_source/$task_split.npy'" |
        ssh -S "$task_socket" server216 "cat >> '$task_partial'"
    ssh -S "$task_socket" server216 \
        "printf '%s  %s\n' '$task_hash' '$task_partial' | sha256sum --check && mv '$task_partial' '$task_final'"
    echo "$(date -Iseconds) $pde/$task_split: SHA256 verified"
done
