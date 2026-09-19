#!/usr/bin/env bash
# File transport only. Does not implement or modify any model, training or sampler.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
task_name=helmholtz_10000-128-128_1.mat
task_sha=30d91fc73fcf4d5b86bbaa2385718a8ed1dde90a2bf96bcb7bac8681a270b0ba
task_source="staging/raw/helmholtz/$task_name"
task_chunks="staging/chunks/$task_name"
task_output=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_remote_chunks="$task_output/data/transfer_chunks/$task_name"
printf '%s  %s\n' "$task_sha" "$task_source" | sha256sum --check
mkdir -p "$task_chunks" provenance/chunk_logs
if [ ! -f "$task_chunks/split.complete" ]; then
    split -b 1M -d -a 6 "$task_source" "$task_chunks/part-"
    touch "$task_chunks/split.complete"
fi
ssh -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=15 server216 "mkdir -p '$task_remote_chunks'"
export task_remote_chunks
find "$task_chunks" -maxdepth 1 -type f -name 'part-*' -print0 | sort -z | xargs -0 -n 1 -P 3 bash -c '
    task_piece=$1
    task_piece_name=${task_piece##*/}
    for task_attempt in 1 2 3 4 5; do
        if rsync -a --partial --timeout=120 -e "ssh -o IPQoS=none -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=15" "$task_piece" "server216:$task_remote_chunks/" > "provenance/chunk_logs/$task_piece_name.log" 2>&1; then
            touch "provenance/chunk_logs/$task_piece_name.ok"
            exit 0
        fi
        sleep 3
    done
    exit 1
' bash
ssh -o ControlPath=none -o BatchMode=yes -o ConnectTimeout=15 server216 "set -e; cat '$task_remote_chunks'/part-* > '$task_output/data/raw/helmholtz/.$task_name.assembled'; echo '$task_sha  $task_output/data/raw/helmholtz/.$task_name.assembled' | sha256sum --check; mv '$task_output/data/raw/helmholtz/.$task_name.assembled' '$task_output/data/raw/helmholtz/$task_name'"
