#!/usr/bin/env bash
set -euo pipefail
task_local=/home/tat512/C01Python/DDIS_comparison_20260919
task_source=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ddis_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_stage=$task_local/transfers/shared_prior_assets
mkdir -p "$task_stage"
while ! ssh -o BatchMode=yes -o ConnectTimeout=10 server197 "test -f $task_source/shared_prior_assets/manifest.json"; do
    task_exit=$(ssh -o BatchMode=yes server197 "cat $task_source/setup/shared_prior_assets.exit 2>/dev/null || true")
    if test -n "$task_exit" && test "$task_exit" != 0; then exit 1; fi
    sleep 30
done
rsync -aL --partial --append-verify "server197:$task_source/shared_prior_assets/" "$task_stage/"
ssh -S /tmp/ddis216-recovered-20260919 server216 "mkdir -p $task_out/data/shared_prior_assets"
rsync -a --partial --append-verify -e 'ssh -S /tmp/ddis216-recovered-20260919' \
    "$task_stage/" "server216:$task_out/data/shared_prior_assets/"
