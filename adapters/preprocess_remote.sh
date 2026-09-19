#!/usr/bin/env bash
set -euo pipefail
pde=$1
case "$pde" in poisson|helmholtz) ;; *) exit 2;; esac
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for task_split in id smooth rough train validation; do
    while ! test -f "$task_out/data/compact/$pde/$task_split.npy"; do sleep 30; done
    task_name=${pde}_test_hf
    if test "$task_split" = train; then task_name=${pde}_hf; fi
    "$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/prepare_data.py" hf \
        --source "$task_out/data/compact/$pde" --split "$task_split" \
        --output "$task_out/data/preprocessing/$task_split/data/DiffPDE/$task_name" \
        --cache "$task_out/data/hf_cache/$pde/$task_split"
done
"$task_base/venv/bin/python" -u "$task_base/orchestration/adapters/prepare_data.py" diffusionpde \
    --source "$task_out/data/compact/$pde" --output "$task_out/data/diffusionpde/$pde"
mkdir -p "$task_out/data/surrogate_work/data/DiffPDE"
ln -s "$task_out/data/preprocessing/train/data/DiffPDE/${pde}_hf" \
    "$task_out/data/surrogate_work/data/DiffPDE/${pde}_hf"
ln -s "$task_out/data/preprocessing/validation/data/DiffPDE/${pde}_test_hf" \
    "$task_out/data/surrogate_work/data/DiffPDE/${pde}_test_hf"
