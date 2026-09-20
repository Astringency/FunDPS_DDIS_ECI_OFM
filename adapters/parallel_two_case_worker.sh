#!/usr/bin/env bash
set -u

gpu="$1"
name="$2"
shift 2

base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
root=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919/evaluation_provisional/two_case_poisson_20260920
worker_dir="$root/workers/$name"
mkdir -p "$worker_dir"

"$base/venv/bin/python" -u "$base/orchestration/adapters/run_two_case_poisson.py" worker \
  --root "$root" --gpu "$gpu" --worker-name "$name" \
  --methods ddis fundps ofm eci_fm eci_ofm fm_fm fm_ofm --cells "$@" \
  > "$worker_dir/controller.log" 2>&1
code="$?"
printf '%s\n' "$code" > "$worker_dir/controller.exit"
exit "$code"
