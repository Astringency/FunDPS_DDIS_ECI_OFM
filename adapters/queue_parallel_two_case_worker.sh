#!/usr/bin/env bash
set -u

previous="$1"
gpu="$2"
name="$3"
shift 3

base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
root=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919/evaluation_provisional/two_case_poisson_20260920
worker_dir="$root/workers/$name"
mkdir -p "$worker_dir"

while [[ ! -f "$root/workers/$previous/controller.exit" ]]; do
  sleep 30
done
previous_code="$(cat "$root/workers/$previous/controller.exit")"
if [[ "$previous_code" != 0 ]]; then
  printf '%s\n' "Previous GPU worker $previous failed with exit $previous_code" > "$worker_dir/controller.log"
  printf '%s\n' 1 > "$worker_dir/controller.exit"
  exit 1
fi

bash "$base/orchestration/adapters/parallel_two_case_worker.sh" "$gpu" "$name" "$@"
