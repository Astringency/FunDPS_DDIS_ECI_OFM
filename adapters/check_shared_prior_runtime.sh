#!/usr/bin/env bash
set -euo pipefail
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
while ! test -f "$task_out/setup/shared-prior-runtime-retry.exit"; do sleep 15; done
test "$(cat "$task_out/setup/shared-prior-runtime-retry.exit")" = 0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export FM4PDE_SOURCE=$task_base/official/FM4PDE-cbe627c ECI_SOURCE=$task_base/official/ECI
"$task_base/venv-shared-prior/bin/python" "$task_base/orchestration/tests/test_shared_prior_runtime.py" -v
"$task_base/venv-shared-prior/bin/python" - "$task_out/setup/shared-prior-integration-ready.json" "$task_base/orchestration" <<'PY'
import datetime,json,subprocess,sys
from pathlib import Path
record={'completed':datetime.datetime.now().isoformat(), 'checks':4,
 'orchestration_revision':subprocess.check_output(['git','-C',sys.argv[2],'rev-parse','HEAD'],text=True).strip(),
 'scope':'CPU official-interface integration; each formal task additionally requires a successful full-schedule GPU preflight'}
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
PY
