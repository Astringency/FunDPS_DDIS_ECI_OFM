#!/usr/bin/env bash
# Isolated Python >=3.12 runtime for the unchanged current FM4PDE source.
set -euo pipefail
task_base=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
task_out=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
task_env=$task_base/venv-shared-prior
test ! -e "$task_env"
/data1/zjinzxf2025/miniconda3/envs/fm4pde/bin/python -m venv --system-site-packages "$task_env"
# These official packages expose optional dependencies not used by this task.
# Install their exact sources without downgrading the inherited Torch/NumPy;
# the exercised FNO, GP and exact-OT paths are checked below and in GPU probes.
"$task_env/bin/python" -m pip install --no-deps neuraloperator==0.3.0 torch-harmonics==0.7.2 torchcfm==1.0.5
"$task_env/bin/python" -m pip install gpytorch==1.13 tensorly==0.9.0 tensorly-torch==0.5.0 \
    opt-einsum==3.4.0 configmypy==0.2.0 pot==0.9.6.post1 torchdiffeq==0.2.4 wandb==0.19.8
"$task_env/bin/python" -m pip freeze > "$task_out/setup/shared-prior-runtime-freeze.txt"
"$task_env/bin/python" - "$task_base" <<'PY'
import importlib,sys,torch
from pathlib import Path
base=Path(sys.argv[1])
sys.path.insert(0,str(base/'official/FM4PDE-cbe627c'))
import sampling.runner
sys.path.insert(0,str(base/'orchestration/adapters'))
from train_flow import official_ofm
FNO,OFM=official_ofm(base/'official/OFM')
from torchcfm.optimal_transport import OTPlanSampler
sampler=OTPlanSampler(method='exact')
a,b=sampler.sample_plan(torch.randn(2,1,4,4),torch.randn(2,1,4,4))
assert a.shape==b.shape==(2,1,4,4)
print('Official FM4PDE runner, OFM FNO/GP/OT imports verified',sys.version,torch.__version__)
PY
