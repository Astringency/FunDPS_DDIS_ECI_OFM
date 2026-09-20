"""Calibrate completed OFM priors without using a GPU assigned to OFM training."""
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
root = OUT / 'jobs/calibration_remaining_v2'
root.mkdir(exist_ok=True)
pending = ['helmholtz', 'burger', 'darcy', 'nsnonbounded']
while pending:
    for pde in list(pending):
        if not (OUT / 'jobs' / f'flow_{pde}' / 'training_completed').exists():
            continue
        busy = any((OUT / 'jobs' / f'flow_{q}' / 'gpu_index').exists()
                   and (OUT / 'jobs' / f'flow_{q}' / 'gpu_index').read_text().strip() == '5'
                   and not (OUT / 'jobs' / f'flow_{q}' / 'training_completed').exists()
                   for q in ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger'))
        if busy or os.getloadavg()[0] >= 110:
            continue
        if (root / f'{pde}.exit').exists():
            pending.remove(pde)
            continue
        cmd = [BASE / 'venv-shared-prior/bin/python', '-u', BASE / 'orchestration/adapters/calibrate_poisson_guidance.py',
               '--pde', pde, '--gpu', '5']
        with (root / f'{pde}.log').open('w') as log:
            code = subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT).returncode
        (root / f'{pde}.exit').write_text(str(code))
        print(json.dumps({'pde': pde, 'exit': code}), flush=True)
        pending.remove(pde)
    time.sleep(30)
