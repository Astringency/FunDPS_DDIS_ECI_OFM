"""Evaluate each completed diffusion prior without a global training/evaluation barrier."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
ROOT = OUT / 'evaluation_v2/diffusion'
AD = BASE / 'orchestration/adapters'
PY = BASE / 'venv/bin/python'
p = argparse.ArgumentParser()
p.add_argument('--worker', type=int, required=True)
p.add_argument('--gpu', type=int, help='Restrict worker to a measured resource allocation.')
p.add_argument('--slot', type=int, default=0)
p.add_argument('--method', choices=['ddis', 'fundps'])
p.add_argument('--pde', choices=['poisson', 'helmholtz'])
a = p.parse_args()
state = OUT / 'jobs' / f'diffusion_evaluation_v2_{a.worker}'
state.mkdir(exist_ok=True)
def status(message): (state / 'status').write_text(message + '\n')
while True:
    chosen = None
    matrix = json.loads((BASE / 'orchestration/configs/evaluation_matrix_v2.json').read_text())['evaluations']
    for r in matrix:
        if r['method'] not in ('ddis', 'fundps') or r['pde'] == 'burger':
            continue
        if a.method and r['method'] != a.method: continue
        if a.pde and r['pde'] != a.pde: continue
        training = OUT / 'jobs' / f"{r['method']}_{r['pde']}"
        surrogate = OUT / 'jobs' / f"surrogate_{r['pde']}"
        if not (training / 'training_completed').exists(): continue
        if r['method'] == 'ddis' and not (surrogate / 'training_completed').exists(): continue
        for index in range(100):
            dest = ROOT / r['method'] / r['pde'] / r['task'] / r['split'] / f'case_{index:03d}'
            if dest.exists(): continue
            chosen = r, index, dest, training, surrogate
            break
        if chosen: break
    if chosen is None:
        status('Waiting for additional trained priors or all requested cases already claimed')
        time.sleep(60); continue
    gpu_lock = None
    for gpu in ([a.gpu] if a.gpu is not None else range(8)):
        busy = False
        for pde in ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger'):
            flow = OUT / 'jobs' / f'flow_{pde}'
            if (flow / 'gpu_index').exists() and not (flow / 'training_completed').exists():
                busy |= (flow / 'gpu_index').read_text().strip() == str(gpu)
        if busy: continue
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
        # Official full-schedule batch-one profiles measured <5 GiB for both
        # available two-channel priors; retain at least 15 GiB extra headroom.
        if free < 20480 or os.getloadavg()[0] >= 110: continue
        suffix = f'_slot_{a.slot}' if a.slot else ''
        candidate = (OUT / 'locks' / f'evaluation_gpu_{gpu}{suffix}.lock').open('w')
        try: fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close(); continue
        gpu_lock = candidate; break
    if gpu_lock is None:
        time.sleep(30); continue
    r, index, dest, training, surrogate = chosen
    try: dest.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        gpu_lock.close(); continue
    (dest / 'claim.json').write_text(json.dumps({'worker': a.worker, 'gpu': gpu, 'pid': os.getpid()}))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    checkpoint = (training / 'early_stopping/best_checkpoint').resolve()
    cmd = [PY, '-u', AD / 'evaluate_diffusion.py', '--method', r['method'],
        '--repo', BASE / 'official' / ('DDIS' if r['method'] == 'ddis' else 'FunDPS'),
        '--config', BASE / 'orchestration/configs/evaluation' / f"v2_{r['method']}_{r['pde']}_{r['split']}.yaml",
        '--checkpoint', checkpoint, '--source', OUT / 'data/compact' / r['pde'],
        '--fm4pde', BASE / 'official/FM4PDE-cbe627c',
        '--split', r['split'], '--task', r['task'], '--case-index', index, '--count', '1', '--output', dest]
    if r['method'] == 'ddis': cmd += ['--surrogate', (surrogate / 'early_stopping/best_checkpoint').resolve()]
    status(f"Sampling {r['method']} {r['pde']} {r['task']} {r['split']} case {index}")
    (dest / 'command.json').write_text(json.dumps(list(map(str, cmd)), indent=2))
    try:
        with (dest / 'sampling.log').open('w') as log:
            code = subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        (dest / 'sampling.exit').write_text(str(code))
        if code: raise RuntimeError(f'Official evaluation exited {code}')
        with (dest / 'verification.log').open('w') as log:
            subprocess.run(list(map(str, [PY, AD / 'validate_evaluation.py', '--source', OUT / 'data/compact' / r['pde'],
                '--split', r['split'], '--count', '1', '--output', dest])), stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    except Exception as error:
        (dest / 'worker_failure.json').write_text(json.dumps({'error': str(error)}))
    finally: gpu_lock.close()
