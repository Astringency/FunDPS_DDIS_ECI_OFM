"""Queue official four-PDE forward/inverse evaluation behind OFM and ECI-FM."""
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
a = p.parse_args()
state = OUT / 'jobs' / f'diffusion_evaluation_v2_{a.worker}'
state.mkdir(exist_ok=True)
def status(message): (state / 'status').write_text(message + '\n')
while True:
    if any(not (OUT / 'jobs' / f'flow_{pde}' / 'training_completed').exists()
           for pde in ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger')):
        status('Waiting for priority OFM training'); time.sleep(30); continue
    done = sum(json.loads(p.read_text())['cases'] for p in
               (OUT / 'evaluation_v2/fm4pde/eci').glob('*/*/*/shard_*/verified_summary.json'))
    if done < 2700:
        status(f'Waiting for priority ECI-FM evaluation: {done}/2700'); time.sleep(60); continue
    chosen = None
    matrix = json.loads((BASE / 'orchestration/configs/evaluation_matrix_v2.json').read_text())['evaluations']
    for r in matrix:
        if r['method'] not in ('ddis', 'fundps') or r['pde'] == 'burger':
            continue
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
    for gpu in range(8):
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
        if free < 40960 or os.getloadavg()[0] >= 110: continue
        candidate = (OUT / 'locks' / f'evaluation_gpu_{gpu}.lock').open('w')
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
