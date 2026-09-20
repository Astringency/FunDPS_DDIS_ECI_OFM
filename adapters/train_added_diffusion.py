"""Prepare aligned data, wait behind OFM, profile, then run an official trainer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
p = argparse.ArgumentParser()
p.add_argument('--method', choices=['ddis', 'fundps'], required=True)
p.add_argument('--pde', choices=['darcy', 'nsnonbounded', 'burger'], required=True)
a = p.parse_args()
if (OUT / 'jobs/diffusion_training_cancelled.json').exists():
    raise SystemExit('DDIS/FunDPS training cancelled by user; explicit reauthorization required.')
state = OUT / 'jobs' / f'{a.method}_{a.pde}'
state.mkdir(exist_ok=True)
def status(s):
    (state / 'status').write_text(s + '\n')
def run(cmd, log, **kwargs):
    with log.open('w') as f:
        subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, check=True, **kwargs)
repo = BASE / 'official' / ('DDIS' if a.method == 'ddis' else 'FunDPS')
adapters = BASE / 'orchestration/adapters'
source = OUT / 'data/compact' / a.pde
with (OUT / 'locks' / f'prepare_{a.pde}.lock').open('w') as prepare_lock:
    fcntl.flock(prepare_lock, fcntl.LOCK_EX)
    for split in ('train', 'validation', 'id', 'smooth', 'rough'):
        target = OUT / 'data/preprocessing' / split / 'data/DiffPDE' / (a.pde + ('_hf' if split == 'train' else '_test_hf'))
        if (target / 'metadata.json').exists():
            continue
        status(f'Preparing checksum-verified {split} data')
        run([sys.executable, adapters / 'prepare_data.py', 'hf', '--source', source,
             '--split', split, '--output', target, '--cache', OUT / 'data/hf_cache' / a.pde / split],
            state / f'prepare_{split}.log')
status('Waiting for all five priority OFM training jobs to complete')
while any(not (OUT / 'jobs' / f'flow_{q}' / 'training_completed').exists()
          for q in ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger')):
    time.sleep(30)
lock = None
while lock is None:
    status('Waiting for a training GPU with 40 GiB free')
    for gpu in range(8):
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
        if free < 40960 or os.getloadavg()[0] >= 110 or available < 64000 * 1024:
            continue
        candidate = (OUT / 'locks' / f'gpu_{gpu}.lock').open('w')
        try:
            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close()
            continue
        lock = candidate
        break
    if lock is None:
        time.sleep(30)
os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  WANDB_MODE='offline', WANDB_DIR=str(state), MPLBACKEND='Agg', MASTER_PORT=str(29900 + gpu))
(state / 'gpu_index').write_text(str(gpu))
(state / 'resources_at_start.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
config = yaml.safe_load((BASE / 'orchestration/configs/training' / f'{a.method}_{a.pde}.yaml').read_text())
for batch in (1, config['batch']):
    status(f'Profiling official training at batch {batch}')
    profile = dict(config, batch=batch, batch_gpu=batch, outdir=str(OUT / 'profiles' / f'{a.method}_{a.pde}_v2_b{batch}'))
    cfg = state / f'profile_b{batch}.yaml'
    cfg.write_text(yaml.safe_dump(profile))
    metrics = state / f'profile_b{batch}.json'
    run([sys.executable, adapters / 'profile_official_training.py', '--repo', repo, '--method', a.method,
         '--config', cfg, '--profile-steps', '3', '--metrics', metrics], state / f'profile_b{batch}.log', cwd=repo)
    measured = json.loads(metrics.read_text())['peak_reserved_bytes']
    assert measured < (free - 6000) * 1024 ** 2, 'Insufficient measured headroom'
status('Training official model with epoch-window early stopping')
(state / 'started').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
cfg = state / 'config.yaml'
cfg.write_text(yaml.safe_dump(config))
run([sys.executable, adapters / 'early_stop.py', '--method', a.method,
     '--policy', BASE / 'orchestration/configs/early_stopping_v2.json', '--state', state / 'early_stopping',
     '--training-root', config['outdir'], '--repo', repo,
     '--validation', OUT / 'data/preprocessing/validation/data/DiffPDE' / f'{a.pde}_test_hf', '--',
     sys.executable, '-u', repo / ('scripts/train/train.py' if a.method == 'ddis' else 'train.py'), '-c', cfg],
    state / 'supervisor.log', cwd=repo)
(state / 'training_completed').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
status('Training completed; validation-best checkpoint retained')
