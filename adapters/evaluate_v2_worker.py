"""Sample-level shards with ECI-FM priority; never share a GPU with active OFM training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
ROOT = OUT / 'evaluation_v2'
AD = BASE / 'orchestration/adapters'
PY = BASE / 'venv-shared-prior/bin/python'
p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, required=True)
a = p.parse_args()
ROOT.mkdir(exist_ok=True)
matrix = json.loads((BASE / 'orchestration/configs/evaluation_matrix_v2.json').read_text())
matrix = [r for r in matrix['evaluations'] if r['prior'] in ('fm4pde', 'ofm')]
os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
lock = (OUT / 'locks' / f'evaluation_gpu_{a.gpu}.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
def run(cmd, logfile):
    with logfile.open('w') as f:
        subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, check=True)
while True:
    matrix = json.loads((BASE / 'orchestration/configs/evaluation_matrix_v2.json').read_text())['evaluations']
    matrix = [r for r in matrix if r['prior'] in ('fm4pde', 'ofm')]
    busy = False
    for q in ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger'):
        state = OUT / 'jobs' / f'flow_{q}'
        if (state / 'gpu_index').exists() and not (state / 'training_completed').exists():
            busy |= (state / 'gpu_index').read_text().strip() == str(a.gpu)
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
    if busy or free < 40960 or os.getloadavg()[0] >= 110:
        time.sleep(30)
        continue
    chosen = None
    # Finish all ECI-FM shards before the secondary comparisons.
    for priority in (0, 1):
        items = [r for r in matrix if (0 if (r['prior'], r['method']) == ('fm4pde', 'eci') else 1) == priority]
        pending = False
        for offset in range(0, 100, 10):
            for r in items:
                cell = ROOT / r['prior'] / r['method'] / r['pde'] / r['task'] / r['split']
                dest = cell / f'shard_{offset:03d}'
                if (dest / 'verified_summary.json').exists() or (dest / 'worker_failure.json').exists():
                    continue
                pending = True
                if (dest / 'claimed.json').exists():
                    continue
                assets = OUT / 'data/shared_prior_assets'
                if r['prior'] == 'ofm':
                    state = OUT / 'jobs' / f"flow_{r['pde']}"
                    if not (state / 'training_completed').exists():
                        continue
                    checkpoint = (state / 'early_stopping/best_checkpoint').resolve()
                else:
                    checkpoint = assets / 'weights' / f"{r['pde']}.pth"
                overrides = OUT / 'calibration_v2' / r['pde'] / r['task'] / 'selected.json'
                if r['prior'] == 'ofm' and r['method'] == 'fm4pde' and not overrides.exists():
                    continue
                try:
                    dest.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    continue
                (dest / 'claimed.json').write_text(json.dumps({'gpu': a.gpu, 'pid': os.getpid(), 'time': time.time()}))
                chosen = (r, offset, dest, checkpoint, assets, overrides)
                break
            if chosen:
                break
        if chosen or pending:
            break
    if chosen is None:
        time.sleep(30)
        continue
    r, offset, dest, checkpoint, assets, overrides = chosen
    cmd = [PY, '-u', AD / 'evaluate_shared_prior.py', '--prior', r['prior'], '--method', r['method'],
           '--pde', r['pde'], '--task', r['task'], '--split', r['split'], '--offset', str(offset), '--count', '10',
           '--checkpoint', checkpoint, '--assets', assets, '--source', OUT / 'data/compact' / r['pde'],
           '--fm4pde', BASE / 'official/FM4PDE-cbe627c', '--ofm', BASE / 'official/OFM', '--eci', BASE / 'official/ECI',
           '--output', dest]
    if r['prior'] == 'ofm' and r['method'] == 'fm4pde':
        cmd += ['--fm-overrides', overrides]
    (dest / 'command.json').write_text(json.dumps(list(map(str, cmd)), indent=2))
    try:
        probe = dest / 'resource_profile'
        probe_cmd = list(cmd)
        probe_cmd[probe_cmd.index('--count') + 1] = '1'
        probe_cmd[probe_cmd.index('--output') + 1] = probe
        run(probe_cmd + ['--profile'], dest / 'profile.log')
        records = [json.loads(x.read_text()) for x in probe.glob('case_*.json')]
        peak = max([json.loads((probe / 'model_loaded.json').read_text())['peak_reserved_bytes']] + [x['peak_reserved_bytes'] for x in records])
        assert peak < (free - 6000) * 1024 ** 2, 'Insufficient measured sampling headroom'
        run(cmd, dest / 'sampling.log')
        run([PY, AD / 'validate_shared_prior.py', '--output', dest], dest / 'verification.log')
    except Exception as e:
        (dest / 'worker_failure.json').write_text(json.dumps({'error': str(e), 'time': time.time()}))
    print(str(dest), flush=True)
