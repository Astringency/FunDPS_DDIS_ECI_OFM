"""Measure official ECI batch throughput and check per-ID numerical agreement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
AD = BASE / 'orchestration/adapters'
p = argparse.ArgumentParser()
p.add_argument('--prior', choices=['fm4pde', 'ofm'], required=True)
p.add_argument('--gpu', type=int, required=True)
a = p.parse_args()
os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
root = OUT / 'profiles/eci_batch_20260921' / a.prior
root.mkdir(parents=True, exist_ok=True)
results = {}
for pde in ['poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger']:
    task = 'both' if pde == 'burger' else 'inverse'
    assets = OUT / 'data/shared_prior_assets'
    ckpt = (OUT / 'jobs' / f'flow_{pde}' / 'early_stopping/best_checkpoint').resolve() if a.prior == 'ofm' else assets / 'weights' / f'{pde}.pth'
    measurements = []
    for batch in [1, 4, 8]:
        dest = root / pde / f'b{batch}'
        dest.mkdir(parents=True, exist_ok=True)
        count = 4 if batch == 1 else batch
        cmd = [sys.executable, '-u', AD / 'evaluate_shared_prior.py', '--prior', a.prior, '--method', 'eci',
            '--pde', pde, '--task', task, '--split', 'id', '--offset', '0', '--count', str(count),
            '--checkpoint', ckpt, '--assets', assets, '--source', OUT / 'data/compact' / pde,
            '--fm4pde', BASE / 'official/FM4PDE-cbe627c', '--ofm', BASE / 'official/OFM', '--eci', BASE / 'official/ECI',
            '--output', dest, '--eci-batch-size', str(batch), '--eci-steps', '20', '--profile']
        if not (dest / 'verified_summary.json').exists():
            with (dest / 'sampling.log').open('w') as log:
                subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT, check=True)
            with (dest / 'verification.log').open('w') as log:
                subprocess.run([sys.executable, str(AD / 'validate_shared_prior.py'), '--output', str(dest)], stdout=log, stderr=subprocess.STDOUT, check=True)
        records = [json.loads(x.read_text()) for x in dest.glob('case_*.json')]
        agreement = []
        for index in range(4):
            baseline = np.load(root / pde / 'b1' / f'prediction_{index:03d}.npy')
            actual = np.load(dest / f'prediction_{index:03d}.npy')
            delta = float(np.linalg.norm(actual - baseline) / max(np.linalg.norm(baseline), 1e-30))
            agreement.append(delta)
        row = {'batch': batch, 'seconds_per_case': sum(x['seconds'] for x in records) / count,
               'peak_reserved_bytes': max(x['peak_reserved_bytes'] for x in records),
               'max_prediction_relative_difference': max(agreement),
               'short_agreement_passed': all(np.isfinite(x) and x < 1e-3 for x in agreement)}
        measurements.append(row)
        print(pde, json.dumps(row), flush=True)
    results[pde] = measurements
    (root / 'measurements.json').write_text(json.dumps(results, indent=2))
# Full schedules validate the actual integration budget, including models whose
# short, coarser integration amplifies roundoff. Never relax the full-run check.
full_results = {}
for pde, measurements in results.items():
    batch = min(measurements[1:], key=lambda x: x['seconds_per_case'])['batch']
    task = 'both' if pde == 'burger' else 'inverse'
    dest = root / pde / f'full_b{batch}'
    dest.mkdir(exist_ok=True)
    assets = OUT / 'data/shared_prior_assets'
    ckpt = (OUT / 'jobs' / f'flow_{pde}' / 'early_stopping/best_checkpoint').resolve() if a.prior == 'ofm' else assets / 'weights' / f'{pde}.pth'
    cmd = [sys.executable, '-u', AD / 'evaluate_shared_prior.py', '--prior', a.prior, '--method', 'eci',
           '--pde', pde, '--task', task, '--split', 'id', '--count', str(batch), '--checkpoint', ckpt,
           '--assets', assets, '--source', OUT / 'data/compact' / pde, '--fm4pde', BASE / 'official/FM4PDE-cbe627c',
           '--ofm', BASE / 'official/OFM', '--eci', BASE / 'official/ECI', '--output', dest, '--eci-batch-size', str(batch), '--profile']
    with (dest / 'sampling.log').open('w') as log:
        subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT, check=True)
    full = []
    for index in range(batch):
        ref = OUT / 'evaluation_v2' / a.prior / 'eci' / pde / task / 'id/shard_000' / f'prediction_{index:03d}.npy'
        baseline = np.load(ref)
        actual = np.load(dest / f'prediction_{index:03d}.npy')
        delta = float(np.linalg.norm(actual - baseline) / max(np.linalg.norm(baseline), 1e-30))
        full.append({'sample_id': index, 'relative_difference': delta, 'reference': str(ref)})
    with (dest / 'verification.log').open('w') as log:
        subprocess.run([sys.executable, str(AD / 'validate_shared_prior.py'), '--output', str(dest)], stdout=log, stderr=subprocess.STDOUT, check=True)
    full_results[pde] = {'batch': batch, 'agreement': full, 'passed': all(np.isfinite(x['relative_difference']) and x['relative_difference'] < 1e-3 for x in full)}
    (root / 'full_schedule_agreement.json').write_text(json.dumps(full_results, indent=2))
    print('FULL', pde, 'batch', batch, 'passed', full_results[pde]['passed'], flush=True)
(root / 'completed').write_text('Batch checks completed; promote only per-PDE passing configurations.\n')
