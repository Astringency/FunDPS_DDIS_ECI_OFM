"""Calibrate native guidance parameters on validation cases, with held-out checks."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
import yaml

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
AD = BASE / 'orchestration/adapters'
PY = BASE / 'venv-shared-prior/bin/python'
ROOT = OUT / 'calibration_v2/poisson'
ROOT.mkdir(parents=True, exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES='6', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
lock = (OUT / 'locks/evaluation_gpu_6.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
free = int(subprocess.check_output(['nvidia-smi', '-i', '6', '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
assert free >= 40960 and os.getloadavg()[0] < 110
(ROOT / 'resources.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def run(cmd, log):
    with log.open('w') as f:
        subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, check=True)
old_assets = OUT / 'data/shared_prior_assets'
assets = ROOT / 'validation_assets'
source = ROOT / 'validation_compact'
assets.mkdir(exist_ok=True); source.mkdir(exist_ok=True)
manifest = json.loads((old_assets / 'manifest.json').read_text())
entry = manifest['pdes']['poisson']
saved = torch.load(old_assets / entry['splits']['id']['file'], weights_only=False, map_location='cpu')
original_source = OUT / 'data/compact/poisson'
metadata = json.loads((original_source / 'manifest.json').read_text())
x = np.array(np.load(original_source / 'validation.npy', mmap_mode='r')[:100])
mean = np.asarray(metadata['stats']['mean'])[None, :, None, None]
scale = np.asarray(metadata['stats']['std'])[None, :, None, None] / .5
physical = torch.from_numpy((x * scale + mean).astype(np.float32))
gt = saved['ground_truth']
gt.update(pair=physical, coef=physical[:, :1], sol=physical[:, 1:])
original_ids = np.load(original_source / 'validation_ids.npy')[:100]
gt['metadata'].update(calibration_partition='validation', original_sample_ids=original_ids.tolist())
truth = assets / 'poisson_validation.pt'
torch.save(saved, truth)
for pde, e in manifest['pdes'].items():
    e['checkpoint_file'] = str(old_assets / e['checkpoint_file'])
    for split, record in e['splits'].items():
        record['file'] = str(old_assets / record['file'])
manifest['pdes']['poisson']['splits']['id'].update(file=str(truth), sha256=sha(truth))
manifest['calibration'] = {'partition': 'validation', 'tuning_local_ids': list(range(8)),
    'heldout_local_ids': list(range(16, 24)), 'original_ids': original_ids.tolist()}
(assets / 'manifest.json').write_text(json.dumps(manifest, indent=2))
np.save(source / 'id.npy', x); np.save(source / 'id_ids.npy', np.arange(100))
(source / 'manifest.json').write_text(json.dumps(metadata, indent=2))
checkpoint = (OUT / 'jobs/flow_poisson/early_stopping/best_checkpoint').resolve()
old_checkpoint = OUT / 'training/flow/poisson/epoch_190.pt'
def evaluate(dest, task, overrides, ids, validation=True, weight=checkpoint, trace=False):
    dest.mkdir(parents=True, exist_ok=True)
    cfg = dest / 'overrides.json'; cfg.write_text(json.dumps(overrides))
    cmd = [PY, '-u', AD / 'evaluate_shared_prior.py', '--prior', 'ofm', '--method', 'fm4pde',
           '--pde', 'poisson', '--task', task, '--split', 'id', '--count', len(ids),
           '--case-indices', *ids, '--checkpoint', weight, '--assets', assets if validation else old_assets,
           '--source', source if validation else original_source, '--fm4pde', BASE / 'official/FM4PDE-cbe627c',
           '--ofm', BASE / 'official/OFM', '--eci', BASE / 'official/ECI', '--output', dest, '--fm-overrides', cfg]
    if trace: cmd += ['--trace-native']
    if not (dest / 'verified_summary.json').exists():
        run(cmd, dest / 'sampling.log')
        run([PY, AD / 'validate_shared_prior.py', '--output', dest], dest / 'verification.log')
    summary = json.loads((dest / 'verified_summary.json').read_text())
    field = 'solution' if task == 'forward' else 'coefficient'
    return {'successes': summary['successes'], 'cases': len(ids),
            'error': summary['all_case_mean_relative_l2_' + field], 'checkpoint_sha256': summary['checkpoint_sha256']}
# Same old checkpoint and same failed test case, changing only clipping.
diagnostic = {}
for label, override in [('original', {}), ('clip50', {'clip_threshold': 50.})]:
    diagnostic[label] = evaluate(ROOT / 'diagnostic' / label, 'forward', override, [38],
                                 validation=False, weight=old_checkpoint, trace=True)
(ROOT / 'diagnostic.json').write_text(json.dumps(diagnostic, indent=2))
print('Controlled diagnosis:', diagnostic, flush=True)
for task in ('forward', 'inverse'):
    config = yaml.safe_load((BASE / 'official/FM4PDE-cbe627c/configs/main' / task / 'poisson.yaml').read_text())
    candidates = {'original': {}, 'clip10': {'clip_threshold': 10.},
                  'clip50': {'clip_threshold': 50.}, 'clip200': {'clip_threshold': 200.}}
    for factor in (.001, .01, .1):
        candidates[f'obs_scale_{factor}'] = {'clip_threshold': 50.,
            'zeta_obs_a': config['zeta_obs_a'] * factor, 'zeta_obs_u': config['zeta_obs_u'] * factor,
            'zeta_pde': config['zeta_pde']}
    results = {}
    for name, override in candidates.items():
        results[name] = evaluate(ROOT / task / name, task, override, list(range(8)))
        print(task, name, results[name], flush=True)
    valid = [name for name, row in results.items() if row['successes'] == row['cases'] and row['error'] is not None]
    if not valid:
        raise RuntimeError(f'No stable validation candidate for {task}')
    best = min(valid, key=lambda name: results[name]['error'])
    selected = candidates[best]
    heldout = evaluate(ROOT / task / 'heldout_selected', task, selected, list(range(16, 24)))
    control = evaluate(ROOT / task / 'heldout_original', task, {}, list(range(16, 24)))
    report = {'partition': 'validation', 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'selection_metric': 'physical relative L2 on validation IDs 0:8', 'candidates': results,
        'selected_name': best, 'selected_overrides': selected, 'heldout_selected': heldout, 'heldout_original': control}
    (ROOT / task / 'selection_audit.json').write_text(json.dumps(report, indent=2))
    if heldout['successes'] == heldout['cases'] and (control['error'] is None or heldout['error'] <= control['error']):
        (ROOT / task / 'selected.json').write_text(json.dumps(selected, indent=2))
    else:
        (ROOT / task / 'needs_review.json').write_text(json.dumps(report, indent=2))
    print('Held-out result:', task, report, flush=True)
