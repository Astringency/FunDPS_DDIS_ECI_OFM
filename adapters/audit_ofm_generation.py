"""Unconditional official OFM sampling, independent of sparse observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from train_flow import build_prior

p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--pde', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
state = a.root / 'jobs' / f'flow_{a.pde}' / 'early_stopping'
best = json.loads((state / 'best.json').read_text())
checkpoint = Path(best['checkpoint'])
source = a.root / 'data/compact' / a.pde
metadata = json.loads((source / 'manifest.json').read_text())
validation = np.array(np.load(source / 'validation.npy', mmap_mode='r')[:256])
channels = validation.shape[1]
torch.set_num_threads(4)
model, prior = build_prior(a.base / 'official/OFM', 128, 'cuda', channels)
model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda'))
model.eval().requires_grad_(False)
torch.cuda.reset_peak_memory_stats()
generated = []
started = time.monotonic()
for seed in range(16):
    torch.manual_seed(20260920 + seed)
    x = prior.sample([128, 128], n_channels=channels, n_samples=1).detach().cpu().numpy()
    assert np.isfinite(x).all()
    generated.append(x)
x = np.concatenate(generated)
np.save(a.output / 'generated_normalized.npy', x)
mean = np.asarray(metadata['stats']['mean'])[None, :, None, None]
scale = np.asarray(metadata['stats']['std'])[None, :, None, None] / .5
np.save(a.output / 'generated_physical.npy', x * scale + mean)
def moments(v):
    return {'mean': v.mean(axis=(0, 2, 3)).tolist(), 'std': v.std(axis=(0, 2, 3)).tolist(),
            'rms_per_sample_mean': np.sqrt((v ** 2).mean(axis=(2, 3))).mean(axis=0).tolist(),
            'rms_per_sample_std': np.sqrt((v ** 2).mean(axis=(2, 3))).std(axis=0).tolist()}
report = {'pde': a.pde, 'checkpoint': str(checkpoint), 'best_epoch': best['progress'],
    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'sampler': 'unchanged official OFMModel.sample defaults; no observations/guidance',
    'generated_count': 16, 'reference_count': 256, 'generated': moments(x), 'validation': moments(validation),
    'seconds': time.monotonic() - started, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
    'interpretation': 'Unpaired distribution diagnostics; finite outputs and moments alone do not certify physical fidelity or justify automatic quality-based stopping.'}
(a.output / 'summary.json').write_text(json.dumps(report, indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2 * channels, 4, figsize=(12, 3 * channels * 2), squeeze=False)
for c in range(channels):
    lo, hi = np.quantile(validation[:, c], [.01, .99])
    for i in range(4):
        axes[2*c, i].imshow(x[i,c], vmin=lo, vmax=hi, cmap='RdBu_r')
        axes[2*c, i].set_title(f'Generated {i}, channel {c}')
        axes[2*c+1, i].imshow(validation[i,c], vmin=lo, vmax=hi, cmap='RdBu_r')
        axes[2*c+1, i].set_title(f'Independent validation {i}, channel {c}')
        axes[2*c,i].axis('off'); axes[2*c+1,i].axis('off')
fig.suptitle(f'{a.pde}: unconditional OFM, epoch {best["progress"]}; unpaired fields')
fig.tight_layout(); fig.savefig(a.output / 'generation.png', dpi=150); plt.close(fig)
print(json.dumps(report), flush=True)
