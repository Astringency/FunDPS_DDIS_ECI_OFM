"""Add Poisson Rough2/Rough3 inputs to a separate, read-only pilot asset set."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--existing', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from scipy.io import loadmat

    sys.path.insert(0, str(args.official))
    from sampling.config import load_config
    from sampling.data import load_ground_truth
    from prepare_data import make_hf

    root = args.root
    compact = root / 'compact' / 'poisson'
    assets = root / 'assets'
    if (compact / 'manifest.json').exists() or (assets / 'manifest.json').exists():
        raise FileExistsError('Pilot input manifest already exists')
    compact.mkdir(parents=True, exist_ok=True)
    (assets / 'truth' / 'poisson').mkdir(parents=True, exist_ok=True)
    (assets / 'weights').mkdir(parents=True, exist_ok=True)

    original_compact = args.existing / 'data' / 'compact' / 'poisson'
    original_assets = args.existing / 'data' / 'shared_prior_assets'
    source_manifest = json.loads((original_compact / 'manifest.json').read_text())
    asset_manifest = json.loads((original_assets / 'manifest.json').read_text())
    source_manifest['outputs'] = dict(source_manifest['outputs'])
    asset_manifest['pdes'] = {'poisson': asset_manifest['pdes']['poisson']}
    asset_entry = asset_manifest['pdes']['poisson']
    (assets / 'weights' / 'poisson.pth').symlink_to((original_assets / 'weights' / 'poisson.pth').resolve())
    for split in ('id', 'smooth', 'rough'):
        for suffix in ('.npy', '_ids.npy'):
            name = split + suffix
            (compact / name).symlink_to((original_compact / name).resolve())
        target = assets / 'truth' / 'poisson' / (split + '.pt')
        target.symlink_to((original_assets / 'truth' / 'poisson' / (split + '.pt')).resolve())

    mean = np.asarray(source_manifest['stats']['mean'], dtype=np.float64)[:, None, None]
    std = np.asarray(source_manifest['stats']['std'], dtype=np.float64)[:, None, None]
    for split in ('rough2', 'rough3'):
        raw = root / 'raw' / f'poisson_test_1000-128-128_{split}.mat'
        assert raw.exists(), raw
        data = loadmat(raw)
        coefficient = np.asarray(data['f_data'][:100], dtype=np.float64)
        solution = np.asarray(data['phi_data'][:100], dtype=np.float64)
        assert coefficient.shape == solution.shape == (100, 128, 128)
        physical = np.stack((coefficient, solution), axis=1)
        normalized = ((physical - mean) * (0.5 / std)).astype(np.float32)
        assert np.isfinite(normalized).all()
        array_path = compact / (split + '.npy')
        np.save(array_path, normalized)
        np.save(compact / (split + '_ids.npy'), np.arange(100))
        source_manifest['outputs'][split] = {'file': array_path.name, 'sha256': sha256(array_path),
                                             'samples': 100, 'ids': split + '_ids.npy'}
        source_manifest.setdefault('sources', []).append({'path': str(raw), 'sha256': sha256(raw)})

        config = load_config(args.official / 'configs/main/inverse/poisson.yaml', {
            'data_path': str(raw), 'test_type': split, 'batch_size': 100,
            'offset': 0, 'device': 'cpu', 'checkpoint_path': str(assets / 'weights' / 'poisson.pth')})
        truth = load_ground_truth(config)
        assert truth.pair.shape == (100, 2, 128, 128)
        np.testing.assert_allclose(normalized.astype(np.float64) * (std / 0.5) + mean,
                                   truth.pair.numpy().astype(np.float64), rtol=1e-5, atol=1e-5)
        truth_path = assets / 'truth' / 'poisson' / (split + '.pt')
        torch.save({'ground_truth': asdict(truth), 'config': config.asdict()}, truth_path)
        asset_entry['splits'][split] = {'file': str(truth_path.relative_to(assets)),
                                         'sha256': sha256(truth_path), 'source_file': str(raw),
                                         'source_sha256': sha256(raw), 'source_bytes': raw.stat().st_size,
                                         'pde_params': {}}
        del data, coefficient, solution, physical, normalized, truth

    (compact / 'manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    (assets / 'manifest.json').write_text(json.dumps(asset_manifest, indent=2) + '\n')
    from argparse import Namespace
    for split in ('rough2', 'rough3'):
        make_hf(Namespace(source=compact, split=split,
                          output=root / 'hf' / 'poisson' / split, cache=root / 'hf_cache' / split))
    print(json.dumps({'compact_manifest_sha256': sha256(compact / 'manifest.json'),
                      'assets_manifest_sha256': sha256(assets / 'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
