"""Independently recompute metrics for the expanded shared-prior comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    result = {}
    assert set(manifest['pdes']) == {'poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger'}
    for pde, entry in manifest['pdes'].items():
        weight = root / entry['checkpoint_file']
        assert weight.stat().st_size == entry['checkpoint_bytes']
        assert sha256(weight) == entry['checkpoint_sha256']
        assert set(entry['splits']) == {'id', 'smooth', 'rough'}
        for split, record in entry['splits'].items():
            path = root / record['file']
            assert sha256(path) == record['sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)['ground_truth']
            assert saved['pair'].shape == (100, 1 if pde == 'burger' else 2, 128, 128)
            assert saved['metadata']['sample_offsets'] == list(range(100))
            assert torch.isfinite(saved['pair']).all()
        result[pde] = entry['checkpoint_sha256']
    result = {'manifest_sha256': sha256(root / 'manifest.json'), 'checkpoint_sha256': result}
    (root / 'verified.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def validate(output):
    run = json.loads((output / 'run.json').read_text())
    assets_root = Path(run['assets'])
    assert sha256(Path(run['checkpoint'])) == run['checkpoint_sha256']
    assert sha256(assets_root / 'manifest.json') == run['assets_manifest_sha256']
    manifest = json.loads((assets_root / 'manifest.json').read_text())
    source = assets_root / manifest['pdes'][run['pde']]['splits'][run['split']]['file']
    assert sha256(source) == run['truth_sha256']
    target = torch.load(source, map_location='cpu', weights_only=False)['ground_truth']['pair'].numpy().astype(np.float64)
    channels = target.shape[1]
    rng = np.random.RandomState(run['seed'])
    expected_coefficient, expected_masks = [], []
    for _ in range(100):
        expected_coefficient.append(rng.choice(16384, 500, replace=False))
        expected_masks.append(rng.choice(16384, 500, replace=False))
    masks = np.load(output / 'solution_observation_indices.npy')
    np.testing.assert_array_equal(masks, np.stack(expected_masks))
    coefficient_path = output / 'coefficient_observation_indices.npy'
    if coefficient_path.exists():
        np.testing.assert_array_equal(np.load(coefficient_path), np.stack(expected_coefficient))
    ids = run.get('case_indices') or list(range(run['offset'], run['offset'] + run['count']))
    assert sorted(int(path.stem.split('_')[1]) for path in output.glob('case_*.json')) == sorted(ids)
    records = []
    for index in ids:
        record = json.loads((output / f'case_{index:03d}.json').read_text())
        assert record['sample_id'] == index
        assert record['status'] in ('ok', 'failed')
        if record['status'] == 'failed':
            assert record['relative_l2_coefficient'] is None and record['relative_l2_solution'] is None
            assert record.get('error')
        else:
            prediction = np.load(output / f'prediction_{index:03d}.npy')
            truth = target[index:index + 1]
            assert prediction.shape == truth.shape and np.isfinite(prediction).all()
            errors = np.sqrt(((prediction - truth) ** 2).sum(axis=(0, 2, 3)) / (truth ** 2).sum(axis=(0, 2, 3)))
            np.testing.assert_allclose(record['relative_l2_solution'], errors[-1], rtol=1e-9, atol=1e-12)
            if channels == 2:
                np.testing.assert_allclose(record['relative_l2_coefficient'], errors[0], rtol=1e-9, atol=1e-12)
            else:
                assert record['relative_l2_coefficient'] is None
        records.append(record)
    success = [record for record in records if record['status'] == 'ok']
    result = {'prior': run['prior'], 'method': run['method'], 'pde': run['pde'],
        'split': run['split'], 'task': run['task'],
        'profile_only': run['profile'], 'cases': len(records), 'successes': len(success),
        'failures': len(records) - len(success), 'sample_ids': ids,
        'checkpoint_sha256': run['checkpoint_sha256'], 'truth_sha256': run['truth_sha256'],
        'masks_sha256': sha256(output / 'solution_observation_indices.npy')}
    for field in ('coefficient', 'solution'):
        values = [record['relative_l2_' + field] for record in success]
        values = [value for value in values if value is not None]
        conditional = float(np.mean(values)) if values else None
        result['successful_case_mean_relative_l2_' + field] = conditional
        result['all_case_mean_relative_l2_' + field] = conditional if len(values) == len(records) else None
    (output / 'verified_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--assets', type=Path)
    group.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_assets(args.assets) if args.assets else validate(args.output)))
