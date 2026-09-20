"""Export the three added PDEs through the unchanged FM4PDE data loaders.

Fit channel statistics on the training partition only. Scale standardized
fields by 0.5, matching the coordinate convention of the existing OFM priors.
No model, training algorithm, or sampling algorithm is implemented here.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    sys.path.insert(0, str(args.fm4pde.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'provenance'))
    import torch
    from data.load import PDEloader
    from data.transform import PDEStandardizer
    from data.specs import get_pde_spec
    from fm4pde_split_reference import _train_val_split_indices

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    spec = get_pde_spec(args.pde)
    shape = (spec.img_channels, 128, 128)
    directory = 'burgers' if args.pde == 'burger' else args.pde
    suffix = '-10' if args.pde == 'nsnonbounded' else ''
    seed = args.seed + spec.label_id * 1009
    train_ids, val_ids = _train_val_split_indices(50000, seed, .1)
    ids = {'train': train_ids.numpy(), 'validation': val_ids.numpy()}
    assert len(set(ids['train']).intersection(ids['validation'])) == 0
    assert np.array_equal(np.sort(np.concatenate(list(ids.values()))), np.arange(50000))
    physical = {split: np.lib.format.open_memmap(args.output / f'.physical_{split}.npy',
        mode='w+', dtype=np.float32, shape=(len(indices), *shape)) for split, indices in ids.items()}
    sources = []

    def load(relative, limit=None):
        path = args.raw / directory / relative
        before = path.stat()
        digest = checksum(path)
        loader = PDEloader(args.pde)
        tensor, labels = loader.load_func[args.pde](path, size=1, max_samples=limit)
        expected_count = 10000 if limit is None else limit
        assert tuple(tensor.shape) == (expected_count, *shape)
        assert torch.isfinite(tensor).all()
        assert (labels == spec.label_id).all()
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        sources.append({'path': str(path), 'sha256': digest, 'bytes': after.st_size,
                        'loader': f'FM4PDE PDEloader.load_func[{args.pde}]', 'loaded_samples': expected_count})
        return tensor

    for shard in range(1, 6):
        tail = '_new' if args.pde == 'nsnonbounded' else ''
        tensor = load(f'{args.pde}_10000-128-128{suffix}_{shard}{tail}.mat')
        lower = (shard - 1) * 10000
        for split, indices in ids.items():
            destinations = np.flatnonzero((indices >= lower) & (indices < lower + 10000))
            for start in range(0, len(destinations), 256):
                where = destinations[start:start + 256]
                physical[split][where] = tensor[indices[where] - lower].numpy()
        del tensor
        gc.collect()
        print(f'Exported physical training shard {shard}/5', flush=True)
    for array in physical.values():
        array.flush()
    normalizer = PDEStandardizer.fit(torch.from_numpy(physical['train']),
        channel_names=list(spec.channel_names), pde=args.pde)
    normalizer.save(args.output / 'normalizer.pt')
    outputs = {}

    def write_split(split, array, indices):
        temporary = args.output / f'.{split}.npy'
        output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                           shape=(len(indices), *shape))
        largest_roundtrip_error = 0.
        for start in range(0, len(indices), 256):
            original = torch.from_numpy(np.array(array[start:start + 256], copy=True))
            scaled = normalizer.transform(original) * .5
            assert torch.isfinite(scaled).all()
            reconstructed = normalizer.inverse_transform(scaled / .5)
            largest_roundtrip_error = max(largest_roundtrip_error, float((original - reconstructed).abs().max()))
            assert torch.allclose(original, reconstructed, rtol=1e-5, atol=1e-5)
            output[start:start + len(original)] = scaled.numpy()
        output.flush()
        del output
        final = args.output / f'{split}.npy'
        os.replace(temporary, final)
        np.save(args.output / f'{split}_ids.npy', indices)
        outputs[split] = {'file': final.name, 'sha256': checksum(final), 'samples': len(indices),
            'ids': f'{split}_ids.npy', 'max_physical_roundtrip_absolute_error': largest_roundtrip_error}
        print(f'Verified normalized {split}: {len(indices)} samples', flush=True)

    for split, array in physical.items():
        write_split(split, array, ids[split])
    for split in ('id', 'smooth', 'rough'):
        tensor = load(f'{args.pde}_test_10000-128-128{suffix}_{split}.mat', limit=100)
        write_split(split, tensor.numpy(), np.arange(100))
        del tensor
    revision = subprocess.check_output(['git', '-C', str(args.fm4pde), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {'name': args.pde, 'shape': list(shape), '__version__': 'fm4pde_training_partition_stats_v1',
        'stats': {'mean': normalizer.mean.flatten().tolist(), 'std': normalizer.std.flatten().tolist()},
        'normalization': '(physical-mean)*0.5/std; statistics fitted only to 45,000 training cases by FM4PDE PDEStandardizer.fit',
        'storage_dtype': 'float32', 'seed': args.seed, 'split_seed': seed,
        'fm4pde_revision': revision, 'split_reference': 'provenance/fm4pde_split_reference.py',
        'sources': sources, 'outputs': outputs, 'channel_names': list(spec.channel_names),
        'axis_semantics': 'time_space' if args.pde == 'burger' else 'spatial_2d',
        'temporary_physical_arrays': ['.physical_train.npy', '.physical_validation.npy']}
    temporary = args.output / 'manifest.partial.json'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(args.output / 'manifest.json')
    print('Export completed and verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pde', choices=['darcy', 'nsnonbounded', 'burger'], required=True)
    parser.add_argument('--fm4pde', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
