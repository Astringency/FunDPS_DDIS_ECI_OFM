"""Verify transferred compact fields before any added OFM training starts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(source):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'provenance'))
    from fm4pde_split_reference import _train_val_split_indices
    manifest = json.loads((source / 'manifest.json').read_text())
    channels = 1 if manifest['name'] == 'burger' else 2
    assert manifest['shape'] == [channels, 128, 128]
    train, validation = _train_val_split_indices(50000, manifest['split_seed'], .1)
    expected = {'train': train.numpy(), 'validation': validation.numpy(),
                **{split: np.arange(100) for split in ('id', 'smooth', 'rough')}}
    hashes = {}
    for split, ids in expected.items():
        entry = manifest['outputs'][split]
        path = source / entry['file']
        digest = sha256(path)
        assert digest == entry['sha256'], f'Checksum mismatch: {path}'
        assert np.array_equal(np.load(source / entry['ids']), ids)
        array = np.load(path, mmap_mode='r')
        assert array.shape == (len(ids), channels, 128, 128) and array.dtype == np.float32
        for start in range(0, len(ids), 256):
            assert np.isfinite(array[start:start + 256]).all()
        hashes[split] = digest
    result = {'pde': manifest['name'], 'manifest_sha256': sha256(source / 'manifest.json'),
              'verified_sha256': hashes, 'samples': {key: len(value) for key, value in expected.items()}}
    temporary = source / 'verified.partial.json'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(source / 'verified.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source)))
