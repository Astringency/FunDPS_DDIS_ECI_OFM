"""Export untouched OOD validation cases through the pinned FM4PDE loader."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(a):
    sys.path.insert(0, str(a.fm4pde))
    import torch
    from sampling.config import load_config
    from sampling.data import load_ground_truth
    torch.set_num_threads(1)
    manifest = json.loads(a.manifest.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    for pde in ('nsnonbounded', 'burger', 'poisson', 'helmholtz', 'darcy'):
        for split in ('rough', 'smooth'):
            folder = a.output / pde / split
            if (folder / 'export.json').exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            old = manifest['pdes'][pde]['splits'][split]
            source = Path(old['source_file'])
            before = source.stat()
            assert before.st_size == old['source_bytes']
            assert sha(source) == old['source_sha256']
            task = 'both' if pde == 'burger' else 'inverse'
            config = load_config(a.fm4pde / f'configs/main/{task}/{pde}.yaml', dict(
                data_path=str(source), test_type=split, batch_size=100,
                offset=1000, device='cpu', checkpoint_path=manifest['pdes'][pde]['checkpoint_source']))
            truth = load_ground_truth(config)
            assert truth.metadata['sample_offsets'] == list(range(1000, 1100))
            assert truth.pair.shape == (100, 1 if pde == 'burger' else 2, 128, 128)
            assert torch.isfinite(truth.pair).all()
            truth.metadata.update(original_sample_offsets=list(range(1000, 1100)),
                sample_offsets=list(range(100)), calibration_partition='reserved_ood_validation')
            target = folder / 'truth.pt'
            torch.save(dict(ground_truth=asdict(truth), config=config.asdict()), target)
            after = source.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            record = dict(pde=pde, split=split, original_ids=list(range(1000, 1100)),
                excluded_formal_test_ids=list(range(100)), truth_sha256=sha(target),
                source_file=str(source), source_sha256=old['source_sha256'],
                source_bytes=old['source_bytes'], partition='reserved_ood_validation')
            temporary = folder / 'export.partial.json'
            temporary.write_text(json.dumps(record, indent=2));temporary.replace(folder / 'export.json')
            print('Exported reserved validation:', pde, split, flush=True)
            del truth
            gc.collect()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fm4pde', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
