"""Export official FM4PDE evaluation inputs and pin the existing five priors.

Run on the authoritative data host. No training or sampling algorithm is changed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PDES = ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def main(args):
    sys.path.insert(0, str(args.fm4pde))
    import torch
    from models.legacy_checkpoint import read_checkpoint
    from sampling.config import load_config
    from sampling.data import load_ground_truth
    from sampling.model_io import _validate_checkpoint_payload

    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', '-C', str(args.fm4pde), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {'fm4pde_revision': revision, 'cases_per_split': 100,
                'source': 'FM4PDE load_ground_truth, unchanged', 'pdes': {}}
    for pde in PDES:
        task = 'both' if pde == 'burger' else 'inverse'
        config_file = args.fm4pde / 'configs/main' / task / f'{pde}.yaml'
        config = load_config(config_file)
        relative_checkpoint = Path(config.checkpoint_path).relative_to(args.fm4pde)
        checkpoint = args.fm4pde_work / relative_checkpoint
        payload = read_checkpoint(checkpoint, pde)
        _validate_checkpoint_payload(payload, checkpoint)
        entry = {'task': task, 'checkpoint_source': str(checkpoint),
                 'checkpoint_sha256': sha256(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
                 'checkpoint_file': f'weights/{pde}.pth',
                 'checkpoint_schema_version': payload['checkpoint_schema_version'],
                 'checkpoint_metadata': json_value({key: payload.get(key) for key in (
                     'model_config', 'model_config_metadata', 'model_profile', 'normalizer',
                     'epoch', 'num_channels', 'data_metadata', 'legacy_compatibility')}),
                 'has_ema': isinstance(payload.get('model_ema'), dict),
                 'config_source': str(config_file), 'config_sha256': sha256(config_file), 'splits': {}}
        link = args.output / entry['checkpoint_file']
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(checkpoint)
        del payload
        for split in ('id', 'smooth', 'rough'):
            relative_data = Path(config.data_paths[split]).relative_to(args.fm4pde / 'datasets')
            source = args.data_root / relative_data
            source_stat = source.stat()
            split_config = load_config(config_file, overrides={
                'data_path': str(source), 'test_type': split, 'batch_size': 100,
                'offset': 0, 'device': 'cpu', 'checkpoint_path': str(checkpoint)})
            truth = load_ground_truth(split_config)
            channels = 1 if pde == 'burger' else 2
            assert truth.pair.shape == (100, channels, 128, 128)
            assert torch.isfinite(truth.pair).all()
            assert truth.metadata['sample_offsets'] == list(range(100))
            assert not any(key in truth.pde_params for key in ('trajectory', 'near_endpoint_temporal'))
            target = args.output / 'truth' / pde / f'{split}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'ground_truth': asdict(truth), 'config': split_config.asdict()}, target)
            source_digest = sha256(source)
            assert source.stat().st_size == source_stat.st_size and source.stat().st_mtime_ns == source_stat.st_mtime_ns
            entry['splits'][split] = {'file': str(target.relative_to(args.output)),
                'sha256': sha256(target), 'source_file': str(source), 'source_sha256': source_digest,
                'source_bytes': source_stat.st_size, 'pde_params': json_value(truth.pde_params)}
            print(f'Exported {pde}/{split}: {list(truth.pair.shape)}', flush=True)
        manifest['pdes'][pde] = entry
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('All shared-prior assets exported and pinned', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fm4pde', type=Path, required=True)
    parser.add_argument('--fm4pde-work', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
