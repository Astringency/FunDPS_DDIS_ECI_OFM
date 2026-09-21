"""Batch/data adapters around the unchanged official ECI sampling function."""
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from shared_prior_runtime import ECIVelocityAdapter, case_ground_truth, common_masks, native_model_extra


def atomic_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class FixedInitialNoise:
    """Supply identical per-ID official prior draws independent of batching."""
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def sample(self, grid, dims, n_samples=1):
        self.calls += 1
        assert self.calls == 1  # Official resample_step=None draws only once.
        assert list(self.value.shape) == [n_samples, *dims]
        return self.value.clone()


def combine_extra(extras):
    if all(x is None for x in extras):
        return None
    keys = set(extras[0])
    assert all(x is not None and set(x) == keys for x in extras)
    result = {}
    for key in keys:
        values = [x[key] for x in extras]
        if isinstance(values[0], torch.Tensor):
            assert all(v.ndim > 0 and v.shape[0] == 1 for v in values)
            result[key] = torch.cat(values, dim=0)
        else:
            assert all(v == values[0] for v in values)
            result[key] = values[0]
    return result


def run_eci_batches(args, indices, saved, assets, task, net, normalizer, payload, noise, FFM, DirichletCondition):
    from sampling.config import load_config

    pending = [i for i in indices if not (args.output / f'case_{i:03d}.json').exists()]
    channels = 1 if args.pde == 'burger' else 2
    for offset in range(0, len(pending), args.eci_batch_size):
        batch_ids = pending[offset:offset + args.eci_batch_size]
        truths, masks, extras, draws, noise_hashes = [], [], [], [], []
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for index in batch_ids:
            truth = case_ground_truth(saved['ground_truth'], index, args.device)
            mask, _ = common_masks(truth, index, args.seed, task if args.pde != 'burger' else 'inverse')
            config = load_config(args.fm4pde / 'configs/main' / task / f'{args.pde}.yaml', {
                'test_type': args.split, 'data_path': assets['splits'][args.split]['source_file'],
                'checkpoint_path': str(args.checkpoint), 'device': args.device,
                'batch_size': 1, 'offset': index, 'model_profile': 'auto'})
            extras.append(native_model_extra(payload, truth, config))
            truths.append(truth.pair)
            masks.append(torch.cat((mask.coef.bool(), mask.sol.bool()), dim=1) if channels == 2 else mask.sol.bool())
            torch.manual_seed(args.seed + index)
            draw = noise.sample(None, [channels, 128, 128], n_samples=1)
            draws.append(draw)
            noise_hashes.append(hashlib.sha256(draw.detach().cpu().numpy().tobytes()).hexdigest())
        target = torch.cat(truths)
        fixed = FixedInitialNoise(torch.cat(draws))
        eci = SimpleNamespace(model=ECIVelocityAdapter(net, combine_extra(extras)), gp=fixed)
        with torch.no_grad():
            sampled = FFM.eci_sample(eci, len(batch_ids), args.eci_steps, args.eci_mix, None,
                [channels, 128, 128], args.device,
                DirichletCondition(value=normalizer.transform(target), mask=torch.cat(masks)))
            prediction = normalizer.inverse_transform(sampled).detach().cpu().numpy().astype(np.float64)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        actual = target.detach().cpu().numpy().astype(np.float64)
        assert prediction.shape == actual.shape
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        for k, index in enumerate(batch_ids):
            result = prediction[k:k+1]
            np.save(args.output / f'prediction_{index:03d}.npy', result)
            finite = bool(np.isfinite(result).all())
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                errors = np.linalg.norm((result - actual[k:k+1]).reshape(channels, -1), axis=1) / np.linalg.norm(actual[k:k+1].reshape(channels, -1), axis=1)
            finite = finite and bool(np.isfinite(errors).all())
            record = {'sample_id': index, 'status': 'ok' if finite else 'failed',
                'relative_l2_coefficient': float(errors[0]) if finite and channels == 2 else None,
                'relative_l2_solution': float(errors[-1]) if finite else None,
                'seconds': elapsed / len(batch_ids), 'timing_scope': 'batch wall time divided by actual batch size',
                'batch_seconds': elapsed, 'batch_size': len(batch_ids), 'batch_sample_ids': batch_ids,
                'initial_noise_sha256': noise_hashes[k],
                'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved}
            if not finite:
                record['error'] = 'Nonfinite prediction or relative error in batched official ECI'
            atomic_json(args.output / f'case_{index:03d}.json', record)
        atomic_json(args.output / f'batch_{batch_ids[0]:03d}.json', {
            'sample_ids': batch_ids, 'seconds': elapsed, 'peak_reserved_bytes': peak_reserved})
        print(json.dumps({'batch': batch_ids, 'seconds': elapsed, 'peak_reserved_bytes': peak_reserved}), flush=True)
