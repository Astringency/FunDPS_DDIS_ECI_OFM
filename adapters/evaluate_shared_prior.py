"""Evaluate official samplers on a pinned ordinary FM or operator FM prior."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_flow import pair_observation_indices, ofm_sample
from shared_prior_runtime import (ECIVelocityAdapter, case_ground_truth, common_masks,
    load_eci, load_prior, native_model_extra, native_noise_provider)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(args):
    from native_diagnostics import trace_native
    if args.eci_batch_size == 0:
        policy = Path(__file__).resolve().parents[1] / 'configs/eci_batching.json'
        runtime_policy = args.assets.parent.parent / 'jobs/eci_batch_policy.json'
        if runtime_policy.exists():
            policy = runtime_policy
        settings = json.loads(policy.read_text()) if policy.exists() else {}
        args.eci_batch_size = settings.get(args.prior, {}).get(args.pde, 1) if args.method == 'eci' else 1
    existing_run = args.output / 'run.json'
    if existing_run.exists():
        args.eci_batch_size = json.loads(existing_run.read_text()).get('eci_batch_size', 1)
    fm_overrides = json.loads(args.fm_overrides.read_text()) if args.fm_overrides else {}
    allowed = {'zeta_obs_a', 'zeta_obs_u', 'zeta_pde', 'clip_threshold', 'guidance_components'}
    if set(fm_overrides) - allowed or (fm_overrides and args.method != 'fm4pde'):
        raise ValueError('Only authorized native guidance hyperparameters may be overridden')
    if args.method == 'ofm' and args.prior != 'ofm':
        raise ValueError('OFM regression is only configured for the operator prior group')
    if not 0 <= args.offset < 100 or not 1 <= args.count <= 100 - args.offset:
        raise ValueError('Evaluation IDs must be contained in 0 through 99')
    indices = args.case_indices if args.case_indices is not None else list(range(args.offset, args.offset + args.count))
    if len(indices) != args.count or len(set(indices)) != len(indices) or any(not 0 <= index < 100 for index in indices):
        raise ValueError('Case indices must be unique, within 0 through 99, and match count')
    if not args.profile and (args.eci_steps != 800 or args.eci_mix != 5 or args.langevin_steps != 100 or args.fm_steps != 100):
        raise ValueError('Reduced schedules are reserved for explicitly marked profiles')
    sys.path.insert(0, str(args.fm4pde))
    from sampling.config import load_config
    from sampling.runner import run_single_ablation

    assets_manifest = json.loads((args.assets / 'manifest.json').read_text())
    source_revisions = {}
    for name, path in (('FM4PDE', args.fm4pde), ('OFM', args.ofm), ('ECI', args.eci)):
        source_revisions[name] = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        changed = subprocess.check_output(['git', '-C', str(path), 'diff', '--name-only', 'HEAD', '--', '*.py'], text=True).strip()
        if changed:
            raise ValueError(f'Official Python sources have uncommitted changes in {name}: {changed}')
    assert source_revisions['FM4PDE'] == assets_manifest['fm4pde_revision']
    assets = assets_manifest['pdes'][args.pde]
    truth_path = args.assets / assets['splits'][args.split]['file']
    assert sha256(truth_path) == assets['splits'][args.split]['sha256']
    saved = torch.load(truth_path, map_location='cpu', weights_only=False)
    data = saved['ground_truth']['pair']
    channels = 1 if args.pde == 'burger' else 2
    assert tuple(data.shape) == (100, channels, 128, 128)
    assert saved['ground_truth']['metadata']['sample_offsets'] == list(range(100))
    checkpoint_digest = sha256(args.checkpoint)
    if args.prior == 'fm4pde':
        assert checkpoint_digest == assets['checkpoint_sha256']
    else:
        source_manifest = json.loads((args.source / 'manifest.json').read_text())
        compact = np.load(args.source / f'{args.split}.npy', mmap_mode='r')
        assert compact.shape == tuple(data.shape)
        assert np.array_equal(np.load(args.source / f'{args.split}_ids.npy'), np.arange(100))
        mean = np.array(source_manifest['stats']['mean'])[None, :, None, None]
        scale = np.array(source_manifest['stats']['std'])[None, :, None, None] / 0.5
        np.testing.assert_allclose(np.asarray(compact, dtype=np.float64) * scale + mean,
                                   data.numpy(), rtol=1e-5, atol=1e-5)
    args.output.mkdir(parents=True, exist_ok=True)
    config_record = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    task = args.task or assets['task']
    config_record.update(checkpoint_sha256=checkpoint_digest,
        truth_sha256=assets['splits'][args.split]['sha256'],
        assets_manifest_sha256=sha256(args.assets / 'manifest.json'),
        source_revisions=source_revisions,
        runtime={'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                 **{name: importlib.metadata.version(name) for name in ('neuraloperator', 'torchcfm', 'gpytorch', 'torchdiffeq')}},
        fm4pde_revision=assets_manifest['fm4pde_revision'],
        task=task, observations=500, observation_noise=0,
        noise_family='iid_standard_Gaussian' if args.prior == 'fm4pde' else 'official_OFM_Matern_GP',
        protocol='native main FM4PDE config; ECI 800x5; OFM Langevin 100; identical observations',
        fm4pde_noise_adapter='initial and stochastic bridge provider' if args.prior == 'ofm' else None,
        guidance_overrides=fm_overrides)
    run_path = args.output / 'run.json'
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        previous.setdefault('eci_batch_size', 1)
        if previous != config_record:
            raise ValueError('Output directory already belongs to a different configuration')
    atomic_json(run_path, config_record)
    coefficient_indices, solution_indices = pair_observation_indices(args.seed)
    np.save(args.output / 'coefficient_observation_indices.npy', coefficient_indices)
    np.save(args.output / 'solution_observation_indices.npy', solution_indices)
    from resource_profile_cache import reuse_profile
    if reuse_profile(args.output, config_record):
        return
    native_slot = None
    if args.method == 'ofm':
        from ofm_resource_guard import acquire_native_slot
        native_slot = acquire_native_slot(args.assets.parent.parent)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    net, normalizer, payload, noise, operator_prior = load_prior(args, channels)
    if args.method == 'ofm':
        from ofm_resource_guard import install_guard
        memory_guard = install_guard(operator_prior)
    if args.method == 'eci':
        FFM, DirichletCondition = load_eci(args.eci)
    atomic_json(args.output / 'model_loaded.json', {
        'selected_inference_weight': payload.get('selected_inference_weight'),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else None,
        'normalizer': {key: value.tolist() if isinstance(value, torch.Tensor) else value
                       for key, value in normalizer.state_dict().items()}})
    if args.method == 'eci' and args.eci_batch_size > 1:
        from eci_batch_runtime import run_eci_batches
        run_eci_batches(args, indices, saved, assets, task, net, normalizer, payload, noise, FFM, DirichletCondition)
        return
    for index in indices:
        result_path = args.output / f'case_{index:03d}.json'
        if result_path.exists():
            continue
        torch.manual_seed(args.seed + index)
        truth = case_ground_truth(saved['ground_truth'], index, args.device)
        masks, _ = common_masks(truth, index, args.seed, task if args.pde != 'burger' else 'inverse')
        config = load_config(args.fm4pde / 'configs/main' / task / f'{args.pde}.yaml', {
            'test_type': args.split, 'data_path': assets['splits'][args.split]['source_file'],
            'checkpoint_path': str(args.checkpoint), 'output_dir': str(args.output / f'native_{index:03d}'),
            'batch_size': 1, 'offset': index, 'device': args.device, 'model_profile': 'auto',
            'sample_seed': args.seed + index, 'mask_seed': args.seed, 'noise_level': 0.0,
            'num_steps': args.fm_steps, 'save_plots': False})
        config.runtime_metadata.update(shared_prior_comparison=config_record,
            common_observation_indices_file=str(args.output / 'solution_observation_indices.npy'))
        for key, value in fm_overrides.items():
            setattr(config, key, value)
        config.runtime_metadata['guidance_overrides'] = fm_overrides
        start = time.monotonic()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        record = {'sample_id': index, 'status': 'ok', 'relative_l2_coefficient': None,
                  'relative_l2_solution': None}
        try:
            if args.method == 'fm4pde':
                with native_noise_provider(noise, enabled=args.prior == 'ofm'), trace_native(
                        args.output / f'trace_{index:03d}.jsonl', args.trace_native):
                    result = run_single_ablation(config, checkpoint_bundle=(net, normalizer, payload),
                                                  ground_truth=truth, observation_masks=masks)
                if result['status'] != 'ok' or result['pde_residual_status'] not in ('reliable', 'approximate'):
                    raise RuntimeError(f"Native FM4PDE residual evaluation did not complete: {result['status']}, {result['pde_residual_status']}")
                artifact = torch.load(Path(result['run_dir']) / 'result.pt', map_location='cpu', weights_only=False)
                prediction = artifact['sol_final'] if channels == 1 else torch.cat(
                    (artifact['coef_final'], artifact['sol_final']), dim=1)
                record['native_run_dir'] = result['run_dir']
                record['pde_residual_status'] = result['pde_residual_status']
            else:
                standardized = normalizer.transform(truth.pair)
                mask = torch.cat((masks.coef.bool(), masks.sol.bool()), dim=1) if channels == 2 else masks.sol.bool()
                if args.method == 'eci':
                    extra = native_model_extra(payload, truth, config)
                    eci = SimpleNamespace(model=ECIVelocityAdapter(net, extra), gp=noise)
                    sampled = FFM.eci_sample(eci, 1, args.eci_steps, args.eci_mix, None,
                        [channels, 128, 128], args.device, DirichletCondition(value=standardized, mask=mask))
                else:
                    sampled = ofm_sample(operator_prior, standardized, mask, args)
                prediction = normalizer.inverse_transform(sampled).detach().cpu()
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            prediction = prediction.numpy().astype(np.float64)
            target = truth.pair.detach().cpu().numpy().astype(np.float64)
            assert prediction.shape == target.shape
            np.save(args.output / f'prediction_{index:03d}.npy', prediction)
            if not np.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite prediction')
            norms = np.linalg.norm(target.reshape(channels, -1), axis=1)
            if np.any(norms == 0):
                raise FloatingPointError('Zero ground-truth norm')
            errors = np.linalg.norm((prediction - target).reshape(channels, -1), axis=1) / norms
            if not np.isfinite(errors).all():
                raise FloatingPointError('Nonfinite relative error')
            record['relative_l2_solution'] = float(errors[-1])
            if channels == 2:
                record['relative_l2_coefficient'] = float(errors[0])
        except (FloatingPointError, RuntimeError, AssertionError) as error:
            numerical = isinstance(error, FloatingPointError) or any(word in str(error).lower()
                for word in ('underflow in dt', 'non-finite', 'nonfinite', 'nan', 'infinite'))
            if not numerical and not isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            record.update(status='failed', error=str(error))
            if isinstance(error, torch.cuda.OutOfMemoryError):
                atomic_json(result_path, record)
                raise
        record.update(seconds=time.monotonic() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else None)
        atomic_json(result_path, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', choices=['fm4pde', 'ofm'], required=True)
    parser.add_argument('--method', choices=['eci', 'ofm', 'fm4pde'], required=True)
    parser.add_argument('--pde', choices=['poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger'], required=True)
    for name in ('fm4pde', 'ofm', 'eci', 'checkpoint', 'assets', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--split', choices=['id', 'smooth', 'rough', 'rough2', 'rough3'], required=True)
    parser.add_argument('--task', choices=['forward', 'both', 'inverse'])
    parser.add_argument('--case-indices', nargs='+', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--eci-steps', type=int, default=800)
    parser.add_argument('--eci-mix', type=int, default=5)
    parser.add_argument('--eci-batch-size', type=int, choices=[0, 1, 2, 4, 8, 10, 16], default=0,
                        help='0 loads the validated runtime batch policy; 1 preserves sequential sampling.')
    parser.add_argument('--fm-steps', type=int, default=100)
    parser.add_argument('--langevin-steps', type=int, default=100)
    parser.add_argument('--hutchinson', type=int, default=1)
    parser.add_argument('--noise-variance', type=float, default=1e-3)
    parser.add_argument('--fm-overrides', type=Path)
    parser.add_argument('--trace-native', action='store_true')
    main(parser.parse_args())
