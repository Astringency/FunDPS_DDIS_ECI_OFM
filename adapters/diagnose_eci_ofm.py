"""Trace/re-evaluate unchanged official ECI with explicit sampling parameters.

Outputs live under diagnostics, separate from the original formal evaluations.
No model, correction rule, or update equation is replaced by this adapter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from shared_prior_runtime import ECIVelocityAdapter, case_ground_truth, common_masks, load_eci, load_prior
from eci_batch_runtime import FixedInitialNoise, atomic_json

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
VARIANTS = {
    'baseline_800x5': (800, 5, None, True),
    'mix1_800x1': (800, 1, None, True),
    'steps200_200x5': (200, 5, None, True),
    'steps200_mix1': (200, 1, None, True),
    'notebook_boundary_200x1_resample5': (200, 1, 5, True),
    'unconstrained_800x5': (800, 5, None, False),
}


class TracedVelocity:
    def __init__(self, model, mix):
        self.model, self.mix = model, mix
        self.calls, self.trace = 0, []
        self.peak_x = self.peak_v = 0.
        self.level = 2

    def __call__(self, t, x):
        self.calls += 1
        v = self.model(t, x)
        xm, vm = float(x.abs().max()), float(v.abs().max())
        self.peak_x = max(self.peak_x, xm)
        self.peak_v = max(self.peak_v, vm)
        finite = bool(torch.isfinite(x).all() and torch.isfinite(v).all())
        crossing = math_level(max(xm, vm)) >= self.level
        if self.calls == 1 or self.calls % 100 == 0 or crossing or not finite:
            self.trace.append(dict(call=self.calls, step=(self.calls-1)//self.mix,
                mixing_iteration=(self.calls-1)%self.mix, time=float(t),
                x_max_abs=xm if np.isfinite(xm) else None,
                velocity_max_abs=vm if np.isfinite(vm) else None, finite=finite))
        if crossing:
            self.level = math_level(max(xm, vm)) + 2
        if not finite:
            raise FloatingPointError(f'Nonfinite model state/velocity at call {self.calls}, t={float(t)}')
        return v


def math_level(value):
    return int(np.log10(value)) if np.isfinite(value) and value > 0 else 1000


def main(args):
    reference = OUT / 'evaluation_v2/ofm/eci/poisson/inverse/rough/shard_020/run.json'
    original = json.loads(reference.read_text())
    setup = SimpleNamespace(**original)
    for key in ('fm4pde', 'ofm', 'eci', 'checkpoint', 'assets', 'output', 'source'):
        setattr(setup, key, Path(getattr(setup, key)))
    setup.device = 'cuda'
    if args.output.exists() and (args.output / 'completed.json').exists():
        raise ValueError('This diagnostic run is already complete; select a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    revisions = {}
    for name in ('fm4pde', 'ofm', 'eci'):
        repo = getattr(setup, name)
        changed = subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only', 'HEAD', '--', '*.py'], text=True).strip()
        assert not changed, (name, changed)
        revisions[name] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert hashlib.sha256(setup.checkpoint.read_bytes()).hexdigest() == original['checkpoint_sha256']
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net, normalizer, payload, noise, prior = load_prior(setup, 2)
    FFM, Constraint = load_eci(setup.eci)
    from shared_prior_official_eci.constraints import NoneConstraint
    assets = json.loads((setup.assets / 'manifest.json').read_text())
    path = setup.assets / assets['pdes']['poisson']['splits'][args.split]['file']
    saved = torch.load(path, map_location='cpu', weights_only=False)['ground_truth']
    ids = list(range(100)) if args.all_cases else args.ids
    variants = args.variants or list(VARIANTS)
    atomic_json(args.output / 'run.json', dict(checkpoint=str(setup.checkpoint),
        checkpoint_sha256=original['checkpoint_sha256'], source_revisions=revisions,
        orchestration_revision=subprocess.check_output(['git', '-C', str(BASE / 'orchestration'), 'rev-parse', 'HEAD'], text=True).strip(),
        split=args.split, task='inverse', pde='poisson', ids=ids,
        variants={key: VARIANTS[key] for key in variants}, batch_size=1,
        observation_count=500, seed_policy='same original per-case seed: 0 + sample_id',
        original_reference=str(reference), source_truth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        note='Diagnostic parameter comparison. Official FFM.eci_sample and DirichletCondition unchanged. Test-case tuning, not held-out evaluation.'))
    results = []
    for variant in variants:
        steps, mix, resample, constrained = VARIANTS[variant]
        directory = args.output / variant
        directory.mkdir(exist_ok=True)
        for index in ids:
            truth = case_ground_truth(saved, index, 'cuda')
            masks, _ = common_masks(truth, index, 0, 'inverse')
            mask = torch.cat((masks.coef.bool(), masks.sol.bool()), dim=1)
            target = normalizer.transform(truth.pair)
            torch.manual_seed(index)
            # Draw once with the original official GP. Reset for configurations
            # that use native resampling so their first draw stays identical.
            initial = noise.sample(None, [2, 128, 128], n_samples=1)
            initial_hash = hashlib.sha256(initial.detach().cpu().numpy().tobytes()).hexdigest()
            torch.manual_seed(index)
            traced = TracedVelocity(ECIVelocityAdapter(net, None), mix)
            instance = SimpleNamespace(model=traced, gp=noise if resample else FixedInitialNoise(initial))
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            record = dict(variant=variant, sample_id=index, split=args.split,
                initial_noise_sha256=initial_hash, steps=steps, mix=mix, resample_step=resample,
                constrained=constrained, status='ok')
            try:
                sampled = FFM.eci_sample(instance, 1, steps, mix, resample,
                    [2, 128, 128], 'cuda', Constraint(value=target, mask=mask) if constrained else NoneConstraint())
                prediction = normalizer.inverse_transform(sampled).detach().cpu().numpy().astype(np.float64)
                actual = truth.pair.detach().cpu().numpy().astype(np.float64)
                assert prediction.shape == actual.shape
                errors = np.linalg.norm((prediction-actual).reshape(2,-1),axis=1)/np.linalg.norm(actual.reshape(2,-1),axis=1)
                if not np.isfinite(errors).all() or not np.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite final prediction/error')
                np.save(directory / f'prediction_{index:03d}.npy', prediction)
                record.update(relative_l2_coefficient=float(errors[0]), relative_l2_solution=float(errors[1]),
                    observed_max_abs_error=float((sampled-target)[mask].abs().max()) if constrained else None,
                    prediction_max_abs=float(np.abs(prediction).max()))
            except FloatingPointError as error:
                record.update(status='failed', error=str(error), relative_l2_coefficient=None, relative_l2_solution=None)
            torch.cuda.synchronize()
            record.update(seconds=time.monotonic()-start, calls=traced.calls,
                peak_state_max_abs=traced.peak_x if np.isfinite(traced.peak_x) else None,
                peak_velocity_max_abs=traced.peak_v if np.isfinite(traced.peak_v) else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), trace=traced.trace)
            atomic_json(directory / f'case_{index:03d}.json', record)
            results.append({k:v for k,v in record.items() if k != 'trace'})
            atomic_json(args.output / 'results.json', results)
            print(json.dumps(results[-1], allow_nan=False), flush=True)
    summaries = {}
    for variant in variants:
        records = [r for r in results if r['variant']==variant]
        values = [r['relative_l2_coefficient'] for r in records if r['status']=='ok']
        summaries[variant] = dict(count=len(records), failures=len(records)-len(values),
            finite_mean_relative_l2_coefficient=float(np.mean(values)) if values else None,
            finite_median_relative_l2_coefficient=float(np.median(values)) if values else None,
            finite_max_relative_l2_coefficient=max(values) if values else None)
    atomic_json(args.output / 'completed.json', summaries)
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['id', 'smooth', 'rough'], default='rough')
    parser.add_argument('--ids', type=int, nargs='+', default=[27, 78, 80, 0])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS))
    parser.add_argument('--all-cases', action='store_true')
    main(parser.parse_args())
