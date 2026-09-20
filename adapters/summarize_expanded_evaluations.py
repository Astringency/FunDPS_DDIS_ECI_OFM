"""Verify and aggregate all 87 requested evaluations, preserving both prior groups."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from summarize_evaluations import checksum, read, selected
from validate_evaluation import verify as verify_original
from validate_shared_prior import validate as verify_shared


PDES = ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger')
SPLITS = ('id', 'smooth', 'rough')


def matrix_entries(path):
    matrix = read(path)
    expected = {(prior, method, pde, split)
        for prior, methods in [('fm4pde', ('eci', 'fm4pde')), ('ofm', ('eci', 'ofm', 'fm4pde'))]
        for method, pde, split in itertools.product(methods, PDES, SPLITS)}
    expected.update((method, method, pde, split) for method, pde, split in
                    itertools.product(('ddis', 'fundps'), PDES[:2], SPLITS))
    entries = matrix['evaluations']
    actual = [(item['prior'], item['method'], item['pde'], item['split']) for item in entries]
    assert len(actual) == len(set(actual)) == 87 and set(actual) == expected
    assert matrix['cases_per_split'] == 100 and matrix['observations'] == 500 and matrix['noise'] == 0
    for item in entries:
        legacy = item['pde'] in PDES[:2] and item['method'] != 'fm4pde' and item['prior'] != 'fm4pde'
        assert item['existing_controller'] == legacy
    return entries


def locations(root, item):
    prior, method, pde, split = (item[key] for key in ('prior', 'method', 'pde', 'split'))
    if item['existing_controller']:
        job = root / 'jobs' / f'evaluate_{method}_{pde}_{split}'
        return job, root / 'evaluation' / method / pde / split, job.with_suffix('.exit')
    job = root / 'jobs' / f'shared_{prior}_{method}_{pde}_{split}'
    return job, root / 'evaluation_shared_prior' / prior / method / pde / split, job / 'controller.exit'


def await_results(root, entries, wait):
    while True:
        missing = []
        for item in entries:
            job, _, exit_path = locations(root, item)
            if not (job / 'evaluation_completed').is_file():
                missing.append(job.name)
                if exit_path.exists() and exit_path.read_text().strip() != '0':
                    raise RuntimeError(f'Evaluation failed; inspect its saved log: {job}')
        if not missing:
            return
        if not wait:
            raise FileNotFoundError(f'{len(missing)}/87 evaluations pending: {missing}')
        print(f'Waiting for {len(missing)}/87 evaluations', flush=True)
        time.sleep(60)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pairwise_tables(cases):
    grouped = {}
    for row in cases:
        if row['prior'] not in ('fm4pde', 'ofm'):
            continue
        key = (row['prior'], row['pde'], row['split'])
        records = grouped.setdefault(key, {}).setdefault(row['method'], {})
        if row['sample_id'] in records:
            raise ValueError(f'Duplicate case: {key}, {row["method"]}, {row["sample_id"]}')
        records[row['sample_id']] = row
    rows, paired_cases = [], []
    for (prior, pde, split), methods in sorted(grouped.items()):
        expected_methods = {'eci', 'fm4pde'} if prior == 'fm4pde' else {'eci', 'ofm', 'fm4pde'}
        if set(methods) != expected_methods or any(set(records) != set(range(100)) for records in methods.values()):
            raise ValueError(f'Incomplete paired comparison: {prior}/{pde}/{split}')
        metric = 'relative_l2_solution' if pde == 'burger' else 'relative_l2_coefficient'
        for left, right in itertools.combinations(sorted(methods), 2):
            deltas = []
            left_wins = right_wins = ties = 0
            for index in range(100):
                a, b = methods[left][index], methods[right][index]
                both_ok = a['status'] == b['status'] == 'ok'
                delta = a[metric] - b[metric] if both_ok else None
                if both_ok:
                    assert np.isfinite(delta)
                    deltas.append(delta)
                    left_wins += delta < -1e-12
                    right_wins += delta > 1e-12
                    ties += abs(delta) <= 1e-12
                paired_cases.append(dict(prior=prior, pde=pde, split=split, method_left=left,
                    method_right=right, sample_id=index, primary_metric=metric,
                    status_left=a['status'], status_right=b['status'],
                    error_left=a[metric], error_right=b[metric], delta_left_minus_right=delta))
            conditional = float(np.mean(deltas)) if deltas else None
            rows.append(dict(prior=prior, pde=pde, split=split, method_left=left, method_right=right,
                primary_metric=metric, total_cases=100, jointly_successful_cases=len(deltas),
                left_wins=int(left_wins), right_wins=int(right_wins), ties=int(ties),
                all_case_mean_delta=conditional if len(deltas) == 100 else None,
                jointly_successful_case_mean_delta=conditional))
    return rows, paired_cases


def summarize(root, destination, entries):
    assets_root = root / 'data/shared_prior_assets'
    assets = read(assets_root / 'manifest.json')
    assert read(assets_root / 'verified.json')['manifest_sha256'] == checksum(assets_root / 'manifest.json')
    truth_roundoff, masks_by_data, weights_by_group = {}, {}, {}
    rows, cases = [], []
    for item in entries:
        prior, method, pde, split = (item[key] for key in ('prior', 'method', 'pde', 'split'))
        job, output, exit_path = locations(root, item)
        assert (job / 'evaluation_completed').is_file()
        if exit_path.exists():
            assert exit_path.read_text().strip() == '0'
        run = read(output / 'run.json')
        assert run['method'] == method and run['split'] == split and run['count'] == 100
        assert run.get('offset', 0) == 0 and not run.get('profile', False) and not run.get('profile_only', False)
        source = root / 'data/compact' / pde
        if prior == 'fm4pde':
            checkpoint = (assets_root / assets['pdes'][pde]['checkpoint_file']).resolve(strict=True)
            assert checksum(checkpoint) == assets['pdes'][pde]['checkpoint_sha256']
        else:
            best, checkpoint = selected(root, 'flow' if prior == 'ofm' else method, pde)
            assert read(job / 'prior_selection.json') == best
        used = run['official_config']['pkl_path'] if method in ('ddis', 'fundps') else run['checkpoint']
        assert Path(used).resolve(strict=True) == checkpoint
        checkpoint_hash = checksum(checkpoint)
        assert run['checkpoint_sha256'] == checkpoint_hash
        group_key = (prior, pde)
        assert weights_by_group.setdefault(group_key, checkpoint_hash) == checkpoint_hash
        surrogate_hash = ''
        if method == 'ddis':
            surrogate_best, surrogate = selected(root, 'surrogate', pde)
            assert read(job / 'surrogate_selection.json') == surrogate_best
            assert Path(run['official_config']['surrogate_path']).resolve(strict=True) == surrogate
            surrogate_hash = checksum(surrogate)
            assert run['surrogate_sha256'] == surrogate_hash
        if item['existing_controller']:
            assert Path(run['source']).resolve() == source.resolve()
            verify_original(output, source, split, 100)
        else:
            assert run['prior'] == prior and run['pde'] == pde
            assert Path(run['assets']).resolve() == assets_root.resolve()
            if prior == 'ofm':
                assert Path(run['source']).resolve() == source.resolve()
            verify_shared(output)
        mask_hash = checksum(output / 'solution_observation_indices.npy')
        data_key = (pde, split)
        assert masks_by_data.setdefault(data_key, mask_hash) == mask_hash
        if data_key not in truth_roundoff:
            truth_entry = assets['pdes'][pde]['splits'][split]
            truth_path = assets_root / truth_entry['file']
            assert checksum(truth_path) == truth_entry['sha256']
            physical = torch.load(truth_path, map_location='cpu', weights_only=False)['ground_truth']['pair'].numpy().astype(np.float64)
            compact = np.load(source / f'{split}.npy', mmap_mode='r')
            manifest = read(source / 'manifest.json')
            assert checksum(source / f'{split}.npy') == manifest['outputs'][split]['sha256']
            assert np.array_equal(np.load(source / f'{split}_ids.npy'), np.arange(100))
            assert physical.shape == compact.shape
            mean = np.asarray(manifest['stats']['mean'])[None, :, None, None]
            scale = np.asarray(manifest['stats']['std'])[None, :, None, None] / 0.5
            error = np.max(np.abs(np.asarray(compact, dtype=np.float64) * scale + mean - physical), axis=(0, 2, 3))
            tolerance = 32 * np.finfo(np.float32).eps * np.maximum(np.max(np.abs(physical), axis=(0, 2, 3)), 1e-12)
            assert np.all(error <= tolerance), f'Physical truth mismatch: {data_key}, {error}, {tolerance}'
            truth_roundoff[data_key] = error.tolist()
        records = [read(output / f'case_{index:03d}.json') for index in range(100)]
        failed_ids = [record['sample_id'] for record in records if record['status'] == 'failed']
        for record in records:
            cases.append(dict(prior=prior, method=method, pde=pde, split=split,
                sample_id=record['sample_id'], status=record['status'],
                relative_l2_coefficient=record['relative_l2_coefficient'],
                relative_l2_solution=record['relative_l2_solution'],
                seconds=record.get('seconds'), error=record.get('error', '')))
        row = dict(prior=prior, method=method, pde=pde, split=split, cases=100,
            successful_cases=100-len(failed_ids), failed_cases=len(failed_ids),
            failure_rate=len(failed_ids)/100, failed_sample_ids=';'.join(map(str,failed_ids)))
        for field in ('coefficient', 'solution'):
            values = [record['relative_l2_' + field] for record in records if record['status'] == 'ok'
                      and record['relative_l2_' + field] is not None]
            row['mean_relative_l2_all_cases_' + field] = float(np.mean(values)) if len(values) == 100 else None
            row['mean_relative_l2_successful_cases_' + field] = float(np.mean(values)) if values else None
            row['median_relative_l2_successful_cases_' + field] = float(np.median(values)) if values else None
        if method in ('ddis', 'fundps'):
            timing = read(output / 'completed.json')
            seconds, peak = timing['seconds'], timing['peak_reserved_bytes']
            scope = 'whole generation call including setup and output I/O'
        else:
            seconds = sum(record['seconds'] for record in records)
            peak = max(record['peak_reserved_bytes'] for record in records)
            scope = 'sum of case sampling times excluding model setup'
        row.update(seconds=seconds, seconds_per_case=seconds/100, timing_scope=scope,
            peak_reserved_bytes=peak, gpu_index=int((job/'gpu_index').read_text()),
            prior_checkpoint_sha256=checkpoint_hash, surrogate_checkpoint_sha256=surrogate_hash,
            mask_sha256=mask_hash, result_directory=str(output))
        rows.append(row)
    assert len(rows) == 87 and len(cases) == 8700
    paired, paired_cases = pairwise_tables(cases)
    assert len(paired) == 60 and len(paired_cases) == 6000
    destination.mkdir(parents=True, exist_ok=False)
    for name, records in [('summary', rows), ('cases', cases), ('pairwise', paired), ('pairwise_cases', paired_cases)]:
        write_csv(destination / f'{name}.csv', records)
    result = {'created_utc':datetime.now(timezone.utc).isoformat(), 'verified_splits':87,
        'verified_cases':8700, 'paired_comparisons':60, 'relative_error_units':'physical-field ratio, uncapped',
        'physical_truth_roundoff_max_abs':{f'{pde}/{split}':value for (pde,split),value in truth_roundoff.items()},
        'notes':['Each prior/PDE group uses exactly one checkpoint across all its samplers and splits.',
                 'All groups use identical case IDs and 500 noiseless observation locations.',
                 'Burgers has one trajectory channel; its coefficient metric is inapplicable.',
                 'All-case means are null if any required case fails. Conditional means remain explicitly labeled.',
                 'The existing compact and native physical representations differ only by recorded float32 roundoff.',
                 'Shared-GPU timing is not a controlled speed benchmark; setup boundaries differ by method.'],
        'results':rows, 'paired_results':paired}
    (destination/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (destination/'completed').write_text(result['created_utc']+'\n')
    print(f'Verified 87 splits / 8700 cases: {destination}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--matrix',type=Path,default=Path(__file__).resolve().parents[1]/'configs/expanded_evaluation_matrix.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    entries=matrix_entries(args.matrix)
    await_results(args.root,entries,args.wait)
    summarize(args.root,args.output,entries)
