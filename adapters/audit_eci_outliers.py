"""Independently audit the user-requested eleven ECI-OFM outlier retries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from validate_shared_prior import validate


IDS = {'id': [6, 8, 13, 15, 66, 80], 'smooth': [12, 13, 17, 55, 80]}
VARIANT = 'mix1_800x1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    destination = root / 'diagnostics/eci_ofm_outliers_20260922'
    groups, comparison = [], []
    for split, ids in IDS.items():
        folder = destination / split
        run = json.loads((folder / 'run.json').read_text())
        assert run['ids'] == ids and run['variants'] == {VARIANT: [800, 1, None, True]}
        assert run['split'] == split and run['task'] == 'inverse' and run['pde'] == 'poisson'
        assert run['observation_count'] == 500 and run['seed_policy'] == 'same original per-case seed: 0 + sample_id'
        assert json.loads((folder / 'completed.json').read_text())[VARIANT]['failures'] == 0
        assert sha(Path(run['checkpoint'])) == run['checkpoint_sha256']
        originals, locations = {}, {}
        shards = root / 'evaluation_v2/ofm/eci/poisson/inverse' / split
        # Recompute all original errors so the unchanged complement is audited too.
        for shard in sorted(shards.glob('shard_*')):
            old_run = json.loads((shard / 'run.json').read_text())
            assert old_run['checkpoint_sha256'] == run['checkpoint_sha256']
            assert old_run['eci_steps'] == 800 and old_run['eci_mix'] == 5 and old_run['seed'] == 0
            assert old_run['observations'] == 500 and old_run['observation_noise'] == 0
            assert {k.lower(): v for k, v in old_run['source_revisions'].items()} == run['source_revisions']
            assert validate(shard)['failures'] == 0
            for path in shard.glob('case_*.json'):
                record = json.loads(path.read_text())
                assert record['sample_id'] not in originals
                originals[record['sample_id']] = record
                locations[record['sample_id']] = path
        assert sorted(originals) == list(range(100))
        assert sorted(i for i, r in originals.items() if r['relative_l2_coefficient'] > 10) == ids
        assets = Path(old_run['assets'])
        manifest = json.loads((assets / 'manifest.json').read_text())
        truth_path = assets / manifest['pdes']['poisson']['splits'][split]['file']
        assert sha(truth_path) == run['source_truth_sha256'] == old_run['truth_sha256']
        truth = torch.load(truth_path, map_location='cpu', weights_only=False)['ground_truth']['pair'].numpy().astype(np.float64)
        rng = np.random.RandomState(0)
        observations = []
        for _ in range(100):
            rng.choice(16384, 500, replace=False)
            observations.append(rng.choice(16384, 500, replace=False))
        records = []
        for index in ids:
            source = folder / VARIANT / f'case_{index:03d}.json'
            prediction_path = folder / VARIANT / f'prediction_{index:03d}.npy'
            record = json.loads(source.read_text())
            assert record['sample_id'] == index and record['split'] == split and record['status'] == 'ok'
            assert (record['steps'], record['mix'], record['resample_step'], record['constrained']) == (800, 1, None, True)
            assert record['calls'] == 800
            if 'initial_noise_sha256' in originals[index]:
                assert record['initial_noise_sha256'] == originals[index]['initial_noise_sha256']
            predicted = np.load(prediction_path)
            actual = truth[index:index + 1]
            assert predicted.shape == actual.shape and np.isfinite(predicted).all()
            errors = np.sqrt(np.sum((predicted - actual) ** 2, axis=(0, 2, 3)) / np.sum(actual ** 2, axis=(0, 2, 3)))
            np.testing.assert_allclose([record['relative_l2_coefficient'], record['relative_l2_solution']], errors, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(predicted[0, 1].reshape(-1)[observations[index]],
                actual[0, 1].reshape(-1)[observations[index]], rtol=1e-5, atol=1e-6)
            assert 0 <= errors[0] < 10
            records.append(dict(case=record, source=str(source.relative_to(root)),
                prediction=str(prediction_path.relative_to(root)), prediction_sha256=sha(prediction_path),
                original=originals[index], original_source=str(locations[index].relative_to(root))))
            comparison.append(dict(split=split, sample_id=index,
                original_coefficient_percent=originals[index]['relative_l2_coefficient'] * 100,
                repaired_coefficient_percent=float(errors[0] * 100),
                original_mix=5, repaired_mix=1, steps=800))
        final_errors = [next((r['case']['relative_l2_coefficient'] for r in records if r['case']['sample_id'] == i),
            originals[i]['relative_l2_coefficient']) for i in range(100)]
        validation = dict(verified_predictions=len(ids), failed=0, final_verified_count=100,
            final_mean_percent=float(np.mean(final_errors) * 100),
            final_max_percent=float(max(final_errors) * 100), final_over_1000_count=sum(x > 10 for x in final_errors))
        groups.append(dict(method='ECI-OFM', pde='poisson', task='inverse', split=split,
            sample_ids=ids, repair_scope='selected_samples', run=run, records=records, validation=validation,
            untouched_case_sha256={str(locations[i].relative_to(root)): sha(locations[i]) for i in range(100) if i not in ids}))
    report = destination / 'report'
    report.mkdir(exist_ok=True)
    audit = dict(groups=groups, note='User-requested test-outlier parameter tuning; only the eleven specified cases were rerun. No metric clipping. Remaining 189 records unchanged.')
    temporary = report / 'audit.partial.json'
    temporary.write_text(json.dumps(audit, indent=2, allow_nan=False))
    temporary.replace(report / 'audit.json')
    with (report / 'repaired_cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps([dict(split=g['split'], **g['validation']) for g in groups]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
