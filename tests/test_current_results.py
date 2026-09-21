"""Protect partial-result and numerical-failure accounting in the live export."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from summarize_current_results import aggregate
from verified_repair_results import apply_repairs


def snapshot(ids=range(100), failed=(), verified=True, worker_failure=False):
    key = 'evaluation_v2/ofm/eci/poisson/inverse/id/shard_000'
    cases = [dict(method='ECI-OFM', pde='poisson', task='inverse', split='id',
        run_key=key, sample_id=i, status='failed' if i in failed else 'ok',
        relative_l2_coefficient=None if i in failed else (1e20 if i == 0 else .5),
        relative_l2_solution=None if i in failed else .25) for i in ids]
    ok = [c for c in cases if c['status'] == 'ok']
    meta = dict(method='ECI-OFM', pde='poisson', task='inverse', split='id')
    if verified:
        meta['verified'] = dict(sample_ids=list(ids), cases=len(cases), successes=len(ok),
            failures=len(cases)-len(ok), successful_case_mean_relative_l2_coefficient=
            sum(c['relative_l2_coefficient'] for c in ok)/len(ok) if ok else None)
    if worker_failure:
        meta['worker_failure'] = {'error': 'OFM_RESOURCE_LIMIT'}
    return dict(runs={key: meta}, cases=cases)


def target(value):
    return next(r for r in aggregate(value) if (r['method'], r['pde'], r['task'], r['split']) ==
                ('ECI-OFM', 'poisson', 'inverse', 'id'))


def add_repair(value):
    repaired = snapshot()
    meta = next(iter(repaired['runs'].values()))
    meta.update(result_version='repair_test', sampling_parameters={'steps': 200})
    meta['verified']['successful_case_mean_relative_l2_coefficient'] = .8
    key = 'diagnostics/test_repair'
    for row in repaired['cases']:
        row.update(run_key=key, relative_l2_coefficient=.8)
    value['repairs'] = [dict(run_key=key, meta=meta, cases=repaired['cases'])]
    return value


class CurrentResultsTest(unittest.TestCase):
    def test_complete_large_finite_values_are_not_capped(self):
        row = target(snapshot())
        self.assertEqual(row['status'], 'complete')
        self.assertEqual(row['n_verified'], 100)
        self.assertAlmostEqual(row['mean_percent'], 1e20)
        self.assertEqual(row['finite_over_1000_percent'], 1)

    def test_partial_verified_results_do_not_become_full_means(self):
        row = target(snapshot(range(10)))
        self.assertEqual(row['status'], 'partial')
        self.assertEqual(row['n'], 10)
        self.assertEqual(row['n_pending'], 90)
        self.assertIsNone(row['mean_percent'])
        self.assertIsNotNone(row['finite_mean_percent'])

    def test_unverified_and_resource_errors_do_not_become_numeric_failures(self):
        row = target(snapshot(range(5), verified=False, worker_failure=True))
        self.assertEqual((row['n_saved'], row['n_verified'], row['n_failed']), (5, 0, 0))
        self.assertEqual(row['worker_failure_runs'], 1)
        self.assertIsNone(row['finite_mean_percent'])

    def test_numeric_failures_preserve_denominator(self):
        row = target(snapshot(failed=(99,)))
        self.assertEqual(row['status'], 'complete_with_failures')
        self.assertEqual((row['n'], row['n_finite'], row['n_failed']), (100, 99, 1))
        self.assertIsNone(row['mean_percent'])
        self.assertIsNotNone(row['finite_mean_percent'])
        empty = target(snapshot(range(1), failed=(0,)))
        self.assertIsNone(empty['finite_mean_percent'])

    def test_duplicate_or_inconsistent_verified_records_rejected(self):
        value = snapshot()
        value['cases'].append(copy.deepcopy(value['cases'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate(value)
        value = snapshot()
        value['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'IDs disagree'):
            aggregate(value)
        value = snapshot()
        value['cases'][0]['relative_l2_coefficient'] = 1.0
        with self.assertRaisesRegex(ValueError, 'mean disagrees'):
            aggregate(value)

    def test_never_started_cells_have_no_statistics(self):
        rows = aggregate({'cases': [], 'runs': {}})
        self.assertEqual(len(rows), 132)
        self.assertTrue(all(r['status'] == 'not_started' and r['finite_mean_percent'] is None for r in rows))

    def test_repair_replaces_whole_setting_and_preserves_original_failures(self):
        value = add_repair(snapshot(failed=(99,)))
        untouched = copy.deepcopy(value)
        row = target(apply_repairs(value))
        self.assertEqual((row['n_saved'], row['n_verified'], row['n_failed']), (100, 100, 0))
        # Use all repaired values, including those worse than their originals.
        self.assertAlmostEqual(row['mean_percent'], 80)
        self.assertEqual(row['result_version'], 'repair_test')
        self.assertEqual(target(value)['n_failed'], 1)
        self.assertEqual(value, untouched)

    def test_partial_duplicate_and_cross_setting_repairs_are_rejected(self):
        value = add_repair(snapshot())
        value['repairs'][0]['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            apply_repairs(value)
        value = add_repair(snapshot())
        value['repairs'].append(copy.deepcopy(value['repairs'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            apply_repairs(value)
        value = add_repair(snapshot())
        value['repairs'][0]['cases'][0]['split'] = 'rough'
        with self.assertRaisesRegex(ValueError, 'wrong setting'):
            apply_repairs(value)

    def test_repair_still_requires_consistent_verification(self):
        value = add_repair(snapshot())
        value['repairs'][0]['cases'][0]['relative_l2_coefficient'] = 100
        with self.assertRaisesRegex(ValueError, 'mean disagrees'):
            aggregate(apply_repairs(value))


if __name__ == '__main__':
    unittest.main()
