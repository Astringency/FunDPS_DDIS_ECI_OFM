"""Keep completion, numerical validity, and partial refinement distinct."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
import refine_fm_ofm_strength as study


def row(**kwargs):
    return dict(pde='poisson', task='forward', split='id', verified=100,
        failures=kwargs.get('failures', 0), finite_over_1000_percent=kwargs.get('extremes', 0),
        reaches_fm_fm_mean=kwargs.get('reaches', False), tuned_mean_percent=20.)


class RefinementTest(unittest.TestCase):
    def test_completion_is_not_numerical_success(self):
        value = dict(rows=[row(reaches=True), row(failures=9), row(extremes=1)])
        with patch.object(study, 'PLAN', Path('/nonexistent/refinement-plan.json')):
            result = study.merge_report(value)
        self.assertEqual(result['stable_settings'], 1)
        self.assertEqual(result['numeric_failures'], 9)
        self.assertEqual(result['extreme_finite_samples'], 1)
        self.assertEqual(result['reaches_target_settings'], 1)
        self.assertTrue(all(r['result_round'] == 1 for r in result['rows']))

    def test_partial_retry_cannot_replace_complete_previous_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = dict(groups=[dict(pde='poisson', task='forward', test_splits=['id'])])
            (root / 'plan.json').write_text(json.dumps(plan))
            output = root / 'poisson/forward/test/id/selected'
            output.mkdir(parents=True)
            (output / 'case_000.json').write_text('{}')
            value = dict(rows=[row(failures=9)])
            with patch.object(study, 'PLAN', root / 'plan.json'), patch.object(study, 'STUDY', root):
                result = study.merge_report(copy.deepcopy(value))
                self.assertEqual(result['rows'][0]['failures'], 9)
                self.assertEqual(result['rows'][0]['result_round'], 1)
                self.assertEqual(result['refinement']['verified_settings'], 0)
                (output / 'verified_summary.json').write_text(json.dumps(dict(cases=100, sample_ids=list(range(100)))))
                with self.assertRaises((AssertionError, KeyError)):
                    study.merge_report(copy.deepcopy(value))

    def test_remote_reports_only_merge_assigned_groups_and_do_not_double_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groups = [dict(pde=p, task='forward', test_splits=['id']) for p in ('poisson', 'helmholtz')]
            (root / 'plan.json').write_text(json.dumps(dict(groups=groups)))
            for g in groups:
                folder = root / g['pde'] / g['task']
                folder.mkdir(parents=True)
                (folder / 'worker_error.json').write_text(json.dumps(dict(error=g['pde'])))
            value = dict(rows=[row(), dict(row(), pde='helmholtz')])
            with patch.object(study, 'PLAN', root / 'plan.json'), patch.object(study, 'STUDY', root):
                for host, pde in [('server197', 'poisson'), ('server193', 'helmholtz'), ('server193', 'helmholtz')]:
                    with patch.object(study, 'RUNTIME', dict(host=host, groups=[pde + '/forward'])):
                        value = study.merge_report(value)
            self.assertEqual(len(value['refinement']['errors']), 2)
            self.assertEqual(set(value['refinement']['hosts']), {'server197', 'server193'})
            self.assertEqual(value['round1_rows'], [row(), dict(row(), pde='helmholtz')])


if __name__ == '__main__':
    unittest.main()
