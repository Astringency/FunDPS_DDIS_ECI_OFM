"""Check completeness and failure accounting in paired comparison tables."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters'))
from summarize_expanded_evaluations import matrix_entries, pairwise_tables


class ComparisonIntegrityTests(unittest.TestCase):
    def test_matrix_rejects_missing_and_duplicate_requested_evaluations(self):
        path = ROOT / 'configs/expanded_evaluation_matrix.json'
        self.assertEqual(len(matrix_entries(path)), 87)
        original = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / 'matrix.json'
            for entries in (original['evaluations'][:-1], original['evaluations'][:-1] + [original['evaluations'][0]]):
                changed = {**original, 'evaluations': entries}
                temporary.write_text(json.dumps(changed))
                with self.assertRaises(AssertionError):
                    matrix_entries(temporary)

    def records(self, pde='poisson'):
        rows = []
        for method, error in [('eci', 0.1), ('fm4pde', 0.2)]:
            for index in range(100):
                rows.append(dict(prior='fm4pde', method=method, pde=pde, split='id', sample_id=index,
                    status='ok', relative_l2_coefficient=None if pde=='burger' else error,
                    relative_l2_solution=error))
        return rows

    def test_failed_case_remains_in_pairing_and_invalidates_all_case_mean(self):
        records = self.records()
        records[17].update(status='failed', relative_l2_coefficient=None, relative_l2_solution=None)
        summaries, cases = pairwise_tables(records)
        row = summaries[0]
        self.assertEqual(len(cases), 100)
        self.assertEqual(row['jointly_successful_cases'], 99)
        self.assertIsNone(row['all_case_mean_delta'])
        self.assertAlmostEqual(row['jointly_successful_case_mean_delta'], -0.1)
        self.assertEqual(row['left_wins'], 99)
        failed = next(case for case in cases if case['sample_id']==17)
        self.assertEqual(failed['status_left'], 'failed')
        self.assertIsNone(failed['delta_left_minus_right'])

    def test_missing_or_duplicate_sample_cannot_pass_pairing(self):
        records = self.records()
        for corrupted in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                pairwise_tables(corrupted)

    def test_burgers_compares_trajectory_without_a_coefficient_metric(self):
        summaries, cases = pairwise_tables(self.records('burger'))
        self.assertEqual(summaries[0]['primary_metric'], 'relative_l2_solution')
        self.assertEqual(summaries[0]['jointly_successful_cases'], 100)
        self.assertAlmostEqual(summaries[0]['all_case_mean_delta'], -0.1)


if __name__ == '__main__':
    unittest.main()
