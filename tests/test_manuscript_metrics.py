"""Prevent line wrapping, column order, and step counts from changing paper metrics."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from manuscript_metrics import parse_paper


def manuscript(forward_n='100', inverse_n='100', burger_n='100', reorder=False):
    blocks = []
    for task, count in [('forward', forward_n), ('inverse', inverse_n)]:
        fields = ['PDE', 'Distribution', r'\shortstack{DiffusionPDE\\1,000 steps}', 'FM4PDE']
        if reorder:
            fields = [fields[3], fields[0], fields[2], fields[1]]
        rows = [' & '.join(fields) + r' \\']
        for pde in ['Poisson', 'Helmholtz', 'Darcy', 'Navier--Stokes']:
            for split in ['ID', 'Smooth', 'Rough']:
                values = {'PDE': r'\multirow{3}{*}{' + pde + '}' if split == 'ID' else '',
                    'Distribution': split, r'\shortstack{DiffusionPDE\\1,000 steps}': r'$999.0\pm1.0$',
                    'FM4PDE': r'$\mathbf{11.25}^{\dagger}\pm0.75$'}
                # Each logical row spans multiple physical source lines.
                rows.append('\n & '.join(values[f] for f in fields) + r' \\')
        blocks.append('\n'.join([r'\begin{table}',
            r'\caption{Sparse reconstruction on the first ' + count + r' test inputs: errors (\%).}',
            r'\label{tab:sparse-' + task + '-results}', r'\begin{tabular}{@{}cccc@{}}',
            r'\toprule', *rows, r'\end{tabular}', r'\end{table}']))
    fields = ['Observations', 'Method', 'ID', 'Smooth', 'Rough']
    if reorder:
        fields = ['Method', 'Rough', 'Observations', 'Smooth', 'ID']
    rows = [' & '.join(fields) + r' \\']
    for obs in ['Random', 'Structured']:
        rows.append(' & '.join({'Observations': r'\multirow{2}{*}{' + obs + '}', 'Method': 'CoCoGen',
            'ID': '--', 'Smooth': '--', 'Rough': '--'}[f] for f in fields) + r' \\')
        values = {'Observations': '', 'Method': 'FM4PDE (100 steps)',
            'ID': r'$\mathbf{.32}\pm.25$', 'Smooth': r'$2.4e-1\pm.13$',
            'Rough': r'$\mathbf{2.91}^{1}\pm2.08$' if obs == 'Random' else r'$99\pm2$'}
        rows.append(' &\n'.join(values[f] for f in fields) + r' \\')
    blocks.append('\n'.join([r'\begin{table}',
        r'\caption{Burgers over ' + burger_n + r' test cases: error (\%).}',
        r'\label{tab:burgers-results}', r'\begin{tabular}{ccccc}', *rows,
        r'\end{tabular}', r'\end{table}']))
    return '\n'.join(blocks)


class ManuscriptMetricsTest(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paper.tex'
            path.write_text(text)
            return parse_paper(path)

    def test_wrapped_random_row_and_nested_header_breaks(self):
        text = manuscript()
        rows = self.parse(text)
        self.assertEqual(len(rows), 27)
        rough = next(r for r in rows if r['pde'] == 'burger' and r['split'] == 'rough')
        self.assertEqual((rough['mean_percent'], rough['finite_std_percent']), (2.91, 2.08))
        self.assertIn('2.91', text.splitlines()[rough['source_line'] - 1])
        self.assertEqual({r['n'] for r in rows}, {100})

    def test_caption_counts_not_diffusion_steps_and_not_hardcoded(self):
        rows = self.parse(manuscript(inverse_n='1,000', burger_n='1000'))
        for r in rows:
            self.assertEqual(r['n'], 100 if r['task'] == 'forward' else 1000)
            self.assertEqual(r['n'], r['expected_n'])

    def test_reordered_columns_have_identical_metrics(self):
        def metrics(rows):
            return {(r['pde'], r['task'], r['split']): (r['mean_percent'], r['finite_std_percent'], r['n']) for r in rows}
        self.assertEqual(metrics(self.parse(manuscript())), metrics(self.parse(manuscript(reorder=True))))

    def test_commented_old_tables_are_not_selected(self):
        old = '\n'.join('% ' + line for line in manuscript(forward_n='1000').splitlines())
        rows = self.parse(old + '\n' + manuscript())
        self.assertEqual(len(rows), 27)
        self.assertEqual({r['n'] for r in rows}, {100})

    def test_missing_metric_reports_setting_and_source_location(self):
        text = manuscript().replace(r'$\mathbf{2.91}^{1}\pm2.08$', '--')
        with self.assertRaisesRegex(ValueError, r'paper.tex:\d+: tab:burgers-results, FM4PDE burger/both/rough'):
            self.parse(text)

    def test_absent_caption_count_does_not_use_1000_steps_header(self):
        text = manuscript().replace('first 100 test inputs', 'test inputs')
        with self.assertRaisesRegex(ValueError, 'sample count in caption'):
            self.parse(text)

    def test_duplicate_active_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'expected one active table'):
            self.parse(manuscript() + manuscript())


if __name__ == '__main__':
    unittest.main()
