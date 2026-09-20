"""Audit paired two-case outputs and render reconstruction comparisons."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ('ddis', 'fundps', 'ofm', 'eci_fm', 'eci_ofm', 'fm_fm', 'fm_ofm')
TASKS = ('inverse', 'forward', 'both')
SPLITS = ('id', 'smooth', 'rough', 'rough2', 'rough3')
FIELDS = {'inverse': (('coefficient', 0),), 'forward': (('solution', 1),),
          'both': (('coefficient', 0), ('solution', 1))}


def score(errors, task):
    values = [errors['relative_l2_' + name] for name, _ in FIELDS[task]]
    return sum(values) / len(values) if all(value is not None for value in values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    import torch

    root = args.root
    plan = json.loads((root / 'plan.json').read_text())
    selection = plan['selected']['cases']
    rows, missing = [], []
    rng = np.random.RandomState(0)
    observation_indices = [(rng.choice(128 * 128, 500, replace=False),
                            rng.choice(128 * 128, 500, replace=False)) for _ in range(100)]
    truths = {}
    for split in SPLITS:
        saved = torch.load(root / 'assets' / 'truth' / 'poisson' / (split + '.pt'),
                           map_location='cpu', weights_only=False)
        truths[split] = saved['ground_truth']['pair'].numpy().astype(np.float64)
    if args.render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (root / 'figures').mkdir(exist_ok=True)
    for task in TASKS:
        for split in SPLITS:
            chosen = [item for item in selection if item['task'] == task and item['split'] == split]
            assert {item['label'] for item in chosen} == {'good', 'poor'}
            chosen.sort(key=lambda item: 0 if item['label'] == 'good' else 1)
            pictures = {}
            for item in chosen:
                index, kind = item['index'], item['label']
                truth = truths[split][index]
                for method in METHODS:
                    if method in ('ddis', 'fundps'):
                        output = root / 'evaluation' / method / task / split / kind
                    else:
                        output = root / 'evaluation' / method / task / split
                    verified = output / 'verified_summary.json'
                    case_path = output / f'case_{index:03d}.json'
                    if not verified.exists() or not case_path.exists():
                        missing.append(f'{method}/{task}/{split}/{kind}/{index}')
                        continue
                    summary = json.loads(verified.read_text())
                    assert index in (summary.get('sample_ids') or [index])
                    record = json.loads(case_path.read_text())
                    assert record['sample_id'] == index
                    run = json.loads((output / 'run.json').read_text())
                    prior_key = ('ddis' if method == 'ddis' else 'fundps' if method == 'fundps'
                                 else 'fm4pde' if method in ('eci_fm', 'fm_fm') else 'ofm')
                    assert run['task'] == task and run['split'] == split
                    assert run['checkpoint_sha256'] == plan['checkpoints'][prior_key]['sha256']
                    if method == 'ddis':
                        assert run['surrogate_sha256'] == plan['checkpoints']['surrogate']['sha256']
                    timing = record.get('seconds')
                    if method in ('ddis', 'fundps'):
                        timing = json.loads((output / 'completed.json').read_text())['seconds_per_case']
                    data = {'method': method, 'task': task, 'split': split, 'case': kind,
                            'sample_id': index, 'status': record['status'],
                            'historical_fm4pde_selection_score': item['selection_score'],
                            'historical_fm4pde_a': item['historical_fm4pde_relative_l2'].get('a'),
                            'historical_fm4pde_u': item['historical_fm4pde_relative_l2'].get('u'),
                            'checkpoint_sha256': plan['checkpoints'][prior_key]['sha256'],
                            'relative_l2_coefficient': record.get('relative_l2_coefficient'),
                            'relative_l2_solution': record.get('relative_l2_solution'),
                            'observed_relative_l2_coefficient': None,
                            'observed_relative_l2_solution': None,
                            'seconds': timing}
                    if record['status'] == 'ok':
                        prediction_path = (Path(record['prediction']) if method in ('ddis', 'fundps') else
                                           output / f'prediction_{index:03d}.npy')
                        prediction = np.load(prediction_path).astype(np.float64)
                        assert prediction.shape == (1, 2, 128, 128) and np.isfinite(prediction).all()
                        prediction = prediction[0]
                        for name, channel in (('coefficient', 0), ('solution', 1)):
                            rel = float(np.linalg.norm(prediction[channel] - truth[channel]) /
                                        np.linalg.norm(truth[channel]))
                            np.testing.assert_allclose(data['relative_l2_' + name], rel, rtol=1e-6, atol=1e-8)
                            active = ((task in ('forward', 'both') and channel == 0) or
                                      (task in ('inverse', 'both') and channel == 1))
                            if active:
                                pixels = observation_indices[index][channel]
                                observed = truth[channel].reshape(-1)[pixels]
                                estimated = prediction[channel].reshape(-1)[pixels]
                                data['observed_relative_l2_' + name] = float(
                                    np.linalg.norm(estimated - observed) / np.linalg.norm(observed))
                        pictures[(kind, method)] = prediction
                    data['task_score'] = score(data, task)
                    rows.append(data)
            if args.render:
                fields = FIELDS[task]
                fig, axes = plt.subplots(2 * len(fields), 8, figsize=(25, 3.1 * 2 * len(fields)),
                                         squeeze=False, layout='constrained')
                for i, item in enumerate(chosen):
                    truth = truths[split][item['index']]
                    for f, (name, channel) in enumerate(fields):
                        row = i * len(fields) + f
                        limits = np.nanpercentile(truth[channel], [1, 99])
                        vmin, vmax = float(limits[0]), float(limits[1])
                        if vmin == vmax:
                            vmax = vmin + 1e-6
                        axes[row, 0].imshow(truth[channel], cmap='viridis', vmin=vmin, vmax=vmax)
                        axes[row, 0].set_title(f"{item['label']} #{item['index']} | truth {name}", fontsize=8)
                        for col, method in enumerate(METHODS, 1):
                            picture = pictures.get((item['label'], method))
                            if picture is None:
                                axes[row, col].text(0.5, 0.5, 'pending/failed', ha='center', va='center')
                            else:
                                axes[row, col].imshow(picture[channel], cmap='viridis', vmin=vmin, vmax=vmax)
                            hit = next((entry for entry in rows if entry['task'] == task and entry['split'] == split
                                        and entry['case'] == item['label'] and entry['method'] == method), None)
                            error = hit['relative_l2_' + name] if hit else None
                            axes[row, col].set_title(f'{method} | L2={error:.3f}' if error is not None else method,
                                                     fontsize=8)
                        for axis in axes[row]:
                            axis.set_xticks([])
                            axis.set_yticks([])
                fig.suptitle(f'Poisson {task}/{split}: selected FM4PDE good and poor cases', fontsize=13)
                fig.savefig(root / 'figures' / f'{task}_{split}.png', dpi=130)
                plt.close(fig)

    output = root / 'comparison.csv'
    if rows:
        with output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    comparisons = {}
    for task in TASKS:
        for method in METHODS:
            if method == 'fm_fm':
                continue
            paired = []
            for split in SPLITS:
                for kind in ('good', 'poor'):
                    base = next((row for row in rows if row['task'] == task and row['split'] == split and
                                 row['case'] == kind and row['method'] == 'fm_fm'), None)
                    other = next((row for row in rows if row['task'] == task and row['split'] == split and
                                  row['case'] == kind and row['method'] == method), None)
                    if base and other and base['task_score'] is not None and other['task_score'] is not None:
                        paired.append((base['task_score'], other['task_score']))
            comparisons[f'{method}/{task}'] = {'paired_cases': len(paired),
                                               'better_than_fm_fm': sum(other < base for base, other in paired)}
    failed = [f"{row['method']}/{row['task']}/{row['split']}/{row['case']}/{row['sample_id']}"
              for row in rows if row['status'] != 'ok']
    status = ('partial' if len(rows) != 210 or missing else
              'complete_with_failures' if failed else 'complete')
    report = {'status': status,
              'verified_rows': len(rows), 'expected_rows': 210,
              'successful_rows': len(rows) - len(failed), 'failed_rows': len(failed),
              'missing': missing, 'failed': failed, 'comparisons': comparisons,
              'selection_source_sha256': plan['selected']['source_csv_sha256'],
              'provisional_checkpoint_sha256': {k: value['sha256'] for k, value in plan['checkpoints'].items()}}
    (root / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# Two-case Poisson sampler comparison', '',
             f"Status: {report['status']}; verified cases: {len(rows)}/210; numerical failures: {len(failed)}.", '',
             'Each forward/inverse/both × ID/Smooth/Rough/Rough2/Rough3 cell contains the best and worst',
             'FM4PDE cases among the first 100, selected from the verified prior stress-test audit.',
             'The previous FM4PDE errors select cases only. All seven new samplings use the same',
             '500 noiseless observations per active field and selected checkpoints during training.',
             'DDIS/FunDPS task-specific channel weights mirror their inverse setting; no tuning was done.',
             'Good/poor means lowest/highest historical error within each task/split, not an',
             'absolute quality threshold. These two-case diagnostics do not estimate',
             'population-level ranking.', '',
             '| Method/task | Paired cases | Better than FM4PDE ordinary prior/native |',
             '|---|---:|---:|']
    for key, value in comparisons.items():
        lines.append(f"| {key} | {value['paired_cases']} | {value['better_than_fm_fm']} |")
    lines += ['', 'Task score is physical-unit relative L2 of the solution for forward,',
              'coefficient for inverse, and the mean of both relative L2 values for both.',
              'Lower is better. Each table entry is good / poor, using the independently',
              'selected sample IDs printed in the second column. Numerical failures are',
              'shown as FAIL; missing runs are shown as —.', '']
    for task in TASKS:
        lines += [f'## {task}', '',
                  '| Split | IDs (good / poor) | ' + ' | '.join(METHODS) + ' |',
                  '|---|---|'+ '|'.join(['---:'] * len(METHODS)) + '|']
        for split in SPLITS:
            chosen = [item for item in selection if item['task'] == task and item['split'] == split]
            selected_ids = {item['label']: item['index'] for item in chosen}
            values = []
            for method in METHODS:
                scores = []
                for kind in ('good', 'poor'):
                    row = next((item for item in rows if item['task'] == task and
                                item['split'] == split and item['case'] == kind and
                                item['method'] == method), None)
                    value = row['task_score'] if row is not None else None
                    scores.append('FAIL' if row is not None and row['status'] != 'ok' else
                                  f'{value:.3f}' if value is not None else '—')
                values.append(' / '.join(scores))
            lines.append(f"| {split} | {selected_ids['good']} / {selected_ids['poor']} | " +
                         ' | '.join(values) + ' |')
        lines.append('')
    lines += ['## Sampling time', '', '| Method | Completed cases | Median seconds per case |',
              '|---|---:|---:|']
    for method in METHODS:
        elapsed = [row['seconds'] for row in rows if row['method'] == method and
                   row['status'] == 'ok' and row['seconds'] is not None]
        median = f'{np.median(elapsed):.1f}' if elapsed else '—'
        lines.append(f'| {method} | {len(elapsed)} | {median} |')
    lines += ['', 'Timings are adapter wall time; DDIS/FunDPS include solver setup, while',
              'shared-prior timings exclude the one-time network load. They are not a',
              'controlled throughput benchmark.', '']
    if missing:
        lines.append(f'Incomplete: {len(missing)} missing cases. See summary.json.')
        lines.append('')
    if failed:
        lines.append(f'Numerical failures: {len(failed)} cases; original sampler logs and nonfinite outputs are retained.')
        lines.append('')
    lines += ['See comparison.csv for both field errors and observed-point fit, and figures/',
              'for paired prediction maps. These selected extremes cannot establish overall',
              'method ranking across the full 100-case benchmark.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'status': report['status'], 'verified_rows': len(rows), 'missing': len(missing)}))


if __name__ == '__main__':
    main()
