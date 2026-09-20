"""Record official diffusion sampler NaNs as failed pilot cases, retaining raw output."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from validate_evaluation import verify


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def recover(output, source):
    exit_code = int((output / 'sampling.exit').read_text().strip())
    if exit_code == 0 or (output / 'verified_summary.json').exists():
        return False
    run = json.loads((output / 'run.json').read_text())
    if run['method'] not in ('ddis', 'fundps') or run['count'] != 1:
        return False
    index = run['case_index']
    if not isinstance(index, int) or not 0 <= index < 100:
        return False
    log_path = output / 'sampling.log'
    log = log_path.read_text(errors='replace')
    if not all(text in log for text in ('NaN detected!', 'plot_process',
                                        'ZeroDivisionError: division by zero')):
        return False
    prediction_path = output / 'results' / 'batch_0.npy'
    if not prediction_path.exists():
        return False
    prediction = np.load(prediction_path)
    if prediction.shape != (1, 2, 128, 128) or np.isfinite(prediction).all():
        return False
    case_path = output / f'case_{index:03d}.json'
    if case_path.exists():
        case = json.loads(case_path.read_text())
        if case.get('status') != 'failed' or case.get('sample_id') != index:
            raise ValueError(f'Conflicting case record: {case_path}')
    else:
        case = {'sample_id': index, 'status': 'failed',
                'error': 'Official sampler generated nonfinite output; see sampling.log',
                'failure_class': 'official_sampler_nonfinite',
                'prediction': str(prediction_path), 'prediction_units': 'physical',
                'relative_l2_coefficient': None, 'relative_l2_solution': None}
        case_path.write_text(json.dumps(case, indent=2, allow_nan=False) + '\n')
    completed = output / 'completed.json'
    if not completed.exists():
        completed.write_text(json.dumps({'evaluated': 1, 'failed': 1,
            'seconds': None, 'seconds_per_case': None,
            'sampling_exit_code': exit_code}, indent=2) + '\n')
    summary = verify(output, source, run['split'], 1)
    assert summary['failed_cases'] == [index]
    (output / 'verification.exit').write_text('0\n')
    (output / 'verification.log').write_text(json.dumps(summary, indent=2) + '\n')
    marker = {'status': 'verified_numerical_failure', 'sample_id': index,
              'sampling_exit_code': exit_code,
              'sampling_log_sha256': sha256(log_path),
              'nonfinite_prediction_sha256': sha256(prediction_path),
              'case_record_sha256': sha256(case_path)}
    (output / 'recovered_numerical_failure.json').write_text(json.dumps(marker, indent=2) + '\n')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    source = args.root / 'compact' / 'poisson'
    recovered, unrecovered = [], []
    for exit_path in sorted((args.root / 'evaluation').rglob('sampling.exit')):
        if int(exit_path.read_text().strip()) == 0:
            continue
        output = exit_path.parent
        if recover(output, source):
            recovered.append(str(output))
        elif not (output / 'recovered_numerical_failure.json').exists():
            unrecovered.append(str(output))
    print(json.dumps({'recovered': recovered, 'unrecovered': unrecovered}), flush=True)
    if unrecovered:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
