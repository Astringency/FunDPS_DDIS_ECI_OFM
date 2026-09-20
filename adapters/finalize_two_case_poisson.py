"""Finalize the pinned Poisson pilot after both GPU workers have exited."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--workers', nargs='+', default=['gpu4', 'gpu6'])
    args = parser.parse_args()
    root = args.root
    exits = [root / 'workers' / name / 'controller.exit' for name in args.workers]
    while not all(path.exists() for path in exits):
        print('Waiting for ' + ', '.join(str(path) for path in exits if not path.exists()), flush=True)
        time.sleep(args.poll_seconds)
    codes = [int(path.read_text().strip()) for path in exits]
    print(f'Worker exit codes: {codes}', flush=True)
    recovery_script = Path(__file__).with_name('recover_two_case_numerical_failures.py')
    subprocess.run([sys.executable, str(recovery_script), '--root', str(root)], check=True)
    for exit_path, code in zip(exits, codes):
        if not code:
            continue
        failures_path = exit_path.parent / 'failures.json'
        if not failures_path.exists():
            raise RuntimeError(f'Worker failed without a failure list: {exit_path}')
        failures = json.loads(failures_path.read_text())
        if not failures or any(not (Path(path) / 'recovered_numerical_failure.json').exists()
                               for path in failures):
            raise RuntimeError(f'Worker has unverified failures: {failures_path}')
    report_script = Path(__file__).with_name('summarize_two_case_poisson.py')
    subprocess.run([sys.executable, str(report_script), '--root', str(root), '--render'], check=True)
    summary = json.loads((root / 'summary.json').read_text())
    if summary['status'] not in ('complete', 'complete_with_failures') or summary['verified_rows'] != 210:
        raise RuntimeError(f'Pilot incomplete: worker exits={codes}, summary={summary["status"]}, '
                           f'verified rows={summary["verified_rows"]}/210')
    artifacts = [root / name for name in ('plan.json', 'selection.json', 'comparison.csv',
                                          'summary.json', 'report.md', 'parallel_shards.json')]
    figures = sorted((root / 'figures').glob('*.png'))
    if len(figures) != 15:
        raise RuntimeError(f'Expected 15 paired figures, found {len(figures)}')
    artifact_hashes = {str(path.relative_to(root)): sha256(path) for path in artifacts + figures}
    record = {'status': summary['status'], 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'workers': args.workers, 'worker_exit_codes': codes,
              'verified_rows': summary['verified_rows'], 'failed_rows': summary['failed_rows'],
              'artifact_sha256': artifact_hashes}
    (root / 'finalized.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'verified_rows': 210,
                      'failed_rows': summary['failed_rows'], 'figures': len(figures)}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root') + 1])
            (root / 'finalization_failed.json').write_text(json.dumps({
                'status': 'failed', 'failed_utc': datetime.now(timezone.utc).isoformat(),
                'error': str(error), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise
