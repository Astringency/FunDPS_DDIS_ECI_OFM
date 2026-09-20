"""Copy the completed remote Poisson pilot report to this checkout."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


REPORT_FILES = ('plan.json', 'selection.json', 'comparison.csv', 'summary.json',
                'report.md', 'finalized.json', 'figures')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='server216')
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--local-output', required=True, type=Path)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--timeout-hours', type=float, default=24)
    args = parser.parse_args()
    remote = args.remote_root.rstrip('/')
    marker = remote + '/finalized.json'
    failed_marker = remote + '/finalization_failed.json'
    deadline = time.monotonic() + args.timeout_hours * 3600
    while subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                          args.server, 'test', '-f', marker], check=False,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        failed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                 args.server, 'test', '-f', failed_marker], check=False,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if failed:
            raise RuntimeError(f'Remote pilot finalization failed: {failed_marker}')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Remote pilot did not finalize within {args.timeout_hours} hours')
        time.sleep(args.poll_seconds)
    destination = args.local_output
    temporary = destination.with_name('.' + destination.name + '.partial')
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'Report destination already exists: {destination} or {temporary}')
    temporary.mkdir(parents=True)
    producer = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                 args.server, 'tar', '-C', remote, '-cf', '-', *REPORT_FILES],
                                stdout=subprocess.PIPE)
    consumer = subprocess.run(['tar', '-C', str(temporary), '-xf', '-'],
                              stdin=producer.stdout, check=False)
    producer.stdout.close()
    source_code = producer.wait()
    if source_code or consumer.returncode:
        raise RuntimeError(f'Report transfer failed: ssh/tar={source_code}, local tar={consumer.returncode}')
    manifest = json.loads((temporary / 'finalized.json').read_text())
    assert manifest['status'] == 'complete' and manifest['verified_rows'] == 210
    for name, expected in manifest['artifact_sha256'].items():
        actual = sha256(temporary / name)
        if actual != expected:
            raise ValueError(f'Report artifact changed during transfer: {name}: {actual} != {expected}')
    if len(list((temporary / 'figures').glob('*.png'))) != 15:
        raise ValueError('Expected 15 final reconstruction figures')
    os.replace(temporary, destination)
    print(json.dumps({'status': 'complete', 'destination': str(destination),
                      'verified_rows': 210, 'verified_artifacts': len(manifest['artifact_sha256'])}), flush=True)


if __name__ == '__main__':
    main()
