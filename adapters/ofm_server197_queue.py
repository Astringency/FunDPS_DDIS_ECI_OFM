"""Run reserved OFM shards on server197 with the unchanged official sampler."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', DDIS_OFM_MEMORY_SLOT='0', MPLBACKEND='Agg')
    (a.root / 'locks').mkdir(exist_ok=True)
    jobs = json.loads((a.root / 'assignment.json').read_text())['shards']
    ad = Path(__file__).resolve().parent
    for number, item in enumerate(jobs):
        if number % 2 != a.gpu:
            continue
        dest = a.root / item['relative']
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'verified_summary.json').exists():
            continue
        free = lambda: int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
        while free() < 76000 or os.getloadavg()[0] > 100:
            time.sleep(30)
        # Existing finite cases retain their exact bytes and source provenance.
        cmd = [sys.executable, '-u', str(ad / 'evaluate_shared_prior.py'),
            '--prior', 'ofm', '--method', 'ofm', '--pde', 'darcy', '--task', 'forward',
            '--split', item['split'], '--offset', str(item['offset']), '--count', '10',
            '--checkpoint', str(a.root / 'training/flow/darcy/epoch_100.pt'),
            '--assets', str(a.root / 'data/shared_prior_assets'),
            '--source', str(a.root / 'data/compact/darcy'),
            '--fm4pde', str(a.base / 'official/FM4PDE-cbe627c'),
            '--ofm', str(a.base / 'official/OFM'), '--eci', str(a.base / 'official/ECI'),
            '--output', str(dest)]
        (dest / 'command.json').write_text(json.dumps(cmd, indent=2))
        with (dest / 'sampling.log').open('a') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        (dest / 'sampling.exit').write_text(str(result.returncode))
        if result.returncode:
            (dest / 'worker_failure.json').write_text(json.dumps(dict(
                host='server197', returncode=result.returncode, time=time.time())))
            print('Failed shard; retained for review:', dest, flush=True)
            continue
        with (dest / 'verification.log').open('w') as log:
            result = subprocess.run([sys.executable, str(ad / 'validate_shared_prior.py'),
                '--output', str(dest)], stdout=log, stderr=subprocess.STDOUT)
        (dest / 'verification.exit').write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(f'Independent validation failed: {dest}')
        failure = dest / 'worker_failure.json'
        if failure.exists():
            audit = dest / 'resource_retry_audit'
            audit.mkdir(exist_ok=True)
            failure.rename(audit / f'recovered_failure_{time.time_ns()}.json')
        print('Verified:', dest, flush=True)
    (a.root / f'gpu_{a.gpu}_completed.json').write_text(json.dumps(dict(time=time.time())))


if __name__ == '__main__':
    main()
