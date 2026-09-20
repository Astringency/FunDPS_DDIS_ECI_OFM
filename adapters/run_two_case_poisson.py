"""Run the selected Poisson cases through unchanged official samplers."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
METHODS = {
    'ddis': ('ddis', 'ddis'), 'fundps': ('fundps', 'fundps'),
    'ofm': ('ofm', 'ofm'), 'eci_fm': ('fm4pde', 'eci'),
    'eci_ofm': ('ofm', 'eci'), 'fm_fm': ('fm4pde', 'fm4pde'),
    'fm_ofm': ('ofm', 'fm4pde'),
}
TASKS = ('inverse', 'forward', 'both')
SPLITS = ('id', 'smooth', 'rough', 'rough2', 'rough3')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(root):
    selected = json.loads((root / 'selection.json').read_text())
    assert len(selected['cases']) == 30
    checkpoints = {}
    for key, job in (('ddis', 'ddis_poisson'), ('fundps', 'fundps_poisson'),
                     ('ofm', 'flow_poisson'), ('surrogate', 'surrogate_poisson')):
        selection = OUT / 'jobs' / job / 'early_stopping'
        checkpoint = (selection / 'best_checkpoint').resolve(strict=True)
        best = json.loads((selection / 'best.json').read_text())
        assert Path(best['checkpoint']).resolve(strict=True) == checkpoint
        checkpoints[key] = {'path': str(checkpoint), 'sha256': sha256(checkpoint),
                            'best': best}
    fm = root / 'assets' / 'weights' / 'poisson.pth'
    checkpoints['fm4pde'] = {'path': str(fm.resolve(strict=True)), 'sha256': sha256(fm)}
    expected = json.loads((root / 'assets' / 'manifest.json').read_text())['pdes']['poisson']['checkpoint_sha256']
    assert checkpoints['fm4pde']['sha256'] == expected
    plan = {'selected': selected, 'checkpoints': checkpoints,
            'methods': METHODS, 'tasks': TASKS, 'splits': SPLITS,
            'sampler': 'full official schedule',
            'mask_policy': 'NumPy RandomState(0): one 500-point draw per field and sample; task activates input field(s)',
            'source_repo': str(BASE / 'orchestration')}
    path = root / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Existing pilot plan differs; use a new result directory')
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'plan': str(path), 'checkpoint_sha256': {k: v['sha256'] for k, v in checkpoints.items()}}, indent=2))


def free_mib(gpu):
    output = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    return int(output.strip())


def commands(root, plan, label, task, split):
    prior, method = METHODS[label]
    selected = [row for row in plan['selected']['cases'] if row['task'] == task and row['split'] == split]
    assert {row['label'] for row in selected} == {'good', 'poor'}
    indices = [next(row['index'] for row in selected if row['label'] == kind) for kind in ('good', 'poor')]
    checkpoint = plan['checkpoints'][prior]['path']
    source = root / 'compact' / 'poisson'
    destination = root / 'evaluation' / label / task / split
    if label in ('ddis', 'fundps'):
        for kind, index in zip(('good', 'poor'), indices):
            output = destination / kind
            config_split = split if split in ('id', 'smooth', 'rough') else 'rough'
            command = [str(BASE / 'venv/bin/python'), '-u', str(BASE / 'orchestration/adapters/evaluate_diffusion.py'),
                       '--method', method, '--repo', str(BASE / 'official' / ('DDIS' if label == 'ddis' else 'FunDPS')),
                       '--config', str(BASE / 'orchestration/configs/evaluation' / f'{method}_poisson_{config_split}.yaml'),
                       '--checkpoint', checkpoint, '--source', str(source), '--split', split,
                       '--task', task, '--case-index', str(index), '--count', '1', '--output', str(output)]
            if split in ('rough2', 'rough3'):
                command += ['--data-path', str(root / 'hf' / 'poisson' / split)]
            if label == 'ddis':
                command += ['--surrogate', plan['checkpoints']['surrogate']['path']]
            validator = [str(BASE / 'venv/bin/python'), str(BASE / 'orchestration/adapters/validate_evaluation.py'),
                         '--source', str(source), '--split', split, '--count', '1', '--output', str(output)]
            yield kind, output, command, validator
    else:
        command = [str(BASE / 'venv-shared-prior/bin/python'), '-u',
                   str(BASE / 'orchestration/adapters/evaluate_shared_prior.py'),
                   '--prior', prior, '--method', method, '--pde', 'poisson', '--split', split,
                   '--task', task, '--count', '2', '--case-indices', *(str(index) for index in indices),
                   '--checkpoint', checkpoint, '--assets', str(root / 'assets'), '--source', str(source),
                   '--fm4pde', str(BASE / 'official/FM4PDE-cbe627c'), '--ofm', str(BASE / 'official/OFM'),
                   '--eci', str(BASE / 'official/ECI'), '--output', str(destination)]
        validator = [str(BASE / 'venv-shared-prior/bin/python'),
                     str(BASE / 'orchestration/adapters/validate_shared_prior.py'), '--output', str(destination)]
        yield 'pair', destination, command, validator


def worker(root, gpu, methods, cells=None, worker_name=None):
    plan = json.loads((root / 'plan.json').read_text())
    assert all(method in METHODS for method in methods)
    allowed = set(cells or (f'{task}:{split}' for task in TASKS for split in SPLITS))
    if not allowed or any(cell not in {f'{task}:{split}' for task in TASKS for split in SPLITS}
                          for cell in allowed):
        raise ValueError(f'Unknown task/split cell: {sorted(allowed)}')
    lock_path = OUT / 'locks' / f'provisional_gpu_{gpu}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg', WANDB_MODE='offline',
                   PYTHONDONTWRITEBYTECODE='1')
        worker_dir = root / 'workers' / (worker_name or f'gpu{gpu}')
        worker_dir.mkdir(parents=True, exist_ok=True)
        (worker_dir / 'assignment.json').write_text(json.dumps({
            'gpu': gpu, 'methods': methods, 'cells': sorted(allowed)}, indent=2) + '\n')
        failures = []
        for task in TASKS:
            for split in SPLITS:
                if f'{task}:{split}' not in allowed:
                    continue
                for label in methods:
                    for kind, output, command, validator in commands(root, plan, label, task, split):
                        if (output / 'verified_summary.json').exists():
                            continue
                        while free_mib(gpu) < 40000:
                            (worker_dir / 'status').write_text(f'Waiting for 40000 MiB free on GPU {gpu}\n')
                            time.sleep(30)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        if output.exists():
                            failures.append(str(output) + ': incomplete existing output')
                            continue
                        (worker_dir / 'status').write_text(f'{label}/{task}/{split}/{kind}\n')
                        output.mkdir()
                        (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
                        (output / 'gpu_index').write_text(str(gpu) + '\n')
                        (output / 'free_mib_at_start').write_text(str(free_mib(gpu)) + '\n')
                        with (output / 'sampling.log').open('w') as stream:
                            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                  env=env, check=False).returncode
                        (output / 'sampling.exit').write_text(str(code) + '\n')
                        if code == 0:
                            with (output / 'verification.log').open('w') as stream:
                                code = subprocess.run(validator, stdout=stream, stderr=subprocess.STDOUT,
                                                      env=env, check=False).returncode
                            (output / 'verification.exit').write_text(str(code) + '\n')
                        if code:
                            failures.append(str(output))
                            print(f'FAILED {output} exit={code}', flush=True)
                        else:
                            print(f'VERIFIED {output}', flush=True)
        (worker_dir / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
        (worker_dir / 'status').write_text('Complete\n' if not failures else 'Completed with failures\n')
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--methods', nargs='+', choices=list(METHODS))
    parser.add_argument('--cells', nargs='+')
    parser.add_argument('--worker-name')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root)
    else:
        if args.gpu is None or not args.methods:
            parser.error('Worker requires --gpu and --methods')
        worker(args.root, args.gpu, args.methods, args.cells, args.worker_name)
