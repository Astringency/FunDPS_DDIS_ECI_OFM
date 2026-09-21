"""Uniform 100-case stability rerun using the unchanged official FunDPS solver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
ROOT = OUT / 'diagnostics/fundps_repair_20260921'
AD = BASE / 'orchestration/adapters'
PY = BASE / 'venv/bin/python'
CHECKPOINT = OUT / 'training/fundps/poisson/0920_012307-fm4pde_poisson_fundps_official-unimk66t/network-snapshot-4507232.pkl'


def normalized_config(run):
    config = dict(run['official_config'])
    for key in ('outdir', 'data_offset'):
        config.pop(key)
    return config


def audit():
    original = json.loads((OUT / 'evaluation_v2/diffusion/fundps/poisson/forward/rough/case_006/run.json').read_text())
    expected = normalized_config(original)
    expected['guidance'] = dict(type='dps', weights=[10000, 0, 0])
    rows = []
    for index in range(100):
        folder = ROOT / 'full' / f'case_{index:03d}'
        run = json.loads((folder / 'run.json').read_text())
        assert normalized_config(run) == expected
        assert run['checkpoint_sha256'] == original['checkpoint_sha256']
        assert run['case_index'] == index and run['count'] == 1
        with (folder / 'verification.log').open('w') as log:
            subprocess.run([str(PY), str(AD / 'validate_evaluation.py'), '--output', str(folder),
                '--source', str(OUT / 'data/compact/poisson'), '--split', 'rough', '--count', '1'],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads((folder / f'case_{index:03d}.json').read_text())
        verified = json.loads((folder / 'verified_summary.json').read_text())
        assert record['sample_id'] == index and record['status'] == 'ok'
        rows.append(dict(case=record, run=run, verified=verified,
            directory=str(folder.relative_to(OUT)),
            prediction_sha256=hashlib.sha256(Path(record['prediction']).read_bytes()).hexdigest()))
    result = dict(method='FunDPS', pde='poisson', task='forward', split='rough',
        checkpoint_sha256=original['checkpoint_sha256'], records=rows,
        parameters=dict(iterations=500, guidance_weights=[10000, 0, 0]),
        validation=dict(verified_predictions=100, failed=0,
            mean_percent=statistics.fmean(r['case']['relative_l2_solution'] for r in rows) * 100))
    report = ROOT / 'report'
    report.mkdir(exist_ok=True)
    temporary = report / 'audit.partial.json'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(report / 'audit.json')
    print(json.dumps(result['validation']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit:
        audit()
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    for index in range(args.worker, 100, args.workers):
        dest = ROOT / 'full' / f'case_{index:03d}'
        if (dest / 'completed.json').exists():
            continue
        while int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'])) < 12000:
            time.sleep(20)
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [str(PY), '-u', str(AD / 'evaluate_diffusion.py'), '--method', 'fundps',
            '--repo', str(BASE / 'official/FunDPS'), '--config',
            str(BASE / 'orchestration/configs/evaluation/repair_fundps_poisson_rough_weight10000.yaml'),
            '--checkpoint', str(CHECKPOINT), '--source', str(OUT / 'data/compact/poisson'),
            '--fm4pde', str(BASE / 'official/FM4PDE-cbe627c'), '--split', 'rough',
            '--task', 'forward', '--case-index', str(index), '--count', '1', '--output', str(dest)]
        (dest / 'command.json').write_text(json.dumps(cmd, indent=2))
        with (dest / 'sampling.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        (dest / 'sampling.exit').write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(f'FunDPS repair failed to run: {dest}')
        record = json.loads((dest / f'case_{index:03d}.json').read_text())
        if record['status'] != 'ok':
            raise RuntimeError(f'FunDPS repair is still numerically unstable: {dest}')
        print('Completed', index, flush=True)


if __name__ == '__main__':
    main()
