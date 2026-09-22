"""Confirm frozen guidance choices, then evaluate all 27 settings with official code."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
ROOT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
STUDY = ROOT / 'diagnostics/fm_ofm_strength_confirm_20260922'
AD = BASE / 'orchestration/adapters'
PY = BASE / 'venv-shared-prior/bin/python'
PLAN = AD.parent / 'configs/evaluation/fm_ofm_strength_confirm_20260922.json'


def read(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 2:
                raise
            time.sleep(.1)


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def resources(gpu):
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
    return free >= 32768 and os.getloadavg()[0] < 110


def run_cases(group, phase, split, label, overrides, ids, gpu):
    pde, task = group['pde'], group['task']
    output = STUDY / pde / task / phase / split / label
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'verified_summary.json').exists():
        return read(output / 'verified_summary.json')
    while not resources(gpu):
        time.sleep(20)
    reference = read(ROOT / 'evaluation_v2/ofm/fm4pde' / pde / task / split / 'shard_000/run.json')
    assets, source = Path(reference['assets']), Path(reference['source'])
    if phase == 'confirmation':
        assets = ROOT / 'calibration_v2' / pde / 'validation_assets'
        source = ROOT / 'calibration_v2' / pde / 'validation_compact'
        manifest = read(assets / 'manifest.json')
        assert manifest['calibration']['partition'] == 'validation'
        assert not set(ids) & set(manifest['calibration']['tuning_local_ids'])
        assert not set(ids) & set(manifest['calibration']['heldout_local_ids'])
        assert manifest['pdes'][pde]['splits']['id']['sha256'] != reference['truth_sha256']
        assert (source / 'manifest.json').exists()
    assert set(overrides) <= {'zeta_obs_a', 'zeta_obs_u', 'zeta_pde', 'clip_threshold', 'stochastic_guidance_coeff'}
    override_path = output / 'overrides.json'
    if override_path.exists():
        assert read(override_path) == overrides
    else:
        atomic(override_path, overrides)
    command = [str(PY), '-u', str(AD / 'evaluate_shared_prior.py'), '--prior', 'ofm',
        '--method', 'fm4pde', '--pde', pde, '--task', task, '--split', split,
        '--case-indices', *map(str, ids), '--count', str(len(ids)), '--checkpoint', reference['checkpoint'],
        '--assets', str(assets), '--source', str(source), '--fm4pde', reference['fm4pde'],
        '--ofm', reference['ofm'], '--eci', reference['eci'], '--output', str(output),
        '--fm-overrides', str(override_path)]
    atomic(output / 'launch.json', dict(command=command, gpu=gpu, time=time.time(),
        expected_checkpoint_sha256=reference['checkpoint_sha256'], sampling_steps=100))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    with (output / 'sampling.log').open('a') as log:
        code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (output / 'sampling.exit').write_text(str(code))
    if code:
        raise RuntimeError(f'Sampling exited {code}: {output}')
    with (output / 'verification.log').open('w') as log:
        code = subprocess.run([str(PY), str(AD / 'validate_shared_prior.py'), '--output', str(output)],
            env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (output / 'verification.exit').write_text(str(code))
    if code:
        raise RuntimeError(f'Independent verification exited {code}: {output}')
    result = read(output / 'verified_summary.json')
    assert sorted(result['sample_ids']) == sorted(ids)
    assert result['checkpoint_sha256'] == reference['checkpoint_sha256']
    return result


def summarize(plan):
    rows = []
    for group in plan['groups']:
        pde, task = group['pde'], group['task']
        decision_path = STUDY / pde / task / 'decision.json'
        decision = read(decision_path) if decision_path.exists() else {}
        field = 'coefficient' if task == 'inverse' else 'solution'
        for split in ('id', 'smooth', 'rough'):
            output = STUDY / pde / task / 'test' / split / 'selected'
            marker = output / 'verified_summary.json'
            verified = read(marker) if marker.exists() else {}
            cases = [read(p) for p in output.glob('case_*.json')]
            good = [r['relative_l2_' + field] * 100 for r in cases if r['status'] == 'ok']
            n, failures = verified.get('cases', 0), verified.get('failures', 0)
            if verified:
                assert len(cases) == n and sorted(r['sample_id'] for r in cases) == sorted(verified['sample_ids'])
                assert all(math.isfinite(v) and v >= 0 for v in good)
                assert n - len(good) == failures
                if good:
                    assert math.isclose(statistics.fmean(good) / 100,
                        verified['successful_case_mean_relative_l2_' + field], rel_tol=1e-10, abs_tol=1e-12)
            mean = statistics.fmean(good) if n == 100 and not failures else None
            references = {r['method']: r for r in plan['references']
                if (r['pde'], r['task'], r['split']) == (pde, task, split)}
            reference = float(references['FM-FM']['mean_percent'])
            rows.append(dict(pde=pde, task=task, split=split, saved=len(cases), verified=n,
                failures=failures, status=('complete_with_failures' if failures else 'complete') if n == 100 else 'pending',
                tuned_mean_percent=mean,
                tuned_std_percent=statistics.stdev(good) if n == 100 and not failures else None,
                previous_fm_ofm_mean_percent=float(references['FM-OFM']['mean_percent']),
                fm_fm_paper_mean_percent=reference,
                tuned_to_fm_fm_ratio=mean / reference if mean is not None else None,
                reaches_fm_fm_mean=mean <= reference if mean is not None else None,
                selected=decision.get('selected'), parameters_json=json.dumps(decision.get('overrides', {}), sort_keys=True),
                observation_count=500, sampling_steps=100,
                comparison_note='FM-FM is the manuscript reference; settings differ in pretrained network. Parameters selected on validation only.'))
    return dict(updated_at=time.time(), complete_settings=sum(r['verified'] == 100 for r in rows),
        expected_settings=27, rows=rows,
        errors=[dict(group=str(p.parent.relative_to(STUDY)), **read(p)) for p in STUDY.glob('*/*/worker_error.json')])


def write_report(report, dest):
    atomic(dest / 'comparison.json', report)
    temporary = dest / f'comparison.{os.getpid()}.partial.csv'
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['rows'][0]))
        writer.writeheader()
        writer.writerows(report['rows'])
    temporary.replace(dest / 'comparison.csv')


def worker(gpu):
    plan = read(PLAN)
    STUDY.mkdir(parents=True, exist_ok=True)
    atomic(STUDY / 'plan.json', plan)
    lock = (STUDY / f'gpu_{gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        pending = [g for g in plan['groups'] if not any((STUDY / g['pde'] / g['task'] / name).exists()
            for name in ('completed.json', 'worker_error.json'))]
        if not pending:
            break
        if not resources(gpu):
            time.sleep(20)
            continue
        claimed = False
        for group in pending:
            folder = STUDY / group['pde'] / group['task']
            folder.mkdir(parents=True, exist_ok=True)
            claim = (folder / 'claim.lock').open('w')
            try:
                fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                claim.close()
                continue
            if (folder / 'completed.json').exists() or (folder / 'worker_error.json').exists():
                claim.close()
                continue
            claimed = True
            atomic(folder / 'worker.json', dict(pid=os.getpid(), gpu=gpu, started_at=time.time()))
            try:
                field = 'coefficient' if group['task'] == 'inverse' else 'solution'
                values = {}
                for label in ('baseline', 'candidate'):
                    v = run_cases(group, 'confirmation', 'id', label, group[label], plan['confirmation_ids'], gpu)
                    values[label] = dict(mean=v['all_case_mean_relative_l2_' + field], failures=v['failures'])
                candidate, baseline = values['candidate'], values['baseline']
                use_candidate = candidate['failures'] == 0 and (baseline['failures'] > 0 or candidate['mean'] <= .99 * baseline['mean'])
                selected = 'candidate' if use_candidate else 'baseline'
                if values[selected]['failures']:
                    raise RuntimeError('Neither fixed configuration passed confirmation')
                decision = dict(selected=selected, overrides=group[selected], confirmation=values,
                    confirmation_ids=plan['confirmation_ids'], screening_source=group['screening_source'],
                    rule=plan['acceptance'], frozen_before_test_at=time.time())
                old = folder / 'decision.json'
                if old.exists():
                    assert read(old)['overrides'] == decision['overrides']
                else:
                    atomic(old, decision)
                for split in ('id', 'smooth', 'rough'):
                    run_cases(group, 'test', split, 'selected', decision['overrides'], list(range(100)), gpu)
                    write_report(summarize(plan), STUDY / 'report')
                atomic(folder / 'completed.json', dict(time=time.time(), gpu=gpu))
            except Exception as error:
                atomic(folder / 'worker_error.json', dict(error=str(error), time=time.time(), gpu=gpu))
                print('Group needs review:', group['pde'], group['task'], error, flush=True)
            finally:
                claim.close()
            break
        if not claimed:
            time.sleep(20)
    write_report(summarize(plan), STUDY / 'report')
    print('Study worker completed:', gpu, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-gpu', type=int)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'reports/fm_ofm_guidance_20260922')
    args = parser.parse_args()
    if args.fetch:
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
        socket = Path('/tmp/ddis216-reallocation-20260921')
        if socket.exists():
            ssh += ['-S', str(socket)]
        command = ['python3', str(AD / Path(__file__).name), '--report']
        value = subprocess.check_output(ssh + ['server216', shlex.join(command)], text=True, timeout=180)
        result = json.loads(value)
        write_report(result, args.output)
        print(f"Complete settings: {result['complete_settings']}/27; group errors: {len(result['errors'])}")
        print(args.output / 'comparison.csv')
    elif args.report:
        print(json.dumps(summarize(read(PLAN)), allow_nan=False))
    elif args.worker_gpu is not None:
        worker(args.worker_gpu)
    else:
        parser.error('Choose --worker-gpu, --report or --fetch')


if __name__ == '__main__':
    main()
