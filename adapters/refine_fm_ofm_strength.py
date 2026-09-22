"""Continue strength-only tuning with disjoint ID/Smooth/Rough validation.

Only official guidance parameters change. Every attempt uses 100 steps and
500 observations. Original study results and all unsuccessful trials remain.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

RUNTIME = json.loads(Path(os.environ['FM_OFM_RUNTIME']).read_text()) if os.environ.get('FM_OFM_RUNTIME') else {}
BASE = Path(RUNTIME.get('base', '/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919'))
ROOT = Path(RUNTIME.get('root', '/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919'))
STUDY = ROOT / 'diagnostics/fm_ofm_strength_refine_20260922'
PREVIOUS = ROOT / 'diagnostics/fm_ofm_strength_confirm_20260922'
AD = Path(RUNTIME.get('adapters', str(BASE / 'orchestration/adapters')))
PY = Path(RUNTIME.get('python', str(BASE / 'venv-shared-prior/bin/python')))
PLAN = AD.parent / 'configs/evaluation/fm_ofm_strength_refine_20260922.json'
SPLITS = ('id', 'smooth', 'rough')


def assigned(group):
    return not RUNTIME.get('groups') or group['pde'] + '/' + group['task'] in RUNTIME['groups']


def relocate(value):
    for old, new in sorted(RUNTIME.get('path_map', {}).items(), key=lambda item: -len(item[0])):
        if value == old or value.startswith(old + '/'):
            return new + value[len(old):]
    return value


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n');tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def prepare_assets(pde):
    folder = STUDY / 'validation' / pde
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'prepare.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return _prepare_assets(pde)


def _prepare_assets(pde):
    """Use new training-validation cases and reserved OOD offsets 1000:1100."""
    import numpy as np
    import torch
    folder = STUDY / 'validation' / pde
    if (folder / 'ready.json').exists():
        return True
    incoming = STUDY / 'validation_export' / pde
    if not all((incoming / s / 'export.json').exists() for s in ('smooth', 'rough')):
        return False
    assets, source = folder / 'assets', folder / 'compact'
    assets.mkdir(parents=True, exist_ok=True);source.mkdir(exist_ok=True)
    manifest = read(ROOT / 'data/shared_prior_assets/manifest.json')
    stats = read(ROOT / 'data/compact' / pde / 'manifest.json')
    mean = np.asarray(stats['stats']['mean'], dtype=np.float64)[None, :, None, None]
    scale = np.asarray(stats['stats']['std'], dtype=np.float64)[None, :, None, None] / .5
    for e in manifest['pdes'].values():
        e['checkpoint_file'] = str(ROOT / 'data/shared_prior_assets' / e['checkpoint_file'])
        for record in e['splits'].values():
            record['file'] = str(ROOT / 'data/shared_prior_assets' / record['file'])
    provenance = {}
    for split in SPLITS:
        entry = manifest['pdes'][pde]['splits'][split]
        if split == 'id':
            saved = torch.load(entry['file'], map_location='cpu', weights_only=False)
            x = np.array(np.load(ROOT / 'data/compact' / pde / 'validation.npy', mmap_mode='r')[100:200])
            physical = torch.from_numpy((x * scale + mean).astype(np.float32))
            gt = saved['ground_truth']
            gt.update(pair=physical, coef=physical[:, :1], sol=physical[:, -1:])
            original_ids = np.load(ROOT / 'data/compact' / pde / 'validation_ids.npy')[100:200].tolist()
            gt['metadata'].update(sample_offsets=list(range(100)), original_sample_ids=original_ids,
                calibration_partition='fresh_training_validation_100_200')
            provenance[split] = dict(partition='training_validation', compact_offsets=list(range(100, 200)), original_ids=original_ids)
        else:
            record = read(incoming / split / 'export.json')
            path = incoming / split / 'truth.pt'
            assert sha(path) == record['truth_sha256']
            assert record['original_ids'] == list(range(1000, 1100))
            saved = torch.load(path, map_location='cpu', weights_only=False)
            physical = saved['ground_truth']['pair']
            x = ((physical.numpy().astype(np.float64) - mean) / scale).astype(np.float32)
            provenance[split] = record
        truth = assets / f'{split}.pt'
        torch.save(saved, truth)
        entry.update(file=str(truth), sha256=sha(truth))
        np.save(source / f'{split}.npy', x);np.save(source / f'{split}_ids.npy', np.arange(100))
        assert physical.shape == (100, 1 if pde == 'burger' else 2, 128, 128)
        np.testing.assert_allclose(x.astype(np.float64) * scale + mean, physical.numpy(), rtol=1e-5, atol=1e-5)
    manifest['calibration'] = dict(partition='disjoint_multidistribution_validation', provenance=provenance,
        screening_local_ids=list(range(8)), confirmation_local_ids=list(range(32, 48)))
    atomic(assets / 'manifest.json', manifest);atomic(source / 'manifest.json', stats)
    atomic(folder / 'ready.json', dict(provenance=provenance, manifest_sha256=sha(assets / 'manifest.json')))
    return True


def acquire_compute(gpu):
    """Allow two extra trials on busy GPUs; idle GPUs get their own worker."""
    while True:
        free, util = map(int, subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(','))
        load_limit = RUNTIME.get('max_cpu_load', 110)
        if free >= RUNTIME.get('min_free_mib', 32768) and (load_limit is None or os.getloadavg()[0] < load_limit) and util <= 90:
            if util <= 60:
                return None
            for i in range(2):
                lock = (STUDY / f'shared_compute_{i}.lock').open('w')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock
                except BlockingIOError:
                    lock.close()
        time.sleep(15)


def evaluate(group, phase, split, label, overrides, ids, gpu):
    dest = STUDY / group['pde'] / group['task'] / phase / split / label
    marker = dest / 'verified_summary.json'
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        slot = acquire_compute(gpu)
        try:
            old = read(ROOT / 'evaluation_v2/ofm/fm4pde' / group['pde'] / group['task'] / split / 'shard_000/run.json')
            old = {k: relocate(v) if isinstance(v, str) else v for k, v in old.items()}
            if phase == 'test':
                assets, source = Path(old['assets']), Path(old['source'])
            else:
                folder = STUDY / 'validation' / group['pde']
                assets, source = folder / 'assets', folder / 'compact'
                assert read(assets / 'manifest.json')['calibration']['partition'] == 'disjoint_multidistribution_validation'
            path = dest / 'overrides.json'
            if path.exists():
                assert read(path) == overrides
            else:
                atomic(path, overrides)
            assert set(overrides) <= {'clip_threshold', 'zeta_obs_a', 'zeta_obs_u', 'zeta_pde', 'stochastic_guidance_coeff'}
            cmd = [str(PY), '-u', str(AD / 'evaluate_shared_prior.py'), '--prior', 'ofm', '--method', 'fm4pde',
                '--pde', group['pde'], '--task', group['task'], '--split', split, '--case-indices', *map(str, ids),
                '--count', str(len(ids)), '--checkpoint', old['checkpoint'], '--assets', str(assets), '--source', str(source),
                '--fm4pde', old['fm4pde'], '--ofm', old['ofm'], '--eci', old['eci'], '--output', str(dest),
                '--fm-overrides', str(path)]
            atomic(dest / 'launch.json', dict(command=cmd, gpu=gpu, time=time.time()))
            threads = str(RUNTIME.get('cpu_threads', 2))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads, MPLBACKEND='Agg')
            with (dest / 'sampling.log').open('a') as log:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            (dest / 'sampling.exit').write_text(str(code))
            if code:
                raise RuntimeError(f'Sampling exit {code}: {dest}')
            with (dest / 'verification.log').open('w') as log:
                subprocess.run([str(PY), str(AD / 'validate_shared_prior.py'), '--output', str(dest)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            if slot:
                slot.close()
    run = read(dest / 'run.json')
    assert run['guidance_overrides'] == overrides and run['case_indices'] == ids
    v = read(marker)
    cases = [read(p) for p in dest.glob('case_*.json')]
    assert sorted(r['sample_id'] for r in cases) == sorted(v['sample_ids']) == sorted(ids)
    field = 'coefficient' if group['task'] == 'inverse' else 'solution'
    errors = [r['relative_l2_' + field] for r in cases if r['status'] == 'ok']
    assert all(math.isfinite(x) and x >= 0 for x in errors)
    return dict(mean=v['all_case_mean_relative_l2_' + field], failures=v['failures'],
        over_1000=sum(x > 10 for x in errors), maximum=max(errors) if errors else None,
        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in cases))


def stable(results):
    return len(results) == 3 and all(r['failures'] == 0 and r['over_1000'] == 0 for r in results.values())


def score(results):
    return statistics.fmean(r['mean'] for r in results.values())


def work_group(group, gpu, plan):
    folder = STUDY / group['pde'] / group['task']
    while not prepare_assets(group['pde']):
        atomic(folder / 'state.json', dict(state='waiting_validation_export', gpu=gpu, updated_at=time.time()))
        time.sleep(30)
    decision_path = folder / 'decision.json'
    if not decision_path.exists():
        screening = {}
        for label, overrides in group['candidates'].items():
            results = {}
            for split in ('rough', 'id', 'smooth'):
                atomic(folder / 'state.json', dict(state='screening', candidate=label, split=split, gpu=gpu, updated_at=time.time()))
                results[split] = evaluate(group, 'screening', split, label, overrides, plan['screening_ids'], gpu)
                if results[split]['failures'] or results[split]['over_1000']:
                    break
            screening[label] = results
            atomic(folder / 'screening.json', screening)
        ranking = sorted((k for k, v in screening.items() if stable(v)), key=lambda k: score(screening[k]))
        if not ranking:
            raise RuntimeError('No stable candidate on all three validation distributions')
        finalists = ranking[:3]
        if 'anchor' in ranking and 'anchor' not in finalists:
            finalists.append('anchor')
        if 'clip20' in ranking and 'clip20' not in finalists:
            finalists.append('clip20')
        confirmation = {}
        for label in finalists:
            results = {}
            for split in ('rough', 'id', 'smooth'):
                atomic(folder / 'state.json', dict(state='confirmation', candidate=label, split=split, gpu=gpu, updated_at=time.time()))
                results[split] = evaluate(group, 'confirmation', split, label, group['candidates'][label], plan['confirmation_ids'], gpu)
                if results[split]['failures'] or results[split]['over_1000']:
                    break
            confirmation[label] = results
            atomic(folder / 'confirmation.json', confirmation)
        eligible = [k for k, v in confirmation.items() if stable(v)]
        if not eligible:
            raise RuntimeError('No finalist passed independent confirmation across all distributions')
        selected = min(eligible, key=lambda k: score(confirmation[k]))
        if 'anchor' in eligible and score(confirmation[selected]) > .99 * score(confirmation['anchor']):
            selected = 'anchor'
        atomic(decision_path, dict(selected=selected, overrides=group['candidates'][selected],
            screening=screening, confirmation=confirmation, frozen_before_test_at=time.time(),
            selection_note=plan['selection_note']))
    decision = read(decision_path)
    for split in group['test_splits']:
        selected_path = folder / 'test' / split / 'result_selection.json'
        if selected_path.exists():
            continue
        atomic(folder / 'state.json', dict(state='test', split=split, gpu=gpu, updated_at=time.time()))
        result = evaluate(group, 'test', split, 'selected', decision['overrides'], list(range(100)), gpu)
        selected, label, overrides = decision['selected'], 'selected', decision['overrides']
        if result['failures'] or result['over_1000']:
            # Predetermined conservative fallbacks use full 100-case settings,
            # never cherry-pick predictions or select the lowest test error.
            fallback = [k for k in ('anchor', 'clip20') if k != selected and
                k in decision['confirmation'] and stable(decision['confirmation'][k])]
            for candidate in fallback:
                atomic(folder / 'state.json', dict(state='test_stability_retry', split=split,
                    candidate=candidate, gpu=gpu, updated_at=time.time()))
                label = 'fallback_' + candidate
                overrides = group['candidates'][candidate]
                result = evaluate(group, 'test', split, label, overrides, list(range(100)), gpu)
                selected = candidate
                if result['failures'] == 0 and result['over_1000'] == 0:
                    break
        if result['failures'] or result['over_1000']:
            raise RuntimeError(f'No predeclared confirmed fallback stabilized all 100 {split} test cases')
        atomic(selected_path, dict(label=label, selected=selected, overrides=overrides,
            selection_note=decision['selection_note'] + (' Test-triggered stability fallback: whole setting replaced by first stable predeclared confirmed fallback; no test-error minimization.' if label != 'selected' else '')))
    atomic(folder / 'completed.json', dict(time=time.time(), gpu=gpu))
    atomic(folder / 'state.json', dict(state='completed', gpu=gpu, updated_at=time.time()))


def worker(gpu, only=None):
    STUDY.mkdir(parents=True, exist_ok=True)
    plan = read(PLAN)
    atomic(STUDY / 'plan.json', plan)
    gpu_lock = (STUDY / f'gpu_{gpu}.lock').open('w')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for group in plan['groups']:
        if not assigned(group):
            continue
        if only and group['pde'] + '/' + group['task'] != only:
            continue
        folder = STUDY / group['pde'] / group['task']
        folder.mkdir(parents=True, exist_ok=True)
        if any((folder / n).exists() for n in ('completed.json', 'worker_error.json')):
            continue
        claim = (folder / 'claim.lock').open('w')
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            claim.close();continue
        atomic(folder / 'worker.json', dict(pid=os.getpid(), gpu=gpu, host=RUNTIME.get('host', 'server216'), started_at=time.time()))
        try:
            work_group(group, gpu, plan)
        except Exception as error:
            atomic(folder / 'worker_error.json', dict(error=str(error), time=time.time(), gpu=gpu))
            print('Group needs review:', group['pde'], group['task'], error, flush=True)
        finally:
            claim.close()


def merge_report(report):
    """Prefer complete round-two cells; expose every failed or extreme result."""
    if PLAN.exists() and (STUDY / 'plan.json').exists():
        plan = read(PLAN)
        report.setdefault('round1_rows', copy.deepcopy(report['rows']))
        errors, active, count = [], 0, 0
        for group in plan['groups']:
            if not assigned(group):
                continue
            folder = STUDY / group['pde'] / group['task']
            if (folder / 'worker_error.json').exists():
                errors.append(dict(pde=group['pde'], task=group['task'], **read(folder / 'worker_error.json')))
            w = read(folder / 'worker.json') if (folder / 'worker.json').exists() else {}
            try:
                cmdline = Path('/proc', str(w.get('pid', -1)), 'cmdline').read_bytes()
                active += int(b'refine_fm_ofm_strength.py' in cmdline and
                    not any((folder / n).exists() for n in ('completed.json', 'worker_error.json')))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
            for row in report['rows']:
                if (row['pde'], row['task']) != (group['pde'], group['task']) or row['split'] not in group['test_splits']:
                    continue
                selection_path = folder / 'test' / row['split'] / 'result_selection.json'
                selection = read(selection_path) if selection_path.exists() else {}
                out = folder / 'test' / row['split'] / selection.get('label', 'selected')
                state = read(folder / 'state.json') if (folder / 'state.json').exists() else {}
                row.update(refinement_status=state.get('state', 'queued'))
                if not (out / 'verified_summary.json').exists():
                    continue
                v = read(out / 'verified_summary.json')
                assert v['cases'] == 100 and sorted(v['sample_ids']) == list(range(100))
                cases = [read(p) for p in out.glob('case_*.json')]
                assert sorted(r['sample_id'] for r in cases) == list(range(100))
                field = 'coefficient' if group['task'] == 'inverse' else 'solution'
                values = [r['relative_l2_' + field] * 100 for r in cases if r['status'] == 'ok']
                assert all(math.isfinite(x) and x >= 0 for x in values)
                assert len(values) == v['successes'] == 100 - v['failures']
                if values:
                    assert math.isclose(statistics.fmean(values) / 100, v['successful_case_mean_relative_l2_' + field], rel_tol=1e-10)
                mean = statistics.fmean(values) if len(values) == 100 else None
                d = read(folder / 'decision.json')
                selected = selection.get('selected', d['selected'])
                overrides = selection.get('overrides', d['overrides'])
                row.update(previous_round_mean_percent=row['tuned_mean_percent'], previous_round_failures=row['failures'],
                    saved=100, verified=100, failures=v['failures'], status='complete_with_failures' if v['failures'] else 'complete',
                    tuned_mean_percent=mean, tuned_std_percent=statistics.stdev(values) if len(values) == 100 else None,
                    finite_mean_percent=statistics.fmean(values) if values else None,
                    finite_max_percent=max(values) if values else None, finite_over_1000_percent=sum(x > 1000 for x in values),
                    tuned_to_fm_fm_ratio=mean / row['fm_fm_paper_mean_percent'] if mean is not None else None,
                    reaches_fm_fm_mean=mean <= row['fm_fm_paper_mean_percent'] if mean is not None else None,
                    selected=selected, parameters_json=json.dumps(overrides, sort_keys=True), result_round=2,
                    refinement_status='verified', confirmation_baseline_mean_percent=None,
                    confirmation_candidate_mean_percent=score(d['confirmation'][selected]) * 100,
                    comparison_note=selection.get('selection_note', d['selection_note']))
                count += 1
        hosts = report.get('refinement', {}).get('hosts', {})
        hosts[RUNTIME.get('host', 'server216')] = dict(verified_settings=count, active_groups=active, errors=errors,
            assigned_groups=[g['pde'] + '/' + g['task'] for g in plan['groups'] if assigned(g)])
        report['refinement'] = dict(verified_settings=sum(h['verified_settings'] for h in hosts.values()),
            expected_settings=sum(len(g['test_splits']) for g in plan['groups']),
            active_groups=sum(h['active_groups'] for h in hosts.values()),
            errors=[e for h in hosts.values() for e in h['errors']], hosts=hosts)
    for row in report['rows']:
        row.setdefault('result_round', 1)
        row.setdefault('refinement_status', 'not_targeted')
        row.setdefault('previous_round_mean_percent', None)
        row.setdefault('previous_round_failures', None)
    report.update(stable_settings=sum(r['verified'] == 100 and r['failures'] == 0 and r['finite_over_1000_percent'] == 0 for r in report['rows']),
        numeric_failures=sum(r['failures'] for r in report['rows']),
        extreme_finite_samples=sum(r['finite_over_1000_percent'] for r in report['rows']),
        reaches_target_settings=sum(r['reaches_fm_fm_mean'] is True for r in report['rows']))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker-gpu', type=int)
    p.add_argument('--merge-report', action='store_true')
    p.add_argument('--only')
    p.add_argument('--probe', action='store_true')
    a = p.parse_args()
    if a.merge_report:
        print(json.dumps(merge_report(json.load(sys.stdin)), allow_nan=False))
    elif a.worker_gpu is None:
        p.error('--worker-gpu is required for sampling')
    elif a.probe:
        STUDY.mkdir(parents=True, exist_ok=True)
        plan = read(PLAN)
        group = next(g for g in plan['groups'] if assigned(g))
        assert prepare_assets(group['pde']), 'Reserved validation export not ready'
        result = evaluate(group, 'preflight_' + RUNTIME.get('host', 'server216'), 'rough', 'anchor', group['candidates']['anchor'], [0, 1], a.worker_gpu)
        assert result['failures'] == 0 and result['over_1000'] == 0
        assert result['peak_reserved_bytes'] < 20 * 1024 ** 3
        atomic(STUDY / 'preflight.json', dict(result=result, gpu=a.worker_gpu, completed_at=time.time()))
        print(json.dumps(result), flush=True)
    else:
        worker(a.worker_gpu, a.only)
