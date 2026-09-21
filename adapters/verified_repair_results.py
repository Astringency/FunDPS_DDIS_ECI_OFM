"""Read audited whole-setting repairs without altering original evaluations."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def collect_repairs(root, snapshot):
    root = Path(root)
    repairs = []
    specifications = [
        ('ECI-OFM', 'diagnostics/eci_ofm_poisson_repair100_20260921/report/audit.json'),
        ('FM-OFM', 'diagnostics/fm_ofm_repair_20260921/report/audit.json'),
    ]
    for method, relative in specifications:
        audit_path = root / relative
        if not audit_path.exists():
            snapshot['warnings'].append(f'Repair audit unavailable; retaining original settings: {relative}')
            continue
        raw = audit_path.read_bytes()
        audit = json.loads(raw)
        groups = audit['groups'] if method == 'FM-OFM' else [audit]
        for group in groups:
            run = group['run']
            pde, task = run['pde'], run['task']
            split = group['split'] if method == 'FM-OFM' else run['split']
            if method == 'FM-OFM':
                allowed = {('helmholtz', 'inverse', s) for s in ('id', 'smooth', 'rough')}
                allowed |= {('nsnonbounded', 'inverse', 'rough'), ('burger', 'both', 'rough')}
                if (pde, task, split) not in allowed:
                    raise ValueError('Unexpected FM-OFM repair setting')
                full = root / 'diagnostics/fm_ofm_repair_20260921' / pde / f'full_{split}'
                variant = 'clip50'
                folder = full / variant / split
                version = 'repair_clip50_20260921'
                parameters = dict(steps=100, clip_threshold=50)
                if run['variants'] != {'clip50': {'clip_threshold': 50.0}} or run['steps'] != 100:
                    raise ValueError('Unexpected FM-OFM repair parameters')
            else:
                if (pde, task, split) != ('poisson', 'inverse', 'rough'):
                    raise ValueError('Unexpected ECI-OFM repair setting')
                full = root / 'diagnostics/eci_ofm_poisson_repair100_20260921'
                variant = 'notebook_boundary_200x1_resample5'
                folder = full / variant
                version = 'repair_200x1_resample5_20260921'
                parameters = dict(steps=200, mix=1, resample_step=5)
                if run['variants'] != {variant: [200, 1, 5, True]}:
                    raise ValueError('Unexpected ECI-OFM repair parameters')
            if not (full / 'completed.json').is_file():
                raise ValueError(f'Repair is not complete: {full}')
            if json.loads((full / 'run.json').read_text()) != run:
                raise ValueError(f'Repair configuration changed since independent verification: {full}')
            validation = group['validation']
            if validation['verified_predictions'] != 100:
                raise ValueError(f'Repair audit does not verify all 100 predictions: {full}')
            if validation.get('failed', validation.get('numeric_failures')) != 0:
                raise ValueError(f'Repair audit reports numerical failures: {full}')
            records = group['cases']
            if sorted(r['sample_id'] for r in records) != list(range(100)):
                raise ValueError(f'Repair must contain each of the 100 sample IDs exactly once: {full}')
            original_runs = [m for m in snapshot['runs'].values()
                if (m['method'], m['pde'], m['task'], m['split']) == (method, pde, task, split)]
            if not original_runs or any(m.get('run', {}).get('checkpoint_sha256') != run['checkpoint_sha256']
                                        for m in original_runs):
                raise ValueError(f'Repair and original checkpoint IDs disagree: {full}')
            for record in records:
                path = folder / f"case_{record['sample_id']:03d}.json"
                if json.loads(path.read_text()) != record:
                    raise ValueError(f'Repair case changed since independent verification: {path}')
                if record['status'] != 'ok':
                    raise ValueError(f'Repair has a failed case: {path}')
                for field in (('solution',) if pde == 'burger' else ('coefficient', 'solution')):
                    value = record.get('relative_l2_' + field)
                    if value is None or not math.isfinite(value) or value < 0:
                        raise ValueError(f'Repair has an invalid metric: {path}')
            field = 'coefficient' if task == 'inverse' else 'solution'
            actual_mean = statistics.fmean(r['relative_l2_' + field] for r in records) * 100
            audited_mean = validation['mean_percent'] if method == 'FM-OFM' else validation['means_percent'][0]
            if not math.isclose(actual_mean, audited_mean, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f'Repair mean disagrees with independent audit: {full}')
            verified = dict(cases=100, successes=100, failures=0, sample_ids=list(range(100)),
                checkpoint_sha256=run['checkpoint_sha256'], verification_source=relative,
                verification_audit_sha256=hashlib.sha256(raw).hexdigest())
            for field in ('coefficient', 'solution'):
                values = [r['relative_l2_' + field] for r in records if r.get('relative_l2_' + field) is not None]
                verified['successful_case_mean_relative_l2_' + field] = statistics.fmean(values) if values else None
            key = str(folder.relative_to(root))
            meta = dict(method=method, pde=pde, task=task, split=split, run=run, verified=verified,
                result_version=version, sampling_parameters=parameters,
                selection_note='Failure-triggered test-set parameter tuning; whole 100-case setting replaced, not independently held-out selection.')
            cases = [dict(r, method=method, pde=pde, task=task, split=split, run_key=key,
                case_source=f"{key}/case_{r['sample_id']:03d}.json") for r in records]
            repairs.append(dict(run_key=key, meta=meta, cases=cases))
    repairs.extend(collect_fundps_repair(root, snapshot))
    return repairs


def collect_fundps_repair(root, snapshot):
    relative = 'diagnostics/fundps_repair_20260921/report/audit.json'
    path = root / relative
    if not path.exists():
        return []
    raw = path.read_bytes()
    audit = json.loads(raw)
    key = ('FunDPS', 'poisson', 'forward', 'rough')
    assert tuple(audit[f] for f in ('method', 'pde', 'task', 'split')) == key
    assert audit['parameters'] == dict(iterations=500, guidance_weights=[10000, 0, 0])
    assert audit['validation']['verified_predictions'] == 100 and audit['validation']['failed'] == 0
    original = [m for m in snapshot['runs'].values()
        if tuple(m[f] for f in ('method', 'pde', 'task', 'split')) == key]
    assert original and all(m['run']['checkpoint_sha256'] == audit['checkpoint_sha256'] for m in original)
    records = []
    run_key = 'diagnostics/fundps_repair_20260921/full'
    for item in audit['records']:
        folder = root / item['directory']
        record, run, verified = item['case'], item['run'], item['verified']
        assert json.loads((folder / 'run.json').read_text()) == run
        assert json.loads((folder / 'verified_summary.json').read_text()) == verified
        source = folder / f"case_{record['sample_id']:03d}.json"
        assert json.loads(source.read_text()) == record
        assert run['case_index'] == record['sample_id'] and run['count'] == 1
        assert run['checkpoint_sha256'] == audit['checkpoint_sha256']
        assert run['official_config']['guidance']['weights'] == [10000, 0, 0]
        assert run['official_config']['iterations'] == 500
        assert verified['sample_ids'] == [record['sample_id']]
        assert verified['successful_cases'] == 1 and not verified['failed_cases']
        assert record['status'] == 'ok'
        for field in ('coefficient', 'solution'):
            assert math.isfinite(record['relative_l2_' + field]) and record['relative_l2_' + field] >= 0
        # Check predictions still match the independently audited bytes.
        assert hashlib.sha256(Path(record['prediction']).read_bytes()).hexdigest() == item['prediction_sha256']
        records.append(dict(record, method=key[0], pde=key[1], task=key[2], split=key[3],
            run_key=run_key, case_source=str(source.relative_to(root))))
    assert sorted(r['sample_id'] for r in records) == list(range(100))
    means = [statistics.fmean(r['relative_l2_' + f] for r in records) for f in ('coefficient', 'solution')]
    assert math.isclose(means[1] * 100, audit['validation']['mean_percent'], rel_tol=1e-10)
    verified = dict(verified_cases=100, successful_cases=100, failed_cases=[],
        sample_ids=list(range(100)), mean_relative_l2_successful_cases=means,
        verification_source=relative, verification_audit_sha256=hashlib.sha256(raw).hexdigest())
    meta = dict(method=key[0], pde=key[1], task=key[2], split=key[3],
        run=dict(checkpoint_sha256=audit['checkpoint_sha256']), verified=verified,
        result_version='repair_weight10000_20260921', sampling_parameters=audit['parameters'],
        selection_note='Failure-triggered test-set parameter tuning; whole 100-case setting replaced, not independently held-out selection.')
    return [dict(run_key=run_key, meta=meta, cases=records)]


def apply_repairs(snapshot):
    """Select complete audited versions; retain the untouched input snapshot."""
    selected = dict(snapshot, runs=dict(snapshot['runs']), cases=list(snapshot['cases']))
    replaced = set()
    for repair in snapshot.get('repairs', []):
        meta = repair['meta']
        fields = ('method', 'pde', 'task', 'split')
        key = tuple(meta[k] for k in fields)
        if key in replaced or sorted(r['sample_id'] for r in repair['cases']) != list(range(100)):
            raise ValueError(f'Duplicate or incomplete whole-setting repair: {key}')
        if any(tuple(r[k] for k in fields) != key or r['run_key'] != repair['run_key'] for r in repair['cases']):
            raise ValueError(f'Repair case belongs to the wrong setting: {key}')
        replaced.add(key)
        selected['cases'] = [r for r in selected['cases'] if tuple(r[k] for k in fields) != key] + repair['cases']
        selected['runs'] = {k: m for k, m in selected['runs'].items() if tuple(m[f] for f in fields) != key}
        selected['runs'][repair['run_key']] = meta
    return selected
