"""Read audited repairs with explicit whole-setting or selected-sample scope."""
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
    repairs.extend(collect_eci_outlier_repairs(root, snapshot))
    return repairs


def collect_eci_outlier_repairs(root, snapshot):
    relative = 'diagnostics/eci_ofm_outliers_20260922/report/audit.json'
    path = root / relative
    if not path.exists():
        return []
    raw = path.read_bytes()
    audit = json.loads(raw)
    expected = {'id': [6, 8, 13, 15, 66, 80], 'smooth': [12, 13, 17, 55, 80]}
    assert sorted(g['split'] for g in audit['groups']) == sorted(expected)
    repairs = []
    for group in audit['groups']:
        split, run = group['split'], group['run']
        ids = expected[split]
        fields = dict(method='ECI-OFM', pde='poisson', task='inverse', split=split)
        assert all(group[k] == v for k, v in fields.items())
        assert group['sample_ids'] == ids and group['repair_scope'] == 'selected_samples'
        assert run['ids'] == ids and run['variants'] == {'mix1_800x1': [800, 1, None, True]}
        assert group['validation']['verified_predictions'] == len(ids) and group['validation']['failed'] == 0
        full = root / 'diagnostics/eci_ofm_outliers_20260922' / split
        assert json.loads((full / 'run.json').read_text()) == run
        assert (full / 'completed.json').is_file()
        originals = [m for m in snapshot['runs'].values() if all(m[k] == v for k, v in fields.items())]
        assert originals and all(m['run']['checkpoint_sha256'] == run['checkpoint_sha256'] for m in originals)
        for source, digest in group['untouched_case_sha256'].items():
            assert hashlib.sha256((root / source).read_bytes()).hexdigest() == digest
        key = str((full / 'mix1_800x1').relative_to(root))
        records = []
        for item in group['records']:
            record = item['case']
            assert json.loads((root / item['source']).read_text()) == record
            assert json.loads((root / item['original_source']).read_text()) == item['original']
            assert hashlib.sha256((root / item['prediction']).read_bytes()).hexdigest() == item['prediction_sha256']
            assert record['status'] == 'ok' and 0 <= record['relative_l2_coefficient'] < 10
            assert math.isfinite(record['relative_l2_solution']) and record['relative_l2_solution'] >= 0
            records.append(dict(record, **fields, run_key=key, case_source=item['source']))
        assert sorted(r['sample_id'] for r in records) == ids
        verified = dict(cases=len(ids), successes=len(ids), failures=0, sample_ids=ids,
            checkpoint_sha256=run['checkpoint_sha256'], verification_source=relative,
            verification_audit_sha256=hashlib.sha256(raw).hexdigest())
        for field in ('coefficient', 'solution'):
            verified['successful_case_mean_relative_l2_' + field] = statistics.fmean(r['relative_l2_' + field] for r in records)
        meta = dict(**fields, run=run, verified=verified, result_version='repair_selected_outliers_mix1_20260922',
            sampling_parameters=dict(steps=800, mix=1, resample_step=None, sample_ids=ids,
                remaining_original_cases=100-len(ids), original_steps=800, original_mix=5),
            selection_note=f'User-requested test-outlier tuning: only IDs {ids} use mixing 1 instead of 5; all other cases retain original results. Same 800 steps, checkpoint, seeds and 500 observations. Not held-out parameter selection.')
        repairs.append(dict(run_key=key, meta=meta, cases=records, repair_scope='selected_samples'))
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
    assert audit['repair_scope'] == 'selected_samples' and audit['sample_ids'] == [6]
    assert audit['validation']['verified_predictions'] == 1 and audit['validation']['failed'] == 0
    original = [m for m in snapshot['runs'].values()
        if tuple(m[f] for f in ('method', 'pde', 'task', 'split')) == key]
    assert original and all(m['run']['checkpoint_sha256'] == audit['checkpoint_sha256'] for m in original)
    records = []
    run_key = 'diagnostics/fundps_repair_20260921/pilot_weight10000/case_006'
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
    assert [r['sample_id'] for r in records] == [6]
    means = [statistics.fmean(r['relative_l2_' + f] for r in records) for f in ('coefficient', 'solution')]
    assert math.isclose(means[1] * 100, audit['validation']['mean_percent'], rel_tol=1e-10)
    verified = dict(verified_cases=1, successful_cases=1, failed_cases=[],
        sample_ids=[6], mean_relative_l2_successful_cases=means,
        verification_source=relative, verification_audit_sha256=hashlib.sha256(raw).hexdigest())
    meta = dict(method=key[0], pde=key[1], task=key[2], split=key[3],
        run=dict(checkpoint_sha256=audit['checkpoint_sha256']), verified=verified,
        result_version='repair_sample006_weight10000_20260921',
        sampling_parameters=dict(audit['parameters'], sample_ids=[6], other_99_cases_guidance_weights=[20000, 0, 0]),
        selection_note='User-requested failed-sample repair only: ID 6 uses weight 10000; the other 99 original cases retain weight 20000. This setting mixes configurations selected after observing a failure.')
    return [dict(run_key=run_key, meta=meta, cases=records, repair_scope='selected_samples')]


def apply_repairs(snapshot):
    """Select complete audited versions; retain the untouched input snapshot."""
    selected = dict(snapshot, runs=dict(snapshot['runs']), cases=list(snapshot['cases']))
    replaced = set()
    for repair in snapshot.get('repairs', []):
        meta = repair['meta']
        fields = ('method', 'pde', 'task', 'split')
        key = tuple(meta[k] for k in fields)
        if repair.get('repair_scope') == 'selected_samples':
            ids = [r['sample_id'] for r in repair['cases']]
            is_fundps = key == ('FunDPS', 'poisson', 'forward', 'rough') and ids == [6]
            is_eci = key[:3] == ('ECI-OFM', 'poisson', 'inverse') and key[3] in ('id', 'smooth')
            if key in replaced or not (is_fundps or is_eci) or not ids or len(set(ids)) != len(ids):
                raise ValueError(f'Unexpected selected-sample repair: {key}')
            if any(tuple(r[k] for k in fields) != key or r['run_key'] != repair['run_key'] for r in repair['cases']):
                raise ValueError(f'Repair case belongs to the wrong setting: {key}')
            previous = [r for r in selected['cases'] if tuple(r[k] for k in fields) == key and r['sample_id'] in ids]
            if len(previous) != len(ids):
                raise ValueError('Selected-sample repair must replace each original ID exactly once')
            if is_fundps and previous[0]['status'] != 'failed':
                raise ValueError('Selected-sample repair must replace exactly the documented failed sample')
            if is_eci:
                outliers = [r['sample_id'] for r in selected['cases'] if tuple(r[k] for k in fields) == key
                    and r['status'] == 'ok' and r['relative_l2_coefficient'] > 10]
                if sorted(outliers) != sorted(ids):
                    raise ValueError('Selected ECI repair must cover exactly the original finite outliers above 1000 percent')
            retained = [r for r in selected['cases'] if not (tuple(r[k] for k in fields) == key and r['sample_id'] in ids)]
            for old_key in {r['run_key'] for r in previous}:
                remainder = [r for r in retained if r['run_key'] == old_key]
                if not remainder:
                    selected['runs'].pop(old_key)
                    continue
                old_meta = selected['runs'][old_key]
                v = old_meta.get('verified')
                before = [r for r in selected['cases'] if r['run_key'] == old_key]
                if not v or sorted(v['sample_ids']) != sorted(r['sample_id'] for r in before):
                    raise ValueError('Selected-sample repair requires verified original shard coverage')
                for field in ('coefficient', 'solution'):
                    values = [r['relative_l2_' + field] for r in before if r['status'] == 'ok' and r.get('relative_l2_' + field) is not None]
                    expected_mean = v.get('successful_case_mean_relative_l2_' + field)
                    if values and expected_mean is not None and not math.isclose(statistics.fmean(values), expected_mean, rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError('Original verification mean disagrees before selected-sample repair')
                ok = [r for r in remainder if r['status'] == 'ok']
                subset = dict(v, cases=len(remainder), successes=len(ok), failures=len(remainder)-len(ok),
                    sample_ids=sorted(r['sample_id'] for r in remainder),
                    verification_source=f'{old_key}/verified_summary.json (retained subset; original audit in original_verified)')
                for field in ('coefficient', 'solution'):
                    values = [r['relative_l2_' + field] for r in ok if r.get('relative_l2_' + field) is not None]
                    subset['successful_case_mean_relative_l2_' + field] = statistics.fmean(values) if values else None
                    subset['all_case_mean_relative_l2_' + field] = statistics.fmean(values) if len(values) == len(remainder) else None
                selected['runs'][old_key] = dict(old_meta, original_verified=v, verified=subset)
            selected['cases'] = retained + repair['cases']
            selected['runs'][repair['run_key']] = meta
            replaced.add(key)
            continue
        if key in replaced or sorted(r['sample_id'] for r in repair['cases']) != list(range(100)):
            raise ValueError(f'Duplicate or incomplete whole-setting repair: {key}')
        if any(tuple(r[k] for k in fields) != key or r['run_key'] != repair['run_key'] for r in repair['cases']):
            raise ValueError(f'Repair case belongs to the wrong setting: {key}')
        replaced.add(key)
        selected['cases'] = [r for r in selected['cases'] if tuple(r[k] for k in fields) != key] + repair['cases']
        selected['runs'] = {k: m for k, m in selected['runs'].items() if tuple(m[f] for f in fields) != key}
        selected['runs'][repair['run_key']] = meta
    return selected
