#!/usr/bin/env python3
"""Fetch current evaluation metadata and write seven-method metrics/progress CSVs.

Uses only the Python standard library. Reads small JSON records; never runs
sampling, imports a model, or modifies any server-side evaluation result.
Defaults to audited repairs with explicit scopes and also exports original results.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

from verified_repair_results import apply_repairs, collect_repairs
from manuscript_metrics import parse_paper

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = '/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919'
REMOTE_SCRIPT = '/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919/orchestration/adapters/summarize_current_results.py'
PDES = ('poisson', 'helmholtz', 'darcy', 'nsnonbounded', 'burger')
SPLITS = ('id', 'smooth', 'rough')
PREFIXES = {'ECI-FM': 'fm4pde/eci', 'ECI-OFM': 'ofm/eci',
            'DDIS': 'diffusion/ddis', 'FM-OFM': 'ofm/fm4pde',
            'OFM': 'ofm/ofm', 'FunDPS': 'diffusion/fundps'}
METHODS = ('ECI-FM', 'ECI-OFM', 'DDIS', 'FM-FM', 'FM-OFM', 'OFM', 'FunDPS')


def expected_cells():
    for method in PREFIXES:
        for pde in (PDES[:2] if method in ('DDIS', 'FunDPS') else PDES):
            for task in (('both',) if pde == 'burger' else ('forward', 'inverse')):
                for split in SPLITS:
                    yield method, pde, task, split


def read_json(path):
    # Workers publish some JSON files directly. Retry a transient partial write.
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 2:
                raise
            time.sleep(.05)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, include_repairs=True):
    root = Path(root)
    if not (root / 'evaluation_v2').is_dir():
        raise FileNotFoundError(f'Evaluation root not found: {root}')
    result = dict(schema_version=2, collection_started_at=datetime.now(timezone.utc).isoformat(),
                  root=str(root), runs={}, cases=[], warnings=[])
    for method, pde, task, split in expected_cells():
        cell = root / 'evaluation_v2' / PREFIXES[method] / pde / task / split
        pattern = 'case_[0-9][0-9][0-9]' if method in ('DDIS', 'FunDPS') else 'shard_[0-9][0-9][0-9]'
        for folder in sorted(cell.glob(pattern)):
            if not folder.is_dir():
                continue
            run_key = str(folder.relative_to(root))
            meta = dict(method=method, pde=pde, task=task, split=split)
            for filename, key in [('run.json', 'run'), ('verified_summary.json', 'verified'),
                                  ('worker_failure.json', 'worker_failure'),
                                  ('recovery_state.json', 'recovery'),
                                  ('resource_needs_review.json', 'resource_needs_review')]:
                path = folder / filename
                if path.exists():
                    try:
                        meta[key] = read_json(path)
                    except (FileNotFoundError, json.JSONDecodeError) as error:
                        # A final verification record must be readable to certify a run.
                        if key == 'verified':
                            raise ValueError(f'Unreadable verification: {path}') from error
                        result['warnings'].append(f'Skipped a changing file: {path}')
            if meta.get('run', {}).get('profile') or meta.get('run', {}).get('profile_only'):
                continue
            result['runs'][run_key] = meta
            for path in sorted(folder.glob('case_[0-9][0-9][0-9].json')):
                try:
                    case = read_json(path)
                except (FileNotFoundError, json.JSONDecodeError):
                    if 'verified' in meta:
                        raise ValueError(f'Unreadable case in verified run: {path}')
                    result['warnings'].append(f'Skipped a changing file: {path}')
                    continue
                result['cases'].append(dict(case, method=method, pde=pde, task=task,
                    split=split, run_key=run_key, case_source=str(path.relative_to(root))))
    result['repairs'] = collect_repairs(root, result) if include_repairs else []
    result['captured_at'] = datetime.now(timezone.utc).isoformat()
    return result


def quantile(values, q):
    ordered = sorted(values)
    loc = (len(ordered) - 1) * q
    lower, upper = math.floor(loc), math.ceil(loc)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (loc - lower)


def recovery_status(meta, snapshot):
    if meta.get('verified'):
        return 'verified'
    record = meta.get('recovery', {})
    state = record.get('state', 'unassigned')
    now = datetime.fromisoformat(snapshot['captured_at']).timestamp() if snapshot.get('captured_at') else time.time()
    if state in ('running', 'validating', 'queued', 'verified') and now - record.get('updated_at', 0) > 180:
        return 'stale'
    return 'awaiting_sync' if state == 'verified' else state


def aggregate(snapshot):
    grouped, by_run = defaultdict(list), defaultdict(list)
    seen = set()
    expected = set(expected_cells())
    for case in snapshot['cases']:
        key = tuple(case[k] for k in ('method', 'pde', 'task', 'split'))
        case_key = key + (case['sample_id'],)
        if key not in expected or case_key in seen or case['sample_id'] not in range(100):
            raise ValueError(f'Duplicate or unexpected sample: {case_key}')
        if case['status'] not in ('ok', 'failed'):
            raise ValueError(f'Unknown sample status: {case_key}')
        seen.add(case_key)
        grouped[key].append(case)
        by_run[case['run_key']].append(case)

    verified_keys = set()
    for run_key, meta in snapshot['runs'].items():
        v = meta.get('verified')
        if not v:
            continue
        cases = by_run[run_key]
        ids = sorted(c['sample_id'] for c in cases)
        if ids != sorted(v['sample_ids']):
            raise ValueError(f'Verification/case IDs disagree: {run_key}')
        ok = [c for c in cases if c['status'] == 'ok']
        count = v.get('cases', v.get('verified_cases'))
        success = v.get('successes', v.get('successful_cases'))
        failures = v.get('failures', len(v.get('failed_cases', [])))
        if (count, success, failures) != (len(cases), len(ok), len(cases) - len(ok)):
            raise ValueError(f'Verification counts disagree: {run_key}')
        if not cases:
            raise ValueError(f'Empty verified run: {run_key}')
        field = 'coefficient' if cases[0]['task'] == 'inverse' else 'solution'
        values = [c['relative_l2_' + field] for c in ok]
        if any(v is None or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f'Invalid successful-case error: {run_key}')
        for c in cases:
            if c['status'] == 'failed' and (c.get('relative_l2_coefficient') is not None or c.get('relative_l2_solution') is not None):
                raise ValueError(f'Failed case has numeric metrics: {run_key}')
        if cases[0]['method'] in ('DDIS', 'FunDPS'):
            means = v['mean_relative_l2_successful_cases']
            saved = means[0 if field == 'coefficient' else 1] if means is not None else None
        else:
            saved = v['successful_case_mean_relative_l2_' + field]
        if values and (saved is None or not math.isclose(saved, statistics.fmean(values), rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f'Verification mean disagrees: {run_key}')
        if not values and saved is not None:
            raise ValueError(f'Unexpected mean without successful cases: {run_key}')
        verified_keys.add(run_key)

    rows = []
    for key in expected_cells():
        method, pde, task, split = key
        cases = grouped[key]
        records = [c for c in cases if c['run_key'] in verified_keys]
        field = 'coefficient' if task == 'inverse' else 'solution'
        values = [c['relative_l2_' + field] for c in records if c['status'] == 'ok']
        failed = [c['sample_id'] for c in records if c['status'] == 'failed']
        runs = [r for r in snapshot['runs'].values()
                if tuple(r[f] for f in ('method', 'pde', 'task', 'split')) == key]
        recovery_states = [recovery_status(r, snapshot) for r in runs
            if r.get('worker_failure') and not r.get('verified')]
        n = len(records)
        status = ('complete_with_failures' if failed else 'complete') if n == 100 else ('partial' if runs else 'not_started')
        avg = statistics.fmean(values) * 100 if values else None
        row = dict(method=method, pde=pde, task=task, split=split, n=n,
            n_finite=len(values), n_failed=len(failed),
            mean_percent=avg if n == 100 and not failed else None,
            finite_mean_percent=avg,
            finite_std_percent=statistics.stdev(values) * 100 if len(values) >= 2 else None,
            finite_median_percent=statistics.median(values) * 100 if values else None,
            finite_p90_percent=quantile(values, .9) * 100 if values else None,
            finite_max_percent=max(values) * 100 if values else None,
            finite_over_100_percent=sum(v > 1 for v in values),
            finite_over_1000_percent=sum(v > 10 for v in values),
            failed_ids=','.join(map(str, sorted(failed))), source='latest_source_snapshot.json.gz',
            expected_n=100, n_saved=len(cases), n_verified=n, n_unverified=len(cases)-n,
            n_pending=100-n, status=status,
            worker_failure_runs=sum(bool(r.get('worker_failure')) and not r.get('verified') for r in runs),
            verified_shards=sum(bool(r.get('verified')) for r in runs) if method == 'OFM' else 0,
            recovery_active_shards=sum(s in ('running', 'validating') for s in recovery_states),
            recovery_queued_shards=sum(s in ('queued', 'awaiting_sync') for s in recovery_states),
            recovery_failed_shards=recovery_states.count('failed'),
            recovery_unassigned_shards=sum(s not in ('running', 'validating', 'queued', 'awaiting_sync', 'failed') for s in recovery_states),
            resource_review_runs=sum(bool(r.get('resource_needs_review')) and not r.get('verified') for r in runs),
            verified_ids=','.join(map(str, sorted(c['sample_id'] for c in records))),
            result_version=';'.join(sorted({r.get('result_version', 'original') for r in runs})) or 'original',
            sampling_parameters_json=json.dumps([r['sampling_parameters'] for r in runs if 'sampling_parameters' in r], sort_keys=True),
            selection_note=';'.join(sorted({r['selection_note'] for r in runs if r.get('selection_note')})),
            verification_sources=';'.join(sorted({r['verified']['verification_source'] for r in runs
                if r.get('verified', {}).get('verification_source')})))
        rows.append(row)
    return rows


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def csv_bytes(rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def export(snapshot, paper, output, original_only=False):
    original_rows = aggregate(snapshot)
    rows = aggregate(snapshot if original_only else apply_repairs(snapshot))
    original_by_key = {tuple(r[k] for k in ('method', 'pde', 'task', 'split')): r for r in original_rows}
    for r in rows:
        original = original_by_key[tuple(r[k] for k in ('method', 'pde', 'task', 'split'))]
        r['original_n_failed'] = original['n_failed']
        r['original_finite_over_1000_percent'] = original['finite_over_1000_percent']
    paper_rows = parse_paper(paper)
    paper_counts = sorted({r['n'] for r in paper_rows})
    totals = {}
    for method in PREFIXES:
        selected = [r for r in rows if r['method'] == method]
        totals[method] = {key: sum(r[key] for r in selected) for key in
            ('expected_n', 'n_saved', 'n_verified', 'n_finite', 'n_failed', 'worker_failure_runs', 'resource_review_runs',
             'verified_shards', 'recovery_active_shards', 'recovery_queued_shards', 'recovery_failed_shards', 'recovery_unassigned_shards')}
        totals[method].update(cells=len(selected), completed_cells=sum(r['n_verified'] == 100 for r in selected),
            original_n_failed=sum(r['original_n_failed'] for r in selected),
            repaired_settings=sum(any(v.startswith('repair_') for v in r['result_version'].split(';')) for r in selected))
    for r in rows + original_rows + paper_rows:
        r['captured_at'] = snapshot['captured_at']
        r['metric_units'] = 'percent'
        r['target_field'] = 'coefficient' if r['task'] == 'inverse' else 'solution'
    all_rows = sorted(rows + paper_rows, key=lambda r: (PDES.index(r['pde']), r['task'], SPLITS.index(r['split']), METHODS.index(r['method'])))
    original_all_rows = sorted(original_rows + paper_rows, key=lambda r: (PDES.index(r['pde']), r['task'], SPLITS.index(r['split']), METHODS.index(r['method'])))
    warnings = list(snapshot.get('warnings', []))
    if not original_only and 'repairs' not in snapshot:
        warnings.append('This older snapshot contains no repair evidence; retaining original settings.')
    summary = dict(captured_at=snapshot['captured_at'], generated_at=datetime.now(timezone.utc).isoformat(),
        source_root=snapshot['root'], paper=str(paper), paper_sha256=sha(paper),
        methods=totals, metric_rows=len(all_rows), warnings=warnings,
        manuscript_sample_counts=paper_counts,
        result_selection='original_only' if original_only else 'audited_repairs_with_explicit_scope',
        repaired_settings=sum(t['repaired_settings'] for t in totals.values()),
        notes=['All statistics use verified case records only; unfinished shards are counted under n_saved, not n_verified.',
               'mean_percent is blank unless all 100 cases are verified with finite target errors.',
               'finite_* statistics may describe a partial set or exclude failed cases; always inspect status and n_finite.',
               'Finite extreme errors are uncapped. Worker/resource failures do not count as numeric case failures.',
               'FM-FM uses active manuscript main tables and caption-declared sample counts; other methods target 100 cases.',
               'Each cell has 500 noiseless observations. DDIS/FunDPS cover Poisson and Helmholtz only.',
               'Manuscript NS inverse ID/Smooth and Burgers Random ID/Smooth weights used first 100 test inputs for tuning.',
               'Only metadata and saved verification summaries are checked; prediction arrays are not revalidated by this command.',
               'Verified includes documented numerical failures; n_finite counts verified successful cases, not accuracy-qualified cases.',
               'metrics_original.csv always preserves original-configuration statistics.',
               'Repair parameters were selected after inspecting failures; result_version and selection_note identify the scope. FunDPS replaces only failed sample 6 at user request; other repairs replace whole settings.'])
    report = ['# 最新采样统计', '', f'服务器快照时间：{snapshot["captured_at"]}', '',
              f'结果版本：{summary["result_selection"]}；含修复结果的设置：{summary["repaired_settings"]}。原配置统计另存 `metrics_original.csv`。', '',
              '| 方法 | 已保存 | 已验证 / 计划 | 满 100 例的设置 | 数值失败 | 分片错误记录 |',
              '|---|---:|---:|---:|---:|---:|']
    for method, t in totals.items():
        report.append(f'| {method} | {t["n_saved"]} | {t["n_verified"]}/{t["expected_n"]} | {t["completed_cells"]}/{t["cells"]} | {t["n_failed"]} | {t["worker_failure_runs"]} |')
    report += ['', '数值失败也计入已验证数量，但不计入成功样本均值；分片错误记录可能正在重试。', '',
        'complete settings 仅计入已经核验满 100 例的设置；OFM 每个分片为 10 例。分片错误在完整核验前继续保留，恢复中的错误另列 recovering/queued；逐分片状态见 shard_recovery.csv。', '',
        'verified 表示记录已核验，不代表采样成功或精度达标。成功记录由独立核验脚本读取预测数组、检查样本/观测位置并重算误差；失败记录核对状态及缺失指标。汇总命令只交叉核对这些核验证据。', '',
        'ECI-OFM、FM-OFM 修正版按完整 100 例设置替换；FunDPS 按用户要求仅修复 Poisson forward/Rough 的样本 6，其引导权重为 10000，其他 99 例保留原权重 20000。CSV 的 result_version、sampling_parameters_json、selection_note 标注范围和参数；original_n_failed 保留原始失败数。这是失败触发的参数调整，不是独立验证集选参。', '',
        '所有误差列均为百分数。`mean_percent` 仅在完整 100 例均有有限误差时填写；`finite_mean_percent` 等统计列仅使用已验证且成功的样本，可能来自未完成设置。必须结合 `status` 和 `n_finite` 阅读。', '',
        f'FM-FM 来自当前正文主表，标题中的样本数为 {paper_counts}，每项 n/expected_n 按对应标题填写；其他方法计划每格 100 例。DDIS/FunDPS 的 Darcy、NS、Burgers 没有已训练模型，本表不将其计入待采样任务。', '',
        '论文 NS inverse ID/Smooth、Burgers Random ID/Smooth 曾用相应测试集前 100 例调观测权重。各方法计算预算也不同，因此本表不是严格配对、预算匹配的算法比较。', '']
    for method in ('OFM', 'FunDPS'):
        report += [f'## {method} 逐项进度', '', '| 方程 | 任务 | 分布 | 已保存 | 已验证 / 100 | 失败 | 状态 | 已验证成功样本均值（%） |', '|---|---|---|---:|---:|---:|---|---:|']
        for r in rows:
            if r['method'] != method:
                continue
            avg = '—' if r['finite_mean_percent'] is None else f'{r["finite_mean_percent"]:.6g}'
            report.append('| ' + ' | '.join(map(str, [r['pde'], r['task'], r['split'], r['n_saved'], r['n_verified'], r['n_failed'], r['status'], avg])) + ' |')
    report += ['', '来源：`latest_source_snapshot.json.gz`；完整统计：`metrics.csv`；进度：`progress.csv`；统计口径与来源哈希：`latest_summary.json`。',
               '脚本只读取结果元数据，核对样本编号、去重、验证计数与误差均值，不重新采样或重新加载预测数组。', '']
    # Build and validate everything before replacing any output. Each file is atomic.
    recovery_rows = []
    for key, meta in snapshot['runs'].items():
        if meta['method'] != 'OFM' or not meta.get('worker_failure') or meta.get('verified'):
            continue
        state = meta.get('recovery', {})
        recovery_rows.append(dict(pde=meta['pde'], task=meta['task'], split=meta['split'],
            shard=key, state=recovery_status(meta, snapshot), host=state.get('source_host', state.get('host')),
            pid=state.get('pid'), last_step=state.get('last_step'),
            saved=state.get('saved', sum(c['run_key'] == key for c in snapshot['cases'])),
            captured_at=snapshot['captured_at']))
    files = {
        'latest_source_snapshot.json.gz': gzip.compress(json_bytes(snapshot)),
        'metrics.csv': csv_bytes(all_rows), 'metrics.json': json_bytes(all_rows),
        'metrics_original.csv': csv_bytes(original_all_rows),
        'shard_recovery.csv': csv_bytes(recovery_rows, ['pde', 'task', 'split', 'shard', 'state', 'host', 'pid', 'last_step', 'saved', 'captured_at']),
        'progress.csv': csv_bytes(rows, ['method', 'pde', 'task', 'split', 'status', 'expected_n',
            'n_saved', 'n_verified', 'n_finite', 'n_failed', 'n_unverified', 'n_pending',
            'worker_failure_runs', 'verified_shards', 'recovery_active_shards', 'recovery_queued_shards',
            'recovery_failed_shards', 'recovery_unassigned_shards', 'resource_review_runs', 'result_version', 'original_n_failed', 'captured_at']),
        'latest_summary.json': json_bytes(summary), 'live_report.md': '\n'.join(report).encode()}
    for name, data in files.items():
        atomic_write(output / name, data)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='server216')
    parser.add_argument('--root', default=DEFAULT_ROOT, help='Server result root')
    parser.add_argument('--remote-script', default=REMOTE_SCRIPT)
    parser.add_argument('--ssh-control-path', help='Optional SSH multiplex socket')
    parser.add_argument('--paper', type=Path, default=Path.home() / 'C04Papers/fm4pde_jmlr/fm4pde_jmlr_revision.tex')
    parser.add_argument('--output', type=Path, default=REPO / 'reports/five_method_comparison_20260921')
    parser.add_argument('--original-only', action='store_true', help='Report original evaluation configurations without selecting audited repairs')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--snapshot', type=Path, help='Recompute offline from a saved JSON or JSON.gz snapshot')
    source.add_argument('--local-root', type=Path, help='Read a locally mounted result root without SSH')
    source.add_argument('--collect', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.collect:
        sys.stdout.write(json.dumps(collect(args.root, not args.original_only), allow_nan=False))
        return
    if not args.paper.is_file():
        parser.error(f'Manuscript missing: {args.paper}; use --paper PATH')
    if args.snapshot:
        opener = gzip.open if args.snapshot.suffix == '.gz' else open
        with opener(args.snapshot, 'rt') as handle:
            snapshot = json.load(handle)
    elif args.local_root:
        snapshot = collect(args.local_root, not args.original_only)
    else:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                   '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
        socket = args.ssh_control_path
        if socket is None:
            for candidate in ('/tmp/ddis216-reallocation-20260921', '/tmp/ddis216-recovered-20260919', '/tmp/ddis216-20260919-ssh'):
                if Path(candidate).exists():
                    socket = candidate
                    break
        if socket:
            command += ['-S', socket]
        remote_args = ['python3', args.remote_script, '--collect', '--root', args.root]
        if args.original_only:
            remote_args.append('--original-only')
        command += [args.host, shlex.join(remote_args)]
        print(f'Reading latest metadata from {args.host} ...', flush=True)
        process = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if process.returncode:
            raise RuntimeError(f'SSH collection failed; existing CSV unchanged.\n{process.stderr.strip()}')
        snapshot = json.loads(process.stdout)
    summary = export(snapshot, args.paper.expanduser().resolve(), args.output.expanduser().resolve(), args.original_only)
    print('Snapshot:', summary['captured_at'])
    print(f'Result selection: {summary["result_selection"]}; repaired settings={summary["repaired_settings"]}')
    print('verified = audited records (including documented failures); successful = finite predictions, not an accuracy threshold.')
    print('complete settings = all 100 cases audited; OFM uses 10-case shards. Shard errors remain counted until recovery is verified.')
    for method, counts in summary['methods'].items():
        print(f'{method:8s} saved={counts["n_saved"]:4d}  verified={counts["n_verified"]:4d}/{counts["expected_n"]}'
              f'  complete settings={counts["completed_cells"]}/{counts["cells"]}'
              f'  successful={counts["n_finite"]}'
              f'  numeric failures={counts["n_failed"]}  shard errors={counts["worker_failure_runs"]}'
              + (f'  [recovering={counts["recovery_active_shards"]}; queued={counts["recovery_queued_shards"]};'
                 f' retry failed={counts["recovery_failed_shards"]}; unassigned/stale={counts["recovery_unassigned_shards"]}]'
                 if counts['worker_failure_runs'] else '')
              + (f'  [original failures={counts["original_n_failed"]}; repaired settings={counts["repaired_settings"]}]'
                 if counts['repaired_settings'] else ''))
        if method == 'OFM':
            print(f'         verified shards={counts["verified_shards"]}/{counts["expected_n"] // 10}')
    print(f'Wrote {summary["metric_rows"]} rows: {args.output.resolve() / "metrics.csv"}')
    print('Progress:', args.output.resolve() / 'progress.csv')
    print('Original configurations:', args.output.resolve() / 'metrics_original.csv')
    if summary['warnings']:
        print(f'{len(summary["warnings"])} collection/version warnings; see latest_summary.json')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
