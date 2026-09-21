"""Publish only batch sizes that pass the full-schedule per-PDE check."""
import json
from pathlib import Path
import time

OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
root = OUT / 'profiles/eci_batch_20260921'
target = OUT / 'jobs/eci_batch_policy.json'
while True:
    policy, evidence = {}, {}
    done = True
    for prior in ['fm4pde', 'ofm']:
        source = root / prior / 'full_schedule_agreement.json'
        done &= (root / prior / 'completed').exists()
        if not source.exists(): continue
        checks = json.loads(source.read_text())
        policy[prior], evidence[prior] = {}, {}
        for pde, check in checks.items():
            batch = check['batch']
            if not check['passed'] or len(check['agreement']) != batch: continue
            if not all(0 <= x['relative_difference'] < 1e-3 for x in check['agreement']): continue
            directory = root / prior / pde / f'full_b{batch}'
            verified = json.loads((directory / 'verified_summary.json').read_text())
            assert verified['cases'] == batch and verified['failures'] == 0
            records = [json.loads(p.read_text()) for p in directory.glob('case_*.json')]
            peak = max(x['peak_reserved_bytes'] for x in records)
            assert peak < 14 * 1024 ** 3  # Workers require >20 GiB free.
            policy[prior][pde] = batch
            evidence[prior][pde] = {'check': str(source), 'checkpoint_sha256': verified['checkpoint_sha256'],
                'peak_reserved_bytes': peak, 'max_prediction_relative_difference': max(x['relative_difference'] for x in check['agreement'])}
    policy['_evidence'] = evidence
    temp = target.with_suffix('.partial.json')
    temp.write_text(json.dumps(policy, indent=2))
    temp.replace(target)
    print(json.dumps({p: v for p, v in policy.items() if p != '_evidence'}), flush=True)
    if done: break
    time.sleep(20)
