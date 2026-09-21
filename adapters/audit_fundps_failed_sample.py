"""Audit only FunDPS sample 6, as explicitly requested; never resample other IDs."""
import hashlib
import json
from pathlib import Path

from validate_evaluation import verify

ROOT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
original = ROOT / 'evaluation_v2/diffusion/fundps/poisson/forward/rough/case_006'
folder = ROOT / 'diagnostics/fundps_repair_20260921/pilot_weight10000/case_006'
old = json.loads((original / 'run.json').read_text())
run = json.loads((folder / 'run.json').read_text())
assert json.loads((original / 'case_006.json').read_text())['status'] == 'failed'
assert run['checkpoint_sha256'] == old['checkpoint_sha256']
assert run['case_index'] == 6 and run['count'] == 1 and not run['profile_only']
expected = dict(old['official_config'])
expected['outdir'] = str(folder)
expected['guidance'] = dict(type='dps', weights=[10000, 0, 0])
assert run['official_config'] == expected
verified = verify(folder, ROOT / 'data/compact/poisson', 'rough', 1)
record = json.loads((folder / 'case_006.json').read_text())
assert record['status'] == 'ok' and verified['successful_cases'] == 1
audit = dict(method='FunDPS', pde='poisson', task='forward', split='rough',
    repair_scope='selected_samples', sample_ids=[6],
    checkpoint_sha256=run['checkpoint_sha256'],
    parameters=dict(iterations=500, guidance_weights=[10000, 0, 0]),
    validation=dict(verified_predictions=1, failed=0, mean_percent=record['relative_l2_solution'] * 100),
    records=[dict(case=record, run=run, verified=verified, directory=str(folder.relative_to(ROOT)),
        prediction_sha256=hashlib.sha256(Path(record['prediction']).read_bytes()).hexdigest())])
report = ROOT / 'diagnostics/fundps_repair_20260921/report'
report.mkdir(exist_ok=True)
temporary = report / 'audit.partial.json'
temporary.write_text(json.dumps(audit, indent=2, allow_nan=False))
temporary.replace(report / 'audit.json')
print(json.dumps(audit['validation']))
