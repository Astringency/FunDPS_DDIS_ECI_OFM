"""Resume resource-limited OFM shards on a mostly empty GPU; keep all evidence."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
root = OUT / 'evaluation_v2/ofm/ofm'
state = OUT / 'jobs/recover_ofm_resources'
state.mkdir(exist_ok=True)
while True:
    pending = []
    for failure in root.glob('*/*/*/shard_*/worker_failure.json'):
        folder = failure.parent
        if (folder / 'external_assignment.json').exists(): continue
        if (folder / 'verified_summary.json').exists() or (folder / 'resource_needs_review.json').exists(): continue
        logs = [folder / name for name in ('sampling.log', 'profile.log')]
        text = '\n'.join(log.read_text(errors='replace') for log in logs if log.exists()).lower()
        if 'out of memory' in text or 'ofm_resource_limit' in text:
            pending.append(folder)
    if not pending:
        time.sleep(30); continue
    gpu_lock = None
    for gpu in [7, 0, 5, 3, 2, 1, 4, 6]:
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
        if free < 76000 or os.getloadavg()[0] >= 110: continue
        candidate = (OUT / 'locks' / f'ofm_resource_recovery_gpu_{gpu}.lock').open('w')
        try: fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            candidate.close(); continue
        gpu_lock = candidate; break
    if gpu_lock is None:
        (state / 'status').write_text('Waiting for a GPU with at least 76000 MiB free for OFM resource retry\n')
        time.sleep(30); continue
    folder = pending[0]
    audit = folder / 'resource_retry_audit'
    audit.mkdir(exist_ok=True)
    attempt = len(list(audit.glob('attempt_*'))) + 1
    if attempt > 3:
        (folder / 'resource_needs_review.json').write_text(json.dumps({'reason': 'Resource retries exhausted; not a numerical sampler failure'}))
        gpu_lock.close(); continue
    dest = audit / f'attempt_{attempt}'
    dest.mkdir()
    for p in folder.glob('case_*.json'):
        record = json.loads(p.read_text())
        if record['status'] == 'failed' and 'out of memory' in record.get('error', '').lower():
            p.rename(dest / p.name)
    cmd = json.loads((folder / 'command.json').read_text())
    (dest / 'launch.json').write_text(json.dumps({'gpu': gpu, 'free_mib': free, 'command': cmd,
        'time': datetime.now(timezone.utc).isoformat()}, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    (state / 'status').write_text(f'Resuming {folder} on GPU {gpu}, attempt {attempt}\n')
    with (dest / 'sampling.log').open('w') as log:
        code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (dest / 'sampling.exit').write_text(str(code))
    if code == 0:
        with (dest / 'verification.log').open('w') as log:
            code = subprocess.run([str(BASE / 'venv-shared-prior/bin/python'), str(BASE / 'orchestration/adapters/validate_shared_prior.py'), '--output', str(folder)], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        (dest / 'verification.exit').write_text(str(code))
        if code == 0:
            (folder / 'worker_failure.json').rename(dest / 'original_worker_failure.json')
            print('RECOVERED', str(folder), flush=True)
    gpu_lock.close()
