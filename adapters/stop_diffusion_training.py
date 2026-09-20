"""Explicit user-requested cancellation; preserve validated weights for sampling."""
import json
import os
from pathlib import Path
import signal
import subprocess
from datetime import datetime, timezone
from plateau_priority import identity, same_live, stop_validated_job
from early_stop import write_json

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
reason = 'User explicitly cancelled all running and pending DDIS/FunDPS training; use saved best weights for sampling.'
write_json(OUT / 'jobs/diffusion_training_cancelled.json', {'reason': reason, 'time': datetime.now(timezone.utc).isoformat()})
monitors = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    proc = identity(int(p.name))
    if proc and str(BASE / 'orchestration/adapters/monitor_training_v2.py') in proc['argv']:
        os.kill(proc['pid'], signal.SIGSTOP)
        monitors.append(proc)
try:
    # Queued auxiliary FNOs are also unnecessary without their cancelled priors.
    for method in ('ddis', 'fundps', 'surrogate'):
        pdes = ('darcy', 'nsnonbounded') if method == 'surrogate' else ('darcy', 'nsnonbounded', 'burger')
        for pde in pdes:
            folder = OUT / 'jobs' / f'{method}_{pde}'
            assert not (folder / 'early_stopping/process.json').exists(), f'Queued trainer unexpectedly started: {folder}'
            write_json(folder / 'cancelled.json', {'reason': reason, 'no_trained_checkpoint': True})
            folder.joinpath('status').write_text('Cancelled by user before training; no checkpoint available\n')
            session = f'ddis_train_{method}_{pde}_v2'
            if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
                subprocess.run(['tmux', 'kill-session', '-t', session], check=True)
            print('CANCELLED', method, pde, flush=True)
    for method in ('ddis', 'fundps'):
        for pde in ('poisson', 'helmholtz'):
            job = f'{method}_{pde}'
            folder = OUT / 'jobs' / job
            if not (folder / 'training_completed').exists():
                result = stop_validated_job(OUT, job, {'reason': reason}, BASE / 'orchestration',
                                            controllers=[], user_requested=True)
                print('STOPPED', job, json.dumps(result), flush=True)
            session = f'ddis_train_{job}_20260920'
            if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
                subprocess.run(['tmux', 'kill-session', '-t', session], check=True)
finally:
    for proc in monitors:
        if same_live(proc): os.kill(proc['pid'], signal.SIGCONT)
