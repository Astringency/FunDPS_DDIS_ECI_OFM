"""Retire completed dedicated queues and reuse their GPU slots for native OFM."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
state = OUT / 'jobs/rebalance_sampling'
state.mkdir(exist_ok=True)
# Only dedicated queues are retired. General ECI-FM workers already fall
# through to remaining shared-prior work once ECI-FM finishes.
targets = {
    'ddis_eci_ofm_v2_gpu1': ('eci', None, 1, 0),
    'ddis_eci_ofm_extra_gpu3': ('eci', None, 3, 0),
    'ddis_diffusion_evaluation_v2_0': ('ddis', None, 0, 0),
    'ddis_diffusion_evaluation_v2_1': ('ddis', 'helmholtz', 2, 2),
    'ddis_diffusion_evaluation_v2_2': ('fundps', None, 4, 1),
    'ddis_diffusion_evaluation_v2_3': ('fundps', None, 6, 1),
    'ddis_diffusion_evaluation_v2_4': ('ddis', None, 4, 2),
    'ddis_diffusion_evaluation_v2_5': ('ddis', None, 6, 2),
    'ddis_diffusion_evaluation_v2_6': ('fundps', 'helmholtz', 1, 1),
}
finished, released = set(), {}
if (state / 'state.json').exists():
    previous = json.loads((state / 'state.json').read_text())
    finished, released = set(previous['finished']), previous['released']
while True:
    matrix = json.loads((BASE / 'orchestration/configs/evaluation_matrix_v2.json').read_text())['evaluations']
    sessions = subprocess.check_output(['tmux', 'list-sessions', '-F', '#{session_name}'], text=True).splitlines()
    for session, (method, pde, gpu, slot) in targets.items():
        if session in finished: continue
        rows = [r for r in matrix if r['method'] == method and (method != 'eci' or r['prior'] == 'ofm') and (not pde or r['pde'] == pde)]
        def done(r):
            root = OUT / 'evaluation_v2' / ('ofm' if method == 'eci' else 'diffusion') / method / r['pde'] / r['task'] / r['split']
            paths = root.glob('shard_*/verified_summary.json') if method == 'eci' else root.glob('case_*/verified_summary.json')
            key = 'cases' if method == 'eci' else 'verified_cases'
            return sum(json.loads(p.read_text())[key] for p in paths) == r['count']
        if not rows or not all(done(r) for r in rows): continue
        if session in sessions:
            active_child = False
            for pane in subprocess.check_output(['tmux', 'list-panes', '-t', '=' + session, '-F', '#{pane_pid}'], text=True).split():
                for pid in Path('/proc', pane, 'task', pane, 'children').read_text().split():
                    cmd = Path('/proc', pid, 'cmdline').read_bytes()
                    if b'evaluate_' in cmd:
                        active_child |= bool(Path('/proc', pid, 'task', pid, 'children').read_text().strip())
            if active_child: continue
            subprocess.run(['tmux', 'kill-session', '-t', '=' + session], check=True)
        finished.add(session)
        released[session] = {'gpu': gpu, 'slot': slot, 'time': time.time(), 'replacement': None}
        print('COMPLETED', session, flush=True)
    native_done = sum(json.loads(p.read_text())['cases'] for p in (OUT / 'evaluation_v2/ofm/ofm').glob('*/*/*/shard_*/verified_summary.json'))
    for session, info in released.items():
        if info['replacement'] or native_done >= 2700: continue
        gpu, slot = info['gpu'], info['slot']
        sessions = subprocess.check_output(['tmux', 'list-sessions', '-F', '#{session_name}'], text=True).splitlines()
        native_on_gpu = sum(bool(re.search(rf'native_ofm.*_gpu{gpu}(?:_|$)', s)) for s in sessions)
        if native_on_gpu >= 1: continue
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits']))
        if free < 40960 or os.getloadavg()[0] >= 110: continue
        name = f'ddis_native_ofm_rebalanced_gpu{gpu}_slot{slot}'
        job = OUT / 'jobs' / name.removeprefix('ddis_')
        job.mkdir(exist_ok=True)
        cmd = [str(BASE / 'venv-shared-prior/bin/python'), '-u', str(BASE / 'orchestration/adapters/evaluate_v2_worker.py'),
               '--gpu', str(gpu), '--slot', str(slot), '--ofm-only']
        (job / 'launch.json').write_text(json.dumps({'command': cmd, 'replaces': session, 'free_mib': free}, indent=2))
        shell = shlex.join(cmd) + ' > ' + shlex.quote(str(job / 'controller.log')) + ' 2>&1; code=$?; echo "$code" > ' + shlex.quote(str(job / 'controller.exit')) + '; exit "$code"'
        subprocess.run(['tmux', 'new-session', '-d', '-s', name, shell], check=True)
        info['replacement'] = name
        print('REALLOCATED', session, name, flush=True)
    temporary = state / 'state.partial.json'
    temporary.write_text(json.dumps({'finished': sorted(finished), 'released': released}, indent=2))
    temporary.replace(state / 'state.json')
    if len(finished) == len(targets) and native_done >= 2700: break
    time.sleep(30)
