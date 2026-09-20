"""Retire only unstarted legacy evaluation queues; preserve files and training."""
import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--apply', action='store_true')
a = p.parse_args()
names = subprocess.check_output(['tmux', 'list-sessions', '-F', '#{session_name}'], text=True).splitlines()
targets = []
for n in names:
    if n.startswith('ddis_eval_'):
        label = n.removeprefix('ddis_eval_').removesuffix('_20260920')
        state = a.root / 'jobs' / ('evaluate_' + label)
    elif n.startswith('ddis_shared_'):
        label = n.removeprefix('ddis_shared_').removesuffix('_20260920')
        state = a.root / 'jobs' / ('shared_' + label)
    else:
        continue
    if (state / 'started').exists():
        raise RuntimeError(f'Refusing to retire started job: {state}')
    targets.append((n, state))
print(json.dumps({'unstarted_queues': len(targets), 'apply': a.apply}))
if a.apply:
    # This monitor only controls early stopping and owns no trainer process.
    if 'ddis_plateau_priority_20260920' in names:
        subprocess.run(['tmux', 'kill-session', '-t', 'ddis_plateau_priority_20260920'], check=True)
    for name, state in targets:
        record = {'reason': 'User replaced evaluation matrix: forward/inverse; ECI-FM priority',
                  'session': name, 'utc': datetime.now(timezone.utc).isoformat()}
        (state / 'superseded_v2.json').write_text(json.dumps(record, indent=2) + '\n')
        subprocess.run(['tmux', 'kill-session', '-t', name], check=True)
    for name in ('ddis_summary_20260920', 'ddis_summary_expanded_20260920'):
        if name in names:
            subprocess.run(['tmux', 'kill-session', '-t', name], check=True)
