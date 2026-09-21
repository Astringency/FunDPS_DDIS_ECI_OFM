"""Read actual processes and reserved-shard metadata without loading models."""
import argparse
import json
from pathlib import Path
import re
import socket
import time


def collect(root):
    active, controllers = {}, []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
            if any(x.endswith('/evaluate_shared_prior.py') for x in args) and '--output' in args:
                active[args[args.index('--output') + 1]] = int(proc.name)
            if any(x.endswith(('/ofm_server197_queue.py', '/retry_ofm_shards.py')) for x in args):
                if '--root' in args and args[args.index('--root') + 1] == str(root):
                    controllers.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    rows = []
    for item in json.loads((root / 'assignment.json').read_text())['shards']:
        folder = root / item['relative']
        state = dict(relative=item['relative'], host=socket.gethostname(), controller_pids=controllers,
            saved=len(list(folder.glob('case_*.json'))))
        retry = folder / 'recovery_state.json'
        retry = json.loads(retry.read_text()) if retry.exists() else {}
        if (folder / 'verified_summary.json').exists():
            state['state'] = 'verified'
        elif str(folder) in active:
            state.update(state='running', pid=active[str(folder)])
        elif retry.get('state') == 'failed':
            state.update(state='failed', log=retry.get('log'))
        else:
            state['state'] = 'queued' if controllers else 'unassigned'
        logs = [folder / 'sampling.log', *folder.glob('resource_retry_audit/activation_retry_*/sampling.log')]
        logs = [p for p in logs if p.exists()]
        if logs:
            log = max(logs, key=lambda p: p.stat().st_mtime)
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 8192))
                tail = handle.read().decode(errors='replace')
            matches = re.findall(r'OFM Langevin (\d+)/(\d+)', tail)
            state.update(log=str(log), last_step=matches[-1][0] if matches else None)
        rows.append(state)
    return dict(captured_at=time.time(), shards=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    print(json.dumps(collect(p.parse_args().root)))
