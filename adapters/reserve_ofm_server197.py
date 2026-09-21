"""Reserve Darcy forward shards centrally; legacy workers recognize this PID."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
job = a.root / 'jobs/ofm_server197_20260921'
job.mkdir(parents=True, exist_ok=True)
prefix = 'evaluation_v2/ofm/ofm/darcy/forward'
shards = []
active = {int(x) for x in subprocess.check_output(['nvidia-smi',
    '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()}
for split in ('id', 'smooth', 'rough'):
    for offset in range(0, 100, 10):
        relative = f'{prefix}/{split}/shard_{offset:03d}'
        dest = a.root / relative
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'verified_summary.json').exists():
            continue
        waiting = []
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                args = (proc / 'cmdline').read_bytes().decode().split('\0')
                if '--output' not in args or not any(x.endswith('/evaluate_shared_prior.py') for x in args):
                    continue
                output = Path(args[args.index('--output') + 1])
                if output != dest and output != dest / 'resource_profile':
                    continue
                if int(proc.name) in active:
                    raise RuntimeError(f'Refusing to interrupt active GPU sampling: {proc.name}, {dest}')
                waiting.append(int(proc.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        previous = dest / 'claimed.json'
        if previous.exists() and not (dest / 'claim_before_server197.json').exists():
            (dest / 'claim_before_server197.json').write_bytes(previous.read_bytes())
        claim = dict(pid=os.getpid(), host='server216', worker_host='server197', gpu='remote', time=time.time())
        previous.write_text(json.dumps(claim))
        (dest / 'external_assignment.json').write_text(json.dumps(claim))
        for pid in waiting:
            os.kill(pid, signal.SIGTERM)
        shards.append(dict(relative=relative, split=split, offset=offset))
(job / 'assignment.json').write_text(json.dumps(dict(host='server197', shards=shards), indent=2))
print(f'Reserved {len(shards)} shards for server197, keeper PID {os.getpid()}', flush=True)
while not all((a.root / item['relative'] / 'verified_summary.json').exists() for item in shards):
    time.sleep(60)
(job / 'completed.json').write_text(json.dumps(dict(time=time.time())))
