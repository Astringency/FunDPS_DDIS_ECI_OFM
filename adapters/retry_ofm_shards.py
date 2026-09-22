"""Resume designated resource-failed shards; preserve successes and error evidence."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def atomic(path, value):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def active_outputs():
    result = set()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
            if any(x.endswith('/evaluate_shared_prior.py') for x in args) and '--output' in args:
                result.add(args[args.index('--output') + 1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--jobs', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--memory-slot', type=int, choices=(0, 2), default=2,
        help='Bounded runtime slot; slot 0 can share a GPU with existing slot 2 recovery.')
    a = p.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg', DDIS_OFM_MEMORY_SLOT=str(a.memory_slot),
        DDIS_OFM_RECOVERY_SLOT_ALLOWED='1', DDIS_OFM_ACTIVATION_CHECKPOINT='1',
        DDIS_OFM_MAX_ALLOCATED_GIB='12')
    jobs = json.loads(a.jobs.read_text())['shards']
    failed_this_run = set()
    while True:
        pending = [j for j in jobs if not (a.root / j['relative'] / 'verified_summary.json').exists()
                   and j['relative'] not in failed_this_run]
        if not pending:
            break
        # Finish nearly complete failed shards first, without dropping any ID.
        pending.sort(key=lambda j: -len(list((a.root / j['relative']).glob('case_*.json'))))
        active = active_outputs()
        selected = None
        for job in pending:
            dest = a.root / job['relative']
            if str(dest) in active:
                continue
            dest.mkdir(parents=True, exist_ok=True)
            claim = (dest / 'recovery_claim.lock').open('w')
            try:
                fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                claim.close()
                continue
            selected = job, dest, claim
            break
        if selected is None:
            time.sleep(20)
            continue
        job, dest, claim = selected
        while int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'])) < 30720 or os.getloadavg()[0] >= 110:
            time.sleep(20)
        state = dict(state='running', host=socket.gethostname(), gpu=a.gpu,
            controller_pid=os.getpid(), activation_checkpoint=True, maximum_allocated_gib=12)
        audit = dest / 'resource_retry_audit' / f'activation_retry_{time.time_ns()}'
        audit.mkdir(parents=True)
        command = job['command']
        assert command[command.index('--output') + 1] == str(dest)
        atomic(audit / 'command.json', dict(command=command, environment={k: v for k, v in env.items()
            if k.startswith('DDIS_') or k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}))
        with (audit / 'sampling.log').open('w') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                atomic(dest / 'recovery_state.json', dict(state, pid=proc.pid, updated_at=time.time()))
                time.sleep(15)
        (audit / 'sampling.exit').write_text(str(proc.returncode))
        if proc.returncode:
            atomic(dest / 'recovery_state.json', dict(state, state='failed', updated_at=time.time(),
                returncode=proc.returncode, log=str(audit / 'sampling.log')))
            failed_this_run.add(job['relative'])
            print('Retry needs review:', dest, flush=True)
            claim.close()
            continue
        atomic(dest / 'recovery_state.json', dict(state, state='validating', updated_at=time.time()))
        with (audit / 'verification.log').open('w') as log:
            subprocess.run([command[0], str(Path(__file__).with_name('validate_shared_prior.py')),
                '--output', str(dest)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if (dest / 'worker_failure.json').exists():
            (dest / 'worker_failure.json').rename(audit / 'previous_worker_failure.json')
        atomic(dest / 'recovery_state.json', dict(state, state='verified', updated_at=time.time()))
        print('Recovered and independently verified:', dest, flush=True)
        claim.close()
    if failed_this_run:
        raise RuntimeError(f'Resource retry did not finish {sorted(failed_this_run)}')


if __name__ == '__main__':
    main()
