"""User-authorized checkpoint stopping for already-running legacy supervisors.

Preserve official training code and the original supervisor exit evidence.
Only publish training completion after the trainer has exited and its selected
checkpoint is verified. Temporarily suspend dependent waiting controllers to
prevent intentional cancellation from being reported as an evaluation failure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from early_stop import write_json, window_plateau


def read(path):
    return json.loads(path.read_text())


def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        fields = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        argv = proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        return {'pid': pid, 'state': fields[0], 'parent': int(fields[1]),
                'start': fields[19], 'argv': argv}
    except FileNotFoundError:
        return None


def same_live(saved):
    now = identity(saved['pid'])
    return now is not None and now['state'] != 'Z' and now['start'] == saved['start'] and now['argv'] == saved['argv']


def plateau(history, spec, settings):
    if 'window_epochs' in settings:
        stop, evidence = window_plateau(history, settings)
        return stop, evidence
    anchor, bad = math.inf, 0
    for row in history:
        loss = row['validation_loss']
        if not math.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        if loss < anchor * (1 - settings['relative_min_delta']):
            anchor, bad = loss, 0
        else:
            bad += 1
    latest = history[-1]
    eligible = latest['progress'] >= spec['minimum_progress'] and latest['progress'] > spec['after_progress']
    return eligible and bad >= settings['patience'], bad


def selection(state):
    history = [json.loads(line) for line in (state / 'validation_history.jsonl').read_text().splitlines()]
    best = read(state / 'best.json')
    if best['validation_loss'] != min(row['validation_loss'] for row in history):
        raise ValueError('Selected checkpoint is not the validation minimum')
    checkpoint = Path(best['checkpoint']).resolve(strict=True)
    if (state / 'best_checkpoint').resolve(strict=True) != checkpoint or checkpoint.stat().st_size == 0:
        raise ValueError('Invalid best-checkpoint artifact')
    return history, best, checkpoint


def dependent_controllers(root, job, orchestration):
    method, pde = job.split('_', 1)
    method = 'ddis' if method == 'surrogate' else method
    rows = subprocess.check_output(['tmux', 'list-panes', '-a', '-F', '#{session_name}\t#{pane_pid}'], text=True)
    found = []
    for split in ('id', 'smooth', 'rough'):
        wanted = f'ddis_eval_{method}_{pde}_{split}_20260920'
        panes = [line.split('\t')[1] for line in rows.splitlines() if line.split('\t')[0] == wanted]
        if len(panes) != 1:
            raise ValueError(f'Missing evaluation controller: {wanted}')
        pid = panes[0]
        expected = ['bash', str(orchestration / 'adapters/formal_evaluate.sh'), method, pde, split]
        children = (Path('/proc') / pid / 'task' / pid / 'children').read_text().split()
        matches = [identity(int(child)) for child in children]
        matches = [item for item in matches if item and item['argv'] == expected and item['state'] != 'Z']
        if len(matches) != 1:
            raise ValueError(f'Controller identity mismatch: {wanted}')
        evaluation = root / 'jobs' / f'evaluate_{method}_{pde}_{split}'
        if (evaluation / 'started').exists():
            raise ValueError('Dependent sampling has already started')
        found.extend(matches)
    return found


def stop_validated_job(root, job, settings, orchestration, controllers=None, user_requested=False):
    folder = root / 'jobs' / job
    state = folder / 'early_stopping'
    history, best, checkpoint = selection(state)
    should_stop, bad = (True, None) if user_requested else plateau(history, settings['jobs'][job], settings)
    if not should_stop:
        raise ValueError('Checkpoint does not meet the authorized plateau policy')
    if (state / 'failure.json').exists() or (state / 'completed.json').exists():
        raise ValueError('Supervisor has already ended; inspect its original outcome')
    trainer = identity(read(state / 'process.json')['pid'])
    metadata = read(state / 'supervisor.json')
    if not trainer or trainer['state'] == 'Z' or trainer['argv'] != metadata['command']:
        raise ValueError('Trainer identity mismatch')
    supervisor = identity(trainer['parent'])
    if not supervisor or str(state) not in supervisor['argv'] or not any(x.endswith('/early_stop.py') for x in supervisor['argv']):
        raise ValueError('Supervisor identity mismatch')
    if trainer['state'] == 'T' and not user_requested:
        return None
    controllers = dependent_controllers(root, job, orchestration) if controllers is None else controllers
    audit = state / ('user_requested_stop' if user_requested else 'manual_plateau_stop')
    audit.mkdir(exist_ok=False)
    write_json(audit / 'intent.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'reason': settings['reason'], 'policy': settings, 'latest_validation': history[-1],
        'consecutive_checks_without_significant_improvement': bad,
        'trainer': trainer, 'supervisor': supervisor, 'dependent_controllers': controllers})
    suspended = []
    try:
        for controller in controllers:
            if not same_live(controller) or identity(controller['pid'])['state'] == 'T':
                raise ValueError('Controller changed before suspension')
            os.kill(controller['pid'], signal.SIGSTOP)
            suspended.append(controller)
        if not same_live(trainer) or not same_live(supervisor):
            raise ValueError('Training process changed before stopping')
        os.kill(supervisor['pid'], signal.SIGTERM)
        deadline = time.monotonic() + 60
        while same_live(trainer) or same_live(supervisor):
            if time.monotonic() > deadline:
                raise TimeoutError('Supervisor cancellation did not finish')
            time.sleep(.2)
        failure = read(state / 'failure.json')
        if failure['error'] != f'Supervisor received signal {int(signal.SIGTERM)}':
            raise ValueError(f'Unexpected supervisor failure: {failure}')
        history, best, checkpoint = selection(state)
        digest = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for block in iter(lambda: handle.read(8 << 20), b''):
                digest.update(block)
        completion = {'reason': 'user_requested_stop' if user_requested else 'user_requested_validation_plateau',
            'trainer_returncode': failure['trainer_returncode'], 'selected_checkpoint': best,
            'validation_checks': len(history), 'checkpoint_sha256': digest.hexdigest(),
            'manual_stop_audit': str(audit), 'policy': settings}
        write_json(audit / 'completed.json', completion)
        # Preserve the intentional signal exit in a clearly named audit file.
        # Original logs and the outer shell's actual .exit code are not changed.
        (state / 'failure.json').rename(audit / 'supervisor_signal_exit.json')
        write_json(state / 'completed.json', completion)
        (folder / 'status').write_text(('User-requested stop' if user_requested else 'User-authorized validation plateau stop') + '; best checkpoint selected\n')
        (folder / 'training_completed').write_text(datetime.now(timezone.utc).isoformat() + '\n')
        return completion
    finally:
        for controller in suspended:
            if same_live(controller):
                os.kill(controller['pid'], signal.SIGCONT)


def main(args):
    settings = read(args.policy)
    finished = set()
    while len(finished) != len(settings['jobs']):
        for job, spec in settings['jobs'].items():
            if job in finished:
                continue
            folder = args.root / 'jobs' / job
            state = folder / 'early_stopping'
            if (folder / 'training_completed').exists():
                finished.add(job)
                continue
            if (state / 'failure.json').exists():
                raise RuntimeError(f'Existing training failure: {job}')
            history, _, _ = selection(state)
            stop, bad = plateau(history, spec, settings)
            if stop:
                trainer = identity(read(state / 'process.json')['pid'])
                if trainer and trainer['state'] == 'T':
                    continue
                result = stop_validated_job(args.root, job, settings, args.orchestration)
                if result is None:
                    continue
                print(json.dumps({'job': job, 'completion': result}), flush=True)
                finished.add(job)
        time.sleep(settings['poll_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--orchestration', type=Path, required=True)
    main(parser.parse_args())
