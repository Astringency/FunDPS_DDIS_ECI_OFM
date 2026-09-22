"""Retry only documented OFM numeric failures using official sampler parameters.

Original predictions and records are never edited. Selection is the first finite
retry in a declared step-size sequence, not the lowest ground-truth error.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


NAME = 'ofm_numeric_repair_20260922'
PARAMETERS = (1e-4, 1e-5, 1e-6)


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failures(root):
    for verified in sorted((root / 'evaluation_v2/ofm/ofm').rglob('verified_summary.json')):
        folder = verified.parent
        run = json.loads((folder / 'run.json').read_text())
        for path in sorted(folder.glob('case_*.json')):
            record = json.loads(path.read_text())
            if record['status'] != 'failed':
                continue
            if not any(s in record['error'].lower() for s in ('nonfinite', 'non-finite', 'underflow in dt', 'nan', 'infinite')):
                continue
            yield path, run, record


def location(root, run, record):
    return root / 'diagnostics' / NAME / run['pde'] / run['task'] / run['split'] / f"case_{record['sample_id']:03d}"


def sample(a):
    from evaluate_shared_prior import main
    run = json.loads((a.original.parent / 'run.json').read_text())
    record = json.loads(a.original.read_text())
    assert record['status'] == 'failed' and run['method'] == run['prior'] == 'ofm'
    keys = ('prior method pde split task seed device profile eci_steps eci_mix eci_batch_size '
            'fm_steps langevin_steps hutchinson noise_variance fm_overrides trace_native').split()
    values = {k: run[k] for k in keys}
    values.update(offset=record['sample_id'], count=1, case_indices=[record['sample_id']],
        output=a.output, assets=a.root / 'data/shared_prior_assets', source=a.root / 'data/compact' / run['pde'],
        ofm_lr=a.lr, ofm_lr_final=a.lr * 0.8, ofm_trace=True,
        original_case=str(a.original), original_case_sha256=digest(a.original))
    # Relocate published server197 runs using the matching local official files.
    base = Path(__file__).resolve().parents[2]
    values.update(fm4pde=base / 'official/FM4PDE-cbe627c', ofm=base / 'official/OFM', eci=base / 'official/ECI')
    checkpoint = Path(run['checkpoint'])
    if not checkpoint.exists():
        checkpoint = a.root / 'training/flow' / run['pde'] / checkpoint.name
    assert digest(checkpoint) == run['checkpoint_sha256']
    values['checkpoint'] = checkpoint
    main(SimpleNamespace(**values))


def worker(a):
    repair_root = a.root / 'diagnostics' / NAME
    repair_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        MPLBACKEND='Agg', DDIS_OFM_MEMORY_SLOT=str(a.slot), DDIS_OFM_RECOVERY_SLOT_ALLOWED='1',
        DDIS_OFM_ACTIVATION_CHECKPOINT='1', DDIS_OFM_MAX_ALLOCATED_GIB='12')
    pilot = {('forward', 'id', 2), ('forward', 'id', 3), ('forward', 'rough', 2)}
    # The sole inverse failure also participates in the pilot.
    failed_here = set()
    while True:
        work = list(failures(a.root))
        if a.pilot:
            work = [x for x in work if (x[1]['task'], x[1]['split'], x[2]['sample_id']) in pilot or x[1]['task'] == 'inverse']
        elif not (repair_root / 'pilot_accepted.json').exists():
            print('Waiting for independently verified pilot acceptance', flush=True)
            time.sleep(30)
            continue
        selected = None
        for original, run, record in work:
            folder = location(a.root, run, record)
            if (folder / 'accepted.json').exists() or str(original) in failed_here:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            claim = (folder / 'claim.lock').open('w')
            try:
                fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                claim.close()
                continue
            if (folder / 'accepted.json').exists():
                claim.close()
                continue
            selected = original, run, record, folder, claim
            break
        if selected is None:
            if a.pilot:
                return
            verified = list((a.root / 'evaluation_v2/ofm/ofm').rglob('verified_summary.json'))
            if len(verified) == 270 and all((location(a.root, r, c) / 'accepted.json').exists() for _, r, c in work):
                print('All original shards verified and every numeric failure repaired', flush=True)
                return
            time.sleep(30)
            continue
        original, run, record, folder, claim = selected
        while int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu), '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'])) < 30720 or os.getloadavg()[0] >= 110:
            time.sleep(20)
        atomic(folder / 'original.json', dict(source=str(original.relative_to(a.root)), case=record,
            case_sha256=digest(original), run=run, parameters=list(PARAMETERS)))
        accepted = False
        for lr in PARAMETERS:
            output = folder / f'lr_{lr:g}'
            if (output / 'verified_summary.json').exists():
                rc = 0
            else:
                output.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(a.root),
                    '--original', str(original), '--output', str(output), '--lr', str(lr)]
                atomic(output / 'command.json', dict(command=cmd, environment={k: v for k, v in env.items()
                    if k.startswith('DDIS_') or k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}))
                with (output / 'sampling.log').open('a') as log:
                    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while proc.poll() is None:
                        atomic(folder / 'state.json', dict(state='sampling', lr=lr, pid=proc.pid,
                            controller_pid=os.getpid(), gpu=a.gpu, slot=a.slot, updated_at=time.time()))
                        time.sleep(15)
                rc = proc.returncode
                (output / 'sampling.exit').write_text(str(rc))
            if rc:
                atomic(folder / 'worker_error.json', dict(returncode=rc, output=str(output)))
                break  # Resource/runtime errors require review, not a silent parameter retry.
            with (output / 'verification.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__).with_name('validate_shared_prior.py')),
                    '--output', str(output)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            verified = json.loads((output / 'verified_summary.json').read_text())
            if verified['successes'] == 1 and verified['failures'] == 0:
                atomic(folder / 'accepted.json', dict(original_source=str(original.relative_to(a.root)),
                    original_sha256=digest(original), output=str(output.relative_to(a.root)), lr=lr,
                    lr_final=lr*0.8, selection='first finite in declared step-size sequence',
                    accepted_at=time.time()))
                accepted = True
                print('Repaired', original, 'lr', lr, flush=True)
                break
        atomic(folder / 'state.json', dict(state='verified' if accepted else 'needs_review', updated_at=time.time()))
        if not accepted:
            failed_here.add(str(original))
        claim.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--original', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--lr', type=float)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--slot', type=int, choices=(0, 2), default=0)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    if a.original:
        assert a.output is not None and a.lr in PARAMETERS
        sample(a)
    else:
        worker(a)
