"""Apply the user-authorized sustained-plateau rule to existing and future jobs."""
import argparse
import json
from pathlib import Path
import time
from early_stop import window_plateau, write_json
from plateau_priority import selection, stop_validated_job, identity

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--orchestration', type=Path, required=True)
a = p.parse_args()
settings = json.loads((a.orchestration / 'configs/early_stopping_v2.json').read_text())
audit = a.root / 'jobs/monitor_training_v2'
audit.mkdir(exist_ok=True)
while True:
    for folder in sorted((a.root / 'jobs').iterdir()):
        if not folder.name.startswith(('ddis_', 'fundps_', 'flow_', 'surrogate_')):
            continue
        state = folder / 'early_stopping'
        if (state / 'completed.json').exists() or not (state / 'best.json').exists():
            continue
        if (state / 'failure.json').exists():
            continue
        history, best, checkpoint = selection(state)
        stop, evidence = window_plateau(history, settings)
        write_json(audit / (folder.name + '.json'), {'stop_eligible': stop, 'evidence': evidence,
            'best_checkpoint': str(checkpoint), 'best_validation_loss': best['validation_loss']})
        if stop:
            trainer = identity(json.loads((state / 'process.json').read_text())['pid'])
            if not trainer or trainer['state'] in ('T', 'Z'):
                continue
            policy = dict(settings, jobs={folder.name: {}})
            result = stop_validated_job(a.root, folder.name, policy, a.orchestration, controllers=[])
            print(json.dumps({'job': folder.name, 'completion': result}), flush=True)
    time.sleep(30)
