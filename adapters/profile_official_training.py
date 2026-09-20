"""Run the official trainer; optionally shorten only its budget for a memory probe."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--method', choices=['ddis', 'fundps'], required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--profile-steps', type=int)
    p.add_argument('--metrics', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    if a.profile_steps:
        from training import training_loop
        original = training_loop.training_loop
        def short_budget(**kwargs):
            kwargs['total_kimg'] = kwargs['batch_size'] * a.profile_steps / 1000
            return original(**kwargs)
        training_loop.training_loop = short_budget
    torch.cuda.reset_peak_memory_stats()
    script = a.repo / ('scripts/train/train.py' if a.method == 'ddis' else 'train.py')
    sys.argv = [str(script), '-c', str(a.config)]
    runpy.run_path(str(script), run_name='__main__')
    a.metrics.write_text(json.dumps({'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'profile_steps': a.profile_steps}) + '\n')

if __name__ == '__main__':
    main()
