"""Reuse measured memory for an identical, independently verified shard setup."""
import json
import math


def reuse_profile(output, current):
    if output.name != 'resource_profile' or not current.get('profile'):
        return False
    keys = ['prior', 'method', 'pde', 'task', 'split', 'checkpoint_sha256', 'truth_sha256',
            'assets_manifest_sha256', 'source_revisions', 'eci_steps', 'eci_mix', 'eci_batch_size',
            'fm_steps', 'langevin_steps', 'hutchinson', 'noise_variance', 'guidance_overrides']
    for marker in sorted(output.parent.parent.glob('shard_*/verified_summary.json')):
        try:
            previous = json.loads((marker.parent / 'run.json').read_text())
            previous.setdefault('eci_batch_size', 1)
            if any(previous.get(k) != current.get(k) for k in keys): continue
            summary = json.loads(marker.read_text())
            if summary['cases'] != previous['count']: continue
            records = [json.loads(p.read_text()) for p in marker.parent.glob('case_*.json')]
            records += [json.loads(p.read_text()) for p in (marker.parent / 'resource_profile').glob('case_*.json')]
            peaks = [x['peak_reserved_bytes'] for x in records if x.get('peak_reserved_bytes')]
            if not peaks: continue
        except (OSError, ValueError, KeyError):
            continue
        data = {'model_loaded': False, 'resource_profile_reused_from': str(marker),
                'measured_peak_reserved_bytes': max(peaks),
                'peak_reserved_bytes': math.ceil(max(peaks) * 1.25),
                'memory_margin_multiplier': 1.25, 'new_samples_evaluated': 0}
        # This is a memory estimate, not a new prediction or a numerical result.
        for name in ['model_loaded.json', 'resource_profile_reused.json']:
            temporary = output / (name + '.partial')
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(output / name)
        return True
    return False
