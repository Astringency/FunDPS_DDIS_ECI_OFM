"""Return independently verified server197 shards to the central server216 report."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

REMOTE = '/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ofm_sampling_20260921'
CENTRAL = '/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919'
SSH216 = ['ssh', '-S', '/tmp/ddis216-recovered-20260919', '-o', 'BatchMode=yes', 'server216']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes',
                f'server197:{REMOTE}/evaluation_v2/', str(a.cache / 'evaluation_v2') + '/'], check=True)
            for marker in sorted((a.cache / 'evaluation_v2').rglob('verified_summary.json')):
                folder = marker.parent
                if (folder / '.published_to_216').exists():
                    continue
                relative = str(folder.relative_to(a.cache))
                # Stage completely before atomic publication; preserve original errors.
                incoming = CENTRAL + '/jobs/ofm_server197_20260921/incoming/' + relative
                subprocess.run(SSH216 + ['mkdir -p ' + shlex.quote(incoming)], check=True)
                subprocess.run(['rsync', '-a', '-e', 'ssh -S /tmp/ddis216-recovered-20260919 -o BatchMode=yes',
                    str(folder) + '/', 'server216:' + incoming + '/'], check=True)
                code = '''import json, pathlib
root=pathlib.Path(ROOT)
relative=RELATIVE
incoming=root/'jobs/ofm_server197_20260921/incoming'/relative
dest=root/relative
archive=root/'diagnostics/ofm_before_server197_20260921'/relative
assert (incoming/'verified_summary.json').is_file()
v=json.loads((incoming/'verified_summary.json').read_text())
records=[json.loads(p.read_text()) for p in incoming.glob('case_*.json')]
assert sorted(r['sample_id'] for r in records)==sorted(v['sample_ids'])
assert len(records)==v['cases']==10
if not (dest/'verified_summary.json').exists():
 archive.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists(): dest.rename(archive)
 incoming.rename(dest)
 (dest/'returned_from_server197.json').write_text(json.dumps(dict(source=SOURCE,validation_host='server197')))
'''.replace('ROOT', repr(CENTRAL)).replace('RELATIVE', repr(relative)).replace('SOURCE', repr(REMOTE + '/' + relative))
                subprocess.run(SSH216 + ['python3 -c ' + shlex.quote(code)], check=True)
                (folder / '.published_to_216').write_text(str(time.time()))
                print('Published', relative, flush=True)
        except subprocess.CalledProcessError as error:
            print('Transfer will retry:', error, flush=True)
        if a.once:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
