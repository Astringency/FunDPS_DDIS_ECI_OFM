"""Relay completed, hash-checked validation exports from197 to216 via local SSH."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

REMOTE = '/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/fm_ofm_guidance_refine_20260922/validation_export'
CENTRAL = '/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919/diagnostics/fm_ofm_strength_refine_20260922/validation_export'
SSH = ['ssh', '-S', '/tmp/ddis216-reallocation-20260921', '-o', 'BatchMode=yes', 'server216']


def main(cache):
    cache.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes', 'server197:' + REMOTE + '/', str(cache) + '/'], check=True)
            for marker in sorted(cache.glob('*/*/export.json')):
                folder = marker.parent
                if (folder / '.published').exists():
                    continue
                expected = json.loads(marker.read_text())['truth_sha256']
                if hashlib.sha256((folder / 'truth.pt').read_bytes()).hexdigest() != expected:
                    continue
                relative = str(folder.relative_to(cache))
                staging = CENTRAL + '/incoming/' + relative
                subprocess.run(SSH + ['mkdir -p ' + shlex.quote(staging)], check=True)
                subprocess.run(['rsync', '-a', '-e', 'ssh -S /tmp/ddis216-reallocation-20260921 -o BatchMode=yes',
                    str(folder) + '/', 'server216:' + staging + '/'], check=True)
                code = '''from pathlib import Path
import hashlib,json
root=Path(ROOT)
stage=root/'incoming'/RELATIVE
dest=root/RELATIVE
assert hashlib.sha256((stage/'truth.pt').read_bytes()).hexdigest()==DIGEST
if not dest.exists():
 dest.parent.mkdir(parents=True,exist_ok=True)
 stage.rename(dest)
else:
 assert json.loads((dest/'export.json').read_text())['truth_sha256']==DIGEST
'''.replace('ROOT', repr(CENTRAL)).replace('RELATIVE', repr(relative)).replace('DIGEST', repr(expected))
                subprocess.run(SSH + ['python3 -c ' + shlex.quote(code)], check=True)
                (folder / '.published').write_text(str(time.time()))
                print('Published disjoint validation:', relative, flush=True)
            if len(list(cache.glob('*/*/.published'))) == 10:
                print('All ten OOD validation exports published', flush=True)
                return
            status = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'server197',
                'cat /research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/fm_ofm_guidance_refine_20260922/export.exit'],
                capture_output=True, text=True)
            if status.returncode == 0 and status.stdout.strip() != '0':
                raise RuntimeError('Validation export failed on197; inspect export.log')
        except subprocess.CalledProcessError as error:
            print('Transfer will retry:', error, flush=True)
        time.sleep(45)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    main(p.parse_args().cache)
