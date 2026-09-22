"""Launch named, persistent workers with explicit host runtime and exit logs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--probe', action='store_true')
    a = p.parse_args()
    runtime = json.loads(a.runtime.read_text())
    study = Path(runtime['root']) / 'diagnostics/fm_ofm_strength_refine_20260922'
    assert (study / 'migration_verified.json').exists()
    if not a.probe:
        result = json.loads((study / 'preflight.json').read_text())['result']
        assert result['failures'] == result['over_1000'] == 0
        assert result['peak_reserved_bytes'] < runtime['min_free_mib'] * 1024 ** 2
    for gpu in ([0] if a.probe else [0, 1]):
        label = 'preflight' if a.probe else 'refine'
        name = f'ddis_fm_ofm_{label}_{runtime["host"]}_gpu{gpu}_20260922'
        folder = study / 'controllers' / name
        folder.mkdir(parents=True, exist_ok=True)
        command = ['env', 'FM_OFM_RUNTIME=' + str(a.runtime), 'OMP_NUM_THREADS=1',
            'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'MPLBACKEND=Agg',
            'nice', '-n', str(runtime['nice']), runtime['python'], '-u',
            runtime['adapters'] + '/refine_fm_ofm_strength.py', '--worker-gpu', str(gpu)]
        if a.probe:
            command.append('--probe')
        (folder / 'launch.json').write_text(json.dumps(dict(command=command, runtime=runtime), indent=2) + '\n')
        script = shlex.join(command) + ' > ' + shlex.quote(str(folder / 'controller.log')) + ' 2>&1\n'
        script += 'code=$?\nprintf "%s\\n" "$code" > ' + shlex.quote(str(folder / 'controller.exit')) + '\nexit "$code"\n'
        path = folder / 'launch.sh'
        assert not path.exists(), f'Already launched: {name}'
        path.write_text(script)
        subprocess.run(['tmux', 'new-session', '-d', '-s', name, shlex.join(['bash', str(path)])], check=True)
        print(name, flush=True)


if __name__ == '__main__':
    main()
