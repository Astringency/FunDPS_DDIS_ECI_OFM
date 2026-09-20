"""Queue the 63 additions while retaining the 24 original evaluations."""
import json
from pathlib import Path
import shlex
import subprocess


BASE = Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT = Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')


def main():
    matrix = json.loads((BASE / 'orchestration/configs/expanded_evaluation_matrix.json').read_text())
    pending = [item for item in matrix['evaluations'] if not item['existing_controller']]
    assert len(pending) == 63
    for item in pending:
        arguments = [item[key] for key in ('prior', 'method', 'pde', 'split')]
        label = '_'.join(arguments)
        session = 'ddis_shared_' + label + '_20260920'
        state = OUT / 'jobs' / ('shared_' + label)
        if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
            continue
        if (state / 'evaluation_completed').exists():
            continue
        if (state / 'queue_command.json').exists():
            raise RuntimeError(f'Existing queue record without a live session; inspect before retrying: {state}')
        state.mkdir(parents=True, exist_ok=True)
        command = ['bash', str(BASE / 'orchestration/adapters/formal_evaluate_shared_prior.sh'), *arguments]
        (state / 'queue_command.json').write_text(json.dumps(command, indent=2) + '\n')
        shell = (shlex.join(command) + ' > ' + shlex.quote(str(state / 'controller.log')) + ' 2>&1\n'
                 'task_code=$?\nprintf \'%s\\n\' "$task_code" > ' + shlex.quote(str(state / 'controller.exit')) + '\nexit "$task_code"')
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, shell], check=True)
        pane = subprocess.check_output(['tmux', 'display-message', '-p', '-t', session, '#{pane_pid}'], text=True).strip()
        (state / 'controller_pid').write_text(pane + '\n')
        print(session, pane, flush=True)


if __name__ == '__main__':
    main()
