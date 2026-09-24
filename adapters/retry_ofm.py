"""Retry an OFM numerical failure using the manuscript's declared step sequence."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace

STEPS=(1e-4,1e-5,1e-6)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True,help='Failed case_<id>.json with its original run.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step-size',type=float,choices=STEPS,help=argparse.SUPPRESS)
    a=p.parse_args();case=json.loads(a.original.read_text());run=json.loads((a.original.parent/'run.json').read_text())
    if case['status']!='failed' or run['method']!='ofm' or run['prior']!='ofm':
        raise ValueError('Only a recorded OFM numerical failure can be retried')
    if not any(s in case.get('error','').lower() for s in ['underflow in dt','nonfinite','non-finite','nan','infinite']):
        raise ValueError('This retry protocol is for numerical failures, not resource failures')
    if a.step_size is None:
        a.output.mkdir(parents=True,exist_ok=True)
        for lr in STEPS:
            dest=a.output/f'lr_{lr:g}';log=a.output/f'lr_{lr:g}.log'
            with log.open('w') as stream:
                proc=subprocess.run([sys.executable,__file__,'--original',str(a.original),'--output',str(dest),'--step-size',str(lr)],stdout=stream,stderr=subprocess.STDOUT)
            if proc.returncode==0:
                (a.output/'accepted.json').write_text(json.dumps(dict(original=str(a.original),original_sha256=hashlib.sha256(a.original.read_bytes()).hexdigest(),result=str(dest),step_size=lr,selection='first finite verified prediction; no error-based selection'),indent=2)+'\n')
                return
        raise RuntimeError('All declared retries failed; original results are preserved')
    from evaluate_shared_prior import main as evaluate
    from validate_shared_prior import validate
    names='prior method pde split task seed device profile eci_steps eci_mix eci_batch_size fm_batch_size fm_steps langevin_steps hutchinson noise_variance trace_native data_size eci_resample'.split()
    values={k:run[k] for k in names if k in run}
    values.setdefault('data_size',100);values.setdefault('fm_batch_size',1);values.setdefault('eci_resample',None)
    for key in ['assets','source','fm4pde','ofm','eci','checkpoint','fm_overrides']:
        values[key]=Path(run[key]) if run.get(key) else None
    values.update(offset=case['sample_id'],count=1,case_indices=[case['sample_id']],output=a.output,
                  ofm_lr=a.step_size,ofm_lr_final=.8*a.step_size,ofm_trace=True,
                  original_case=str(a.original),original_case_sha256=hashlib.sha256(a.original.read_bytes()).hexdigest())
    evaluate(SimpleNamespace(**values));validate(a.output)
    result = json.loads((a.output / f"case_{case['sample_id']:03d}.json").read_text())
    if result['status'] != 'ok':
        raise RuntimeError('This retry still failed numerically; continue the declared sequence')

if __name__=='__main__':main()
