"""Train a prior or DDIS surrogate using validation-selected checkpoints."""
import argparse,shlex,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--method',choices=['ddis','fundps','ofm','surrogate'],required=True);p.add_argument('--pde',choices=['poisson','helmholtz','darcy','nsnonbounded','burger'],required=True)
p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'outputs/training');p.add_argument('--plan-only',action='store_true')
a=p.parse_args();q=a.pde
if a.method!='ofm' and q not in ['poisson','helmholtz']:p.error('DDIS/FunDPS training in this manuscript covers Poisson and Helmholtz.')
out=a.output/a.method/q;method='flow' if a.method=='ofm' else a.method
repo=ROOT/'official'/({'ddis':'DDIS','fundps':'FunDPS','ofm':'OFM','surrogate':'DDIS'}[a.method]);hf=a.data/'preprocessed';compact=a.data/'compact'/q
if a.method=='ofm':
 cmd=[sys.executable,str(ROOT/'adapters/train_flow.py'),'--ofm',str(repo),'--train',str(compact/'train.npy'),'--validation',str(compact/'validation.npy'),'--output',str(out),'--epochs','300','--batch','100']
else:
 prefix='ddis_surrogate' if a.method=='surrogate' else a.method;c=yaml.safe_load((ROOT/f'configs/training/{prefix}_{q}.yaml').read_text());c['exps_outdir' if a.method=='surrogate' else 'outdir']=str(out)
 if a.method!='surrogate':c['data']=str(hf/'train/data/DiffPDE'/f'{q}_hf')
 config=out/'config.yaml'
 if not a.plan_only:out.mkdir(parents=True,exist_ok=True);config.write_text(yaml.safe_dump(c,sort_keys=False))
 if a.method=='surrogate':cmd=[sys.executable,str(ROOT/'adapters/train_surrogate.py'),'--repo',str(repo),'--config',str(config),'--workdir',str(a.data/'surrogate_work')]
 else:cmd=[sys.executable,str(repo/('scripts/train/train.py' if a.method=='ddis' else 'train.py')),'-c',str(config)]
cmd=[sys.executable,str(ROOT/'adapters/early_stop.py'),'--method',method,'--policy',str(ROOT/'configs/early_stopping_v2.json'),'--state',str(out/'selection'),'--training-root',str(out),'--repo',str(repo),'--validation',str(hf/'validation/data/DiffPDE'/f'{q}_test_hf'),'--']+cmd
print(shlex.join(cmd))
if not a.plan_only:subprocess.run(cmd,check=True,cwd=repo)
