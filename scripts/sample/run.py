"""Run one baseline over a manuscript comparison group."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--method',choices=['ddis','fundps','eci-fm','eci-ofm','ofm','fm4pde-ofm'],required=True)
p.add_argument('--pdes',nargs='+');p.add_argument('--tasks',nargs='+',choices=['forward','inverse','both'],default=['forward','inverse','both'])
p.add_argument('--splits',nargs='+',choices=['id','smooth','rough','rough2','rough3'],default=['id','smooth','rough'])
p.add_argument('--assets',type=Path,default=ROOT/'data/shared_prior_assets');p.add_argument('--data',type=Path,default=ROOT/'data/compact');p.add_argument('--hf-data',type=Path,default=ROOT/'data/preprocessed')
p.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints');p.add_argument('--output',type=Path,default=ROOT/'outputs/comparison')
p.add_argument('--fm4pde',type=Path,default=Path(os.environ.get('FM4PDE_ROOT',str(ROOT.parent/'FM4PDE'))))
p.add_argument('--count',type=int,default=100);p.add_argument('--device',default='cuda:0');p.add_argument('--plan-only',action='store_true')
a=p.parse_args();a.pdes=a.pdes or (['poisson','helmholtz'] if a.method in ['ddis','fundps'] else ['poisson','helmholtz','darcy','nsnonbounded','burger'])
if a.method in ['ddis','fundps'] and (set(a.pdes)-{'poisson','helmholtz'} or set(a.splits)-{'id','smooth','rough'}):p.error('The paper DDIS/FunDPS profiles cover Poisson and Helmholtz on ID/Smooth/Rough.')
for q in a.pdes:
 for task in (['both'] if q=='burger' else a.tasks):
  for split in a.splits:
   out=a.output/a.method/q/task/split
   if a.method in ['ddis','fundps']:
    repo='DDIS' if a.method=='ddis' else 'FunDPS'
    cmd=[sys.executable,str(ROOT/'adapters/evaluate_diffusion.py'),'--method',a.method,'--repo',str(ROOT/'official'/repo),'--config',str(ROOT/f'configs/evaluation/{a.method}_{q}_{split}.yaml'),'--checkpoint',str(a.checkpoints/a.method/(q+'.pkl')),'--source',str(a.data/q),'--data-path',str(a.hf_data/split/'data/DiffPDE'/f'{q}_test_hf'),'--split',split,'--task',task,'--count',str(a.count),'--output',str(out)]
    if a.method=='ddis':cmd+=['--surrogate',str(a.checkpoints/'surrogate'/(q+'.pt'))]
   else:
    method={'eci-fm':'eci','eci-ofm':'eci','fm4pde-ofm':'fm4pde','ofm':'ofm'}[a.method];prior='fm4pde' if a.method=='eci-fm' else 'ofm'
    checkpoint=a.assets/'weights'/(q+'.pth') if prior=='fm4pde' else a.checkpoints/'ofm'/(q+'.pt')
    cmd=[sys.executable,str(ROOT/'adapters/evaluate_shared_prior.py'),'--prior',prior,'--method',method,'--pde',q,'--task',task,'--split',split,'--count',str(a.count),'--checkpoint',str(checkpoint),'--assets',str(a.assets),'--source',str(a.data/q),'--fm4pde',str(a.fm4pde),'--ofm',str(ROOT/'official/OFM'),'--eci',str(ROOT/'official/ECI'),'--output',str(out),'--device',a.device]
    if method=='eci':
     c=yaml.safe_load((ROOT/'configs/eci_sampling.yaml').read_text())[prior][q];cmd+=['--eci-steps',str(c['steps']),'--eci-mix',str(c['mixing'])]
     if c['resample']:cmd+=['--eci-resample',str(c['resample'])]
    if method=='fm4pde':
     values=yaml.safe_load((a.fm4pde/'configs/ofm_guidance.yaml').read_text())[q].get(task)
     if values is None:continue
     c=dict(zip(['zeta_obs_a','zeta_obs_u','zeta_pde','clip_threshold'],values))
     override=out/'guidance.json'
     if not a.plan_only:out.mkdir(parents=True,exist_ok=True);override.write_text(json.dumps(c,indent=2)+'\n')
     cmd+=['--fm-overrides',str(override)]
   print(shlex.join(cmd),flush=True)
   if not a.plan_only:
    subprocess.run(cmd,check=True,cwd=ROOT)
    check='validate_evaluation.py' if a.method in ['ddis','fundps'] else 'validate_shared_prior.py'
    validation=[sys.executable,str(ROOT/'adapters'/check),'--output',str(out)]
    if a.method in ['ddis','fundps']:validation+=['--source',str(a.data/q),'--split',split,'--count',str(a.count)]
    subprocess.run(validation,check=True,cwd=ROOT)
