"""Prepare compact prior data and optional Hugging Face datasets for DDIS/FunDPS."""
import argparse,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'data')
p.add_argument('--fm4pde',type=Path,default=ROOT.parent/'FM4PDE');p.add_argument('--pdes',nargs='+',default=['poisson','helmholtz','darcy','nsnonbounded','burger'])
p.add_argument('--hf',action='store_true');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
for q in a.pdes:
 compact=a.output/'compact'/q
 cmd=[sys.executable,str(ROOT/'adapters'/('prepare_data.py' if q in ['poisson','helmholtz'] else 'prepare_extended_flow_data.py'))]
 cmd+=['export','--ddis',str(ROOT/'official/DDIS')] if q in ['poisson','helmholtz'] else ['--fm4pde',str(a.fm4pde)]
 cmd+=['--pde',q,'--raw',str(a.raw),'--output',str(compact)]
 print(shlex.join(cmd))
 if not a.plan_only and not (compact/'manifest.json').exists():subprocess.run(cmd,check=True)
 if a.hf and q in ['poisson','helmholtz']:
  for split in ['train','validation','id','smooth','rough']:
   name=f'{q}_hf' if split=='train' else f'{q}_test_hf'
   dest=a.output/'preprocessed'/split/'data/DiffPDE'/name
   cmd=[sys.executable,str(ROOT/'adapters/prepare_data.py'),'hf','--source',str(compact),'--split',split,'--output',str(dest),'--cache',str(a.output/'hf_cache')]
   print(shlex.join(cmd))
   if not a.plan_only and not (dest/'metadata.json').exists():subprocess.run(cmd,check=True)
   if split in ['train','validation'] and not a.plan_only:
    link=a.output/'surrogate_work/data/DiffPDE'/name;link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():link.symlink_to(dest.resolve(),target_is_directory=True)
