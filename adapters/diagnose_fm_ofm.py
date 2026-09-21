"""Run unchanged FM4PDE sampling with explicit OFM guidance configurations.

Loads one official prior once and retains every prediction, failure and trace.
New parameter versions remain separate from original evaluation_v2 records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from eci_batch_runtime import atomic_json
from native_diagnostics import trace_native
from shared_prior_runtime import case_ground_truth, common_masks, load_prior, native_noise_provider

BASE=Path('/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919')
OUT=Path('/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919')
CASES={'helmholtz': {'id':[51,73,0], 'smooth':[30,70,0], 'rough':[35,49,0]},
       'burger': {'rough':[41,46,54,57,59,70,74,86,99,80,94,0]},
       'nsnonbounded': {'rough':[49,0,1]}}
VARIANTS={'original':{},'clip50':{'clip_threshold':50.},'clip10':{'clip_threshold':10.},
          'clip2':{'clip_threshold':2.},'clip05':{'clip_threshold':.5},
          'obs01_clip10':{'clip_threshold':10.,'observation_multiplier':.1},
          'no_guidance':{'guidance_components':'noguide'}}


def main(args):
    task='both' if args.pde=='burger' else 'inverse'
    original_root=OUT/'evaluation_v2/ofm/fm4pde'/args.pde/task
    reference=original_root/'rough/shard_000/run.json'
    original=json.loads(reference.read_text())
    setup=SimpleNamespace(**original)
    for key in ('fm4pde','ofm','eci','checkpoint','assets','output','source'):
        setattr(setup,key,Path(getattr(setup,key)))
    setup.device='cuda'
    assert hashlib.sha256(setup.checkpoint.read_bytes()).hexdigest()==original['checkpoint_sha256']
    revisions={}
    for name in ('fm4pde','ofm','eci'):
        path=getattr(setup,name)
        assert not subprocess.check_output(['git','-C',str(path),'diff','--name-only','HEAD','--','*.py'],text=True).strip()
        revisions[name]=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'completed.json').exists():
        raise ValueError('Use a new directory for a new diagnostic run')
    torch.set_num_threads(2);torch.manual_seed(0)
    channels=1 if args.pde=='burger' else 2
    net,normalizer,payload,noise,prior=load_prior(setup,channels)
    from sampling.config import load_config
    from sampling.runner import run_single_ablation
    manifest=json.loads((setup.assets/'manifest.json').read_text())
    assets=manifest['pdes'][args.pde]
    splits=args.splits or list(CASES[args.pde])
    variants=args.variants or ['original','clip50','clip10','clip2','obs01_clip10']
    run=dict(pde=args.pde,task=task,checkpoint=str(setup.checkpoint),checkpoint_sha256=original['checkpoint_sha256'],
        official_revisions=revisions,steps=100,observations=500,noise=0,variants={v:VARIANTS[v] for v in variants},
        original_guidance_overrides=original['guidance_overrides'],splits=splits,all_cases=args.all_cases,
        sample_ids={s:list(range(100)) if args.all_cases else CASES[args.pde][s] for s in splits},
        note='Failure-triggered parameter diagnostics; not independent held-out model selection. Algorithms and prior weights unchanged.')
    run_path=args.output/'run.json'
    if run_path.exists():
        if not args.resume or json.loads(run_path.read_text())!=run:
            raise ValueError('Resuming requires --resume and an identical configuration')
    atomic_json(run_path,run)
    records=[]
    for variant in variants:
        for split in splits:
            truth_path=setup.assets/assets['splits'][split]['file']
            assert hashlib.sha256(truth_path.read_bytes()).hexdigest()==assets['splits'][split]['sha256']
            saved=torch.load(truth_path,map_location='cpu',weights_only=False)['ground_truth']
            indices=list(range(100)) if args.all_cases else CASES[args.pde][split]
            directory=args.output/variant/split;directory.mkdir(parents=True,exist_ok=True)
            for index in indices:
                existing=directory/f'case_{index:03d}.json'
                if args.resume and existing.exists():
                    record=json.loads(existing.read_text())
                    assert (record['sample_id'],record['variant'],record['split'])==(index,variant,split)
                    records.append(record)
                    continue
                torch.manual_seed(index)
                truth=case_ground_truth(saved,index,'cuda')
                masks,_=common_masks(truth,index,0,task if args.pde!='burger' else 'inverse')
                config=load_config(setup.fm4pde/'configs/main'/task/f'{args.pde}.yaml',{
                    'test_type':split,'data_path':assets['splits'][split]['source_file'],
                    'checkpoint_path':str(setup.checkpoint),'output_dir':str(directory/f'native_{index:03d}'),
                    'batch_size':1,'offset':index,'device':'cuda','model_profile':'auto',
                    'sample_seed':index,'mask_seed':0,'noise_level':0.,'num_steps':100,'save_plots':False})
                overrides={**original['guidance_overrides'],**VARIANTS[variant]}
                factor=overrides.pop('observation_multiplier',1.)
                config.zeta_obs_a*=factor;config.zeta_obs_u*=factor
                for key,value in overrides.items():setattr(config,key,value)
                config.runtime_metadata['diagnostic_sampling_configuration']=dict(run,selected_variant=variant)
                start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                record=dict(variant=variant,pde=args.pde,task=task,split=split,sample_id=index,status='ok',
                    relative_l2_coefficient=None,relative_l2_solution=None,clip_threshold=config.clip_threshold,
                    zeta_obs_a=config.zeta_obs_a,zeta_obs_u=config.zeta_obs_u,zeta_pde=config.zeta_pde)
                try:
                    with native_noise_provider(noise,True),trace_native(directory/f'trace_{index:03d}.jsonl',not args.all_cases):
                        result=run_single_ablation(config,checkpoint_bundle=(net,normalizer,payload),ground_truth=truth,observation_masks=masks)
                    assert result['status']=='ok'
                    artifact=torch.load(Path(result['run_dir'])/'result.pt',map_location='cpu',weights_only=False)
                    prediction=artifact['sol_final'] if channels==1 else torch.cat((artifact['coef_final'],artifact['sol_final']),dim=1)
                    prediction=prediction.numpy().astype(np.float64);target=truth.pair.detach().cpu().numpy().astype(np.float64)
                    assert prediction.shape==target.shape
                    np.save(directory/f'prediction_{index:03d}.npy',prediction)
                    errors=np.linalg.norm((prediction-target).reshape(channels,-1),axis=1)/np.linalg.norm(target.reshape(channels,-1),axis=1)
                    if not np.isfinite(prediction).all() or not np.isfinite(errors).all():raise FloatingPointError('Nonfinite prediction/error')
                    record.update(relative_l2_coefficient=float(errors[0]) if channels==2 else None,
                        relative_l2_solution=float(errors[-1]),native_run_dir=result['run_dir'],prediction_max_abs=float(np.abs(prediction).max()))
                except (FloatingPointError,RuntimeError,AssertionError) as error:
                    if not isinstance(error,FloatingPointError) and not any(word in str(error).lower() for word in ['nan','inf','nonfinite','non-finite']):raise
                    record.update(status='failed',error=str(error))
                torch.cuda.synchronize()
                record.update(seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved())
                atomic_json(directory/f'case_{index:03d}.json',record);records.append(record)
                atomic_json(args.output/'results.json',records)
                print(json.dumps(record,allow_nan=False),flush=True)
    summaries={}
    for variant in variants:
        for split in splits:
            selected=[r for r in records if (r['variant'],r['split'])==(variant,split)]
            vals=[r['relative_l2_solution' if channels==1 else 'relative_l2_coefficient'] for r in selected if r['status']=='ok']
            summaries[variant+'/'+split]=dict(cases=len(selected),failures=len(selected)-len(vals),
                finite_mean_percent=float(np.mean(vals)*100) if vals else None,finite_median_percent=float(np.median(vals)*100) if vals else None,
                finite_max_percent=max(vals)*100 if vals else None,over1000_percent=sum(v>10 for v in vals),over1000_ratio=sum(v>1000 for v in vals))
    atomic_json(args.output/'completed.json',summaries)
    print(json.dumps(summaries,allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pde',choices=list(CASES),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',choices=list(VARIANTS),nargs='+')
    p.add_argument('--splits',choices=['id','smooth','rough'],nargs='+')
    p.add_argument('--all-cases',action='store_true')
    p.add_argument('--resume',action='store_true',help='Retain completed cases when moving an identical run to another GPU.')
    main(p.parse_args())
