"""Validate native physics metadata and alignment before model sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    sys.path.insert(0,str(args.fm4pde))
    from sampling.config import load_config
    from sampling.losses import compute_guidance_losses
    from sampling.runner import _disable_unreliable_pde_guidance
    from sampling.state import split_pair_state
    from shared_prior_runtime import case_ground_truth,common_masks

    manifest=json.loads((args.assets/'manifest.json').read_text())
    rows=[]
    for pde,entry in manifest['pdes'].items():
        compact_manifest=json.loads((args.compact/pde/'manifest.json').read_text())
        for split,record in entry['splits'].items():
            truth_path=args.assets/record['file']
            assert sha256(truth_path)==record['sha256']
            saved=torch.load(truth_path,map_location='cpu',weights_only=False)['ground_truth']
            compact_path=args.compact/pde/f'{split}.npy'
            assert sha256(compact_path)==compact_manifest['outputs'][split]['sha256']
            compact=np.load(compact_path,mmap_mode='r')
            physical=saved['pair'].numpy().astype(np.float64)
            assert compact.shape==physical.shape
            mean=np.array(compact_manifest['stats']['mean'])[None,:,None,None]
            scale=np.array(compact_manifest['stats']['std'])[None,:,None,None]/0.5
            error=np.max(np.abs(np.asarray(compact,dtype=np.float64)*scale+mean-physical),axis=(0,2,3))
            tolerance=32*np.finfo(np.float32).eps*np.maximum(np.max(np.abs(physical),axis=(0,2,3)),1e-12)
            assert np.all(error<=tolerance), f'{pde}/{split}: {error}, {tolerance}'
            truth=case_ground_truth(saved,0,'cpu')
            config=load_config(args.fm4pde/'configs/main'/entry['task']/f'{pde}.yaml',
                               {'device':'cpu','batch_size':1})
            _disable_unreliable_pde_guidance(config)
            assert config.guidance_components=='obs_pde' and config.zeta_obs_a==0
            masks,_=common_masks(truth,0)
            state=truth.pair.clone().requires_grad_(True)
            losses=compute_guidance_losses(split_pair_state(state,pde),truth,masks,config)
            assert losses.pde_residual_status in ('reliable','approximate')
            assert torch.isfinite(losses.L_pde).all() and float(losses.L_obs_u.detach())==0
            gradient,=torch.autograd.grad(losses.L_pde,state)
            assert torch.isfinite(gradient).all()
            row={'pde':pde,'split':split,'physical_roundoff_max_abs':error.tolist(),
                 'residual_status':losses.pde_residual_status,'truth_pde_loss':float(losses.L_pde.detach()),
                 'gradient_rms':float(gradient.square().mean().sqrt()),
                 'active_observations':int(masks.sol.sum()),'coefficient_observations':int(masks.coef.sum())}
            rows.append(row)
            print(json.dumps(row),flush=True)
    assert len(rows)==15
    args.output.write_text(json.dumps({'verified_splits':15,'checks':rows},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fm4pde','assets','compact','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
