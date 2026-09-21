"""Optional PyTorch activation recomputation; official FNO and sampler stay intact."""
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_forward(model):
    original = model.forward

    def forward(*args, **kwargs):
        return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)

    model.forward = forward


def configure(prior, args):
    policy = args.assets.parent.parent / 'jobs/ofm_memory_policy.json'
    enabled = policy.exists() and json.loads(policy.read_text()).get('activation_checkpoint', False)
    if enabled:
        checkpoint_forward(prior.model)
    record = dict(activation_checkpoint=bool(enabled),
        implementation='torch.utils.checkpoint.checkpoint(use_reentrant=False, preserve_rng_state=True)',
        changes_sampler_parameters=False, policy_source=str(policy) if policy.exists() else None)
    path = args.output / 'memory_runtime.json'
    if path.exists() and json.loads(path.read_text()) != record:
        history = args.output / 'memory_runtime_history'
        history.mkdir(exist_ok=True)
        (history / f'previous_{len(list(history.iterdir()))}.json').write_bytes(path.read_bytes())
    path.write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    import argparse
    import sys
    from train_flow import official_ofm
    parser = argparse.ArgumentParser()
    parser.add_argument('--ofm', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    FNO, _ = official_ofm(args.ofm)
    model = FNO(modes=32, vis_channels=2, hidden_channels=128, proj_channels=128, x_dim=2).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cuda', weights_only=True))
    model.eval().requires_grad_(False)
    torch.manual_seed(6)
    source = torch.randn(1, 2, 128, 128, device='cuda')
    noise = torch.randn_like(source)
    outputs, gradients, peaks = [], [], []
    for recompute in (False, True):
        if recompute:
            checkpoint_forward(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = source.detach().clone().requires_grad_(True)
        out = model(torch.tensor([.4], device='cuda'), x)
        # Match the official likelihood's inner autograd.grad + outer backward.
        inner = torch.autograd.grad((out * noise).sum(), x, retain_graph=True)[0]
        loss = out.square().sum() + inner.square().sum()
        loss.backward()
        outputs.append(out.detach().cpu())
        gradients.append(x.grad.detach().cpu())
        peaks.append(torch.cuda.max_memory_allocated())
        del x, out, inner, loss
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    result = dict(torch=torch.__version__, forward_bitwise_equal=True,
        gradient_bitwise_equal=True, peak_allocated_bytes=peaks,
        official_revision=__import__('subprocess').check_output(
            ['git', '-C', str(args.ofm), 'rev-parse', 'HEAD'], text=True).strip())
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
