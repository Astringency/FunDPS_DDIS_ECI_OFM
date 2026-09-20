"""Read-only instrumentation around native losses and guidance; no tensor edits."""
from contextlib import contextmanager
import json
import math
import torch

@contextmanager
def trace_native(path, enabled):
    if not enabled:
        yield
        return
    import sampling.runner as runner
    loss_fn, grad_fn = runner.compute_guidance_losses, runner.compute_guidance_gradient
    def scalar(x):
        y = float(x)
        return y if math.isfinite(y) else str(y)
    def stats(x):
        if x is None:
            return None
        x = x.detach()
        return {'finite': bool(torch.isfinite(x).all()), 'max_abs': scalar(x.abs().max())}
    with path.open('w') as f:
        def emit(d):
            f.write(json.dumps(d, allow_nan=False) + '\n')
            f.flush()
        def losses(phys, *args, **kwargs):
            emit({'event': 'loss_input', 'coef': stats(phys.coef), 'sol': stats(phys.sol)})
            return loss_fn(phys, *args, **kwargs)
        def gradients(*args, **kwargs):
            result = grad_fn(*args, **kwargs)
            emit({'event': 'gradient', **{n: scalar(getattr(result, n)) for n in
                 ('grad_norm_obs_a', 'grad_norm_obs_u', 'grad_norm_pde', 'grad_norm_total', 'clip_scale')}})
            return result
        runner.compute_guidance_losses, runner.compute_guidance_gradient = losses, gradients
        try:
            yield
        finally:
            runner.compute_guidance_losses, runner.compute_guidance_gradient = loss_fn, grad_fn
