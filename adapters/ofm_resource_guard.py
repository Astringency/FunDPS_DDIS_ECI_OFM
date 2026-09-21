"""Runtime-only memory guard for OFM's sample-dependent autograd graph."""
import fcntl
import os
from pathlib import Path

import torch


def acquire_native_slot(root):
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    lock = (Path(root) / 'locks' / f'native_ofm_memory_gpu_{gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def install_guard(prior):
    def check(module, inputs):
        free, total = torch.cuda.mem_get_info()
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if free + reusable < 8 * 1024 ** 3:
            raise RuntimeError('OFM_RESOURCE_LIMIT: less than 8 GiB usable headroom; retry on a less occupied GPU')
    return prior.model.register_forward_pre_hook(check)
