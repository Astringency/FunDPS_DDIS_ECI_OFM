"""Runtime-only memory guard for OFM's sample-dependent autograd graph."""
import fcntl
import os
from pathlib import Path

import torch


def acquire_native_slot(root):
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    # Slot zero preserves the original mutex for already-running workers.
    # Additional slots are enabled only for explicitly launched, profiled jobs.
    slot = int(os.environ.get('DDIS_OFM_MEMORY_SLOT', '0'))
    if slot not in (0, 1):
        raise ValueError('At most two independently profiled OFM slots per GPU')
    suffix = '' if slot == 0 else f'_slot_{slot}'
    lock = (Path(root) / 'locks' / f'native_ofm_memory_gpu_{gpu}{suffix}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def install_guard(prior):
    def check(module, inputs):
        maximum = float(os.environ.get('DDIS_OFM_MAX_ALLOCATED_GIB', '0'))
        if maximum and torch.cuda.memory_allocated() > maximum * 1024 ** 3:
            raise RuntimeError('OFM_RESOURCE_LIMIT: profiled concurrent-slot memory budget exceeded; retry in a single slot')
        free, total = torch.cuda.mem_get_info()
        reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if free + reusable < 8 * 1024 ** 3:
            raise RuntimeError('OFM_RESOURCE_LIMIT: less than 8 GiB usable headroom; retry on a less occupied GPU')
    return prior.model.register_forward_pre_hook(check)
