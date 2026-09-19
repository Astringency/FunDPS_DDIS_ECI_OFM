"""CLI/data adapter for the unchanged OFMModel.train implementation.

DDIS Appendix H.3 uses this shared joint prior for OFM regression and ECI.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NpyFields(Dataset):
    def __init__(self, path, resolution=128, limit=None):
        self.array = np.load(path, mmap_mode="r")
        assert self.array.ndim == 4 and self.array.shape[1:] == (2, 128, 128)
        assert 128 % resolution == 0
        self.stride = 128 // resolution
        self.count = len(self.array) if limit is None else min(limit, len(self.array))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.array[index, :, ::self.stride, ::self.stride], copy=True))


def official_ofm(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    # Upstream true_gaussian_process imports the same util directory under
    # the name ofm_utils. Supply its package path without editing source.
    alias = types.ModuleType("ofm_utils")
    alias.__path__ = [str(repo / "util")]
    sys.modules["ofm_utils"] = alias
    spec = importlib.util.spec_from_file_location("official_ofm_fno", repo / "models/fno.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from ofm_OT_likelihood import OFMModel
    return module.FNO, OFMModel


def build_prior(repo, resolution, device):
    FNO, OFMModel = official_ofm(repo)
    model = FNO(modes=32, vis_channels=2, hidden_channels=128,
                proj_channels=128, x_dim=2).to(device)
    prior = OFMModel(model, kernel_length=0.01, kernel_variance=1.0,
                     nu=0.5, sigma_min=1e-4, device=device,
                     dtype=torch.float32, dims=[resolution, resolution])
    return model, prior


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = NpyFields(args.train, args.resolution, args.limit)
    if not args.profile and len(train) != 45000:
        raise ValueError("Formal training must use the aligned 45,000-case split")
    if args.limit is not None and not args.profile:
        raise ValueError("Subset limits are only allowed for resource profiling")
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(training_implementation="official OFMModel.train",
                    consumers=["OFM regression", "ECI sampling"], samples=len(train))
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    validation = None
    if args.validation is not None:
        validation = DataLoader(NpyFields(args.validation, args.resolution), batch_size=args.batch,
                                shuffle=False, num_workers=2, pin_memory=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model, prior = build_prior(args.ofm, args.resolution, args.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    prior.train(loader, optimizer, args.epochs, scheduler=scheduler,
                test_loader=validation, eval_int=10 if validation is not None else 0,
                save_int=1 if args.profile else 10, generate=False,
                save_path=args.output, saved_model=True)
    result = {"completed_epochs": args.epochs, "profile_only": args.profile,
              "elapsed_seconds": time.monotonic() - start}
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (args.output / "completed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ofm", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
