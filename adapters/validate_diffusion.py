"""Evaluate a saved official EMA and its saved official loss on validation data."""
import argparse
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main(args):
    sys.path.insert(0, str(args.repo))
    from training.dataset_hf import PDEDataset

    metadata = json.loads((args.validation / "metadata.json").read_text())
    if not args.profile and (metadata.get("split") != "validation" or metadata["num_samples"] != 5000):
        raise ValueError("Early stopping requires the separate 5,000-case validation split")
    start = time.monotonic()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    with args.checkpoint.open("rb") as stream:
        snapshot = pickle.load(stream)
    model = snapshot["ema"].to("cuda").eval().requires_grad_(False)
    loss_fn = snapshot["loss_fn"]
    # The official snapshot serializes its RBF sampler, including its GPU
    # tensor. Reuse that exact loss/sampler without reconstructing either.
    data = PDEDataset(path=str(args.validation), resolution=model.img_resolution,
                      max_size=args.limit, shuffle=False, use_labels=False)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=False, num_workers=1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for fields, _ in loader:
            fields = fields.to("cuda", dtype=torch.float32)
            if args.method == "ddis":
                fields = fields[:, [0]]
            loss = loss_fn(net=model, images=fields, labels=None, augment_pipe=None)
            value = loss.mean().item()
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation loss")
            total += value * fields.shape[0]
            count += fields.shape[0]
    torch.cuda.synchronize()
    record = {"checkpoint": str(args.checkpoint), "validation": str(args.validation),
              "samples": count, "batch": args.batch, "seed": args.seed, "loss": total / count,
              "metric": "official saved EMA diffusion loss; fixed noise seed; no augmentation",
              "seconds": time.monotonic() - start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--method", choices=["ddis", "fundps"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and not args.profile:
        parser.error("Subset limits are only allowed for resource profiles")
    main(args)
