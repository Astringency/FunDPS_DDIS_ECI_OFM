"""Data/config entry point calling unchanged official DDIS or FunDPS generation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from evaluate_flow import observation_indices


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    config = yaml.safe_load(args.config.read_text())
    assert config["batch_size"] == 1 and config["resolution"] == 128
    assert config["data_offset"] == 0 and config["seed"] == 0
    args.output.mkdir(parents=True, exist_ok=False)
    config.update(outdir=str(args.output), pkl_path=str(args.checkpoint.resolve()),
                  max_size=args.count, wandb=False)
    if args.method == "ddis":
        if args.surrogate is None:
            raise ValueError("DDIS requires the validation-selected surrogate")
        config["surrogate_path"] = str(args.surrogate.resolve())
        assert config["guidance"]["type"] == "daps_multires"
    else:
        assert config["guidance"]["type"] == "dps"
    metadata = json.loads((args.source / "manifest.json").read_text())
    assert metadata["name"] == config["dataset"]
    ids = np.load(args.source / f"{args.split}_ids.npy")
    assert np.array_equal(ids, np.arange(100))
    truth = np.load(args.source / f"{args.split}.npy", mmap_mode="r")
    masks = observation_indices(config["seed"])
    np.save(args.output / "solution_observation_indices.npy", masks)
    record = {"method": args.method, "split": args.split, "count": args.count,
              "source": str(args.source), "official_config": config,
              "checkpoint_sha256": file_hash(args.checkpoint), "profile_only": args.profile}
    if args.surrogate is not None:
        record["surrogate_sha256"] = file_hash(args.surrogate)
    (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    sys.path.insert(0, str(args.repo))
    from utils.yaml_config import Config
    if args.method == "ddis":
        # The upstream generic CLI imports optional modules absent from its
        # checkout. Import the required official solver directly instead.
        from generation.daps_multires import PDESolverDAPS_MultiRes as Solver
    else:
        from generation.dps import PDESolverDPS as Solver
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    solver = Solver(Config(config))
    solver.generate()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    actual_last_mask = solver.observations[0].masks[1].flatten().nonzero().flatten().cpu().numpy()
    assert np.array_equal(np.sort(actual_last_mask), np.sort(masks[args.count - 1]))
    means = np.asarray(metadata["stats"]["mean"])[None, :, None, None]
    scales = np.asarray(metadata["stats"]["std"])[None, :, None, None] / 0.5
    failed = 0
    for index in range(args.count):
        prediction_path = args.output / f"results/batch_{index}.npy"
        prediction = np.load(prediction_path)
        assert prediction.shape == (1, 2, 128, 128)
        target = np.asarray(truth[index:index + 1], dtype=np.float64) * scales + means
        finite = bool(np.isfinite(prediction).all())
        errors = np.linalg.norm((prediction - target).reshape(2, -1), axis=1) / np.linalg.norm(target.reshape(2, -1), axis=1)
        finite = finite and bool(np.isfinite(errors).all())
        case = {"sample_id": int(ids[index]), "status": "ok" if finite else "failed",
                "prediction": str(prediction_path), "prediction_units": "physical",
                "relative_l2_coefficient": float(errors[0]) if finite else None,
                "relative_l2_solution": float(errors[1]) if finite else None}
        (args.output / f"case_{index:03d}.json").write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
        failed += not finite
    summary = {"evaluated": args.count, "failed": failed, "profile_only": args.profile,
               "seconds": elapsed, "seconds_per_case": elapsed / args.count,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    (args.output / "completed.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ddis", "fundps"], required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=["id", "smooth", "rough"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=[1, 100], default=100)
    parser.add_argument("--profile", action="store_true")
    main(parser.parse_args())
