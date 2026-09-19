"""Call the unmodified official PDE sampler once per common evaluation case."""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy.io
import torch
import yaml

from evaluate_flow import observation_indices


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.source / "manifest.json").read_text())
    pde = metadata["name"]
    data = np.load(args.source / f"{args.split}.npy", mmap_mode="r")
    ids = np.load(args.source / f"{args.split}_ids.npy")
    assert data.shape == (100, 2, 128, 128) and np.array_equal(ids, np.arange(100))
    means = np.array(metadata["stats"]["mean"], dtype=np.float64)[None, :, None, None]
    scales = np.array(metadata["stats"]["std"], dtype=np.float64)[None, :, None, None] / 0.5
    physical = np.asarray(data, dtype=np.float64) * scales + means
    physical_path = args.output / "test_physical.mat"
    scipy.io.savemat(physical_path, {"f_data": physical[:, 0],
        "phi_data" if pde == "poisson" else "psi_data": physical[:, 1]})
    masks = observation_indices(args.seed)
    np.save(args.output / "solution_observation_indices.npy", masks)
    config = yaml.safe_load(args.config.read_text())
    assert config["generate"]["zeta_obs_a"] == 0 and config["generate"]["batch_size"] == 1
    config["test"]["pre-trained"] = str(args.checkpoint)
    config["test"]["iterations"] = args.steps
    config["data"]["datapath"] = str(physical_path)
    config["generate"]["device"] = args.device
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config["official_config"] = config
    run_path = args.output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run_config:
        raise ValueError("Output belongs to a different evaluation config")
    run_path.write_text(json.dumps(run_config, indent=2) + "\n")
    sys.path.insert(0, str(args.repo))
    module = importlib.import_module(f"scripts.generate_{pde}")
    official_index = module.random_index
    initial_directory = Path.cwd()
    for index in range(args.offset, min(100, args.offset + args.count)):
        result_path = args.output / f"case_{index:03d}.json"
        if result_path.exists():
            continue
        def supplied_observations(k, grid_size, seed=0, device=torch.device("cuda")):
            if seed != 0:
                return official_index(k, grid_size, seed, device)
            assert k == 500 and grid_size == 128
            mask = torch.zeros((128, 128), device=device)
            mask.view(-1)[torch.as_tensor(masks[index], device=device)] = 1
            return mask
        module.random_index = supplied_observations
        current_config = copy.deepcopy(config)
        current_config["data"]["offset"] = index
        current_config["generate"]["seed"] = args.seed + index
        case_dir = args.output / f"official_case_{index:03d}"
        case_dir.mkdir(exist_ok=True)
        (case_dir / "config.json").write_text(json.dumps(current_config, indent=2) + "\n")
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            os.chdir(case_dir)
            getattr(module, f"generate_{pde}")(current_config)
        finally:
            os.chdir(initial_directory)
        torch.cuda.synchronize()
        result = scipy.io.loadmat(case_dir / f"{pde}_results.mat")
        prediction = np.stack((result["a"].squeeze(), result["u"].squeeze()))[None]
        assert prediction.shape == (1, 2, 128, 128)
        np.save(args.output / f"prediction_{index:03d}.npy", prediction)
        finite = bool(np.isfinite(prediction).all())
        errors = np.linalg.norm((prediction[0] - physical[index]).reshape(2, -1), axis=1) / np.linalg.norm(physical[index].reshape(2, -1), axis=1)
        record = {"sample_id": int(ids[index]), "status": "ok" if finite else "failed",
                  "relative_l2_coefficient": float(errors[0]) if finite else None,
                  "relative_l2_solution": float(errors[1]) if finite else None,
                  "seconds": time.monotonic() - start,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        result_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=["id", "smooth", "rough"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
