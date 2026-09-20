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

from evaluate_flow import pair_observation_indices


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
    if args.case_index is not None and args.count != 1:
        raise ValueError('A selected case requires count=1')
    selected = args.case_index is not None
    if selected and not 0 <= args.case_index < 100:
        raise ValueError('Case index must be within the first 100')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'run.json').exists():
        raise FileExistsError(f'Existing evaluation run: {args.output}')
    config.update(outdir=str(args.output), pkl_path=str(args.checkpoint.resolve()),
                  max_size=args.count, wandb=False)
    if args.data_path is not None:
        config['data_path'] = str(args.data_path)
    if selected:
        config['data_offset'] = args.case_index
    task = args.task or 'inverse'
    if args.task is not None:
        config['task'] = 'forward_inverse' if task == 'both' else task
    if args.method == "ddis":
        if args.surrogate is None:
            raise ValueError("DDIS requires the validation-selected surrogate")
        config["surrogate_path"] = str(args.surrogate.resolve())
        assert config["guidance"]["type"] == "daps_multires"
        if args.task is not None:
            for key in ('weights', 'weights_1', 'weights_2'):
                pair = config['guidance']['langevin'][key]
                active = pair[1]
                config['guidance']['langevin'][key] = [active if task in ('forward', 'both') else 0,
                                                     active if task in ('inverse', 'both') else 0]
    else:
        assert config["guidance"]["type"] == "dps"
        if args.task is not None:
            weights = config['guidance']['weights']
            active = weights[1]
            config['guidance']['weights'] = [active if task in ('forward', 'both') else 0,
                                             active if task in ('inverse', 'both') else 0,
                                             weights[2]]
    metadata = json.loads((args.source / "manifest.json").read_text())
    assert metadata["name"].replace("-", "") == config["dataset"].replace("-", "")
    ids = np.load(args.source / f"{args.split}_ids.npy")
    assert np.array_equal(ids, np.arange(100))
    truth = np.load(args.source / f"{args.split}.npy", mmap_mode="r")
    coefficient_masks, solution_masks = pair_observation_indices(config["seed"])
    np.save(args.output / "coefficient_observation_indices.npy", coefficient_masks)
    np.save(args.output / "solution_observation_indices.npy", solution_masks)
    record = {"method": args.method, "split": args.split, "count": args.count,
              "task": task, "case_index": args.case_index,
              "source": str(args.source), "official_config": config,
              "checkpoint_sha256": file_hash(args.checkpoint), "profile_only": args.profile}
    if metadata['name'] == 'darcy':
        if args.fm4pde is None:
            raise ValueError('FM4PDE Darcy data requires its 4/12 coefficient postprocessing')
        record['physical_postprocessing'] = 'FM4PDE official Darcy binary 4/12 mapping; upstream DDIS/FunDPS defaults target a different 3/12 dataset'
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
    torch.manual_seed(config["seed"] + (args.case_index or 0))
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    solver = Solver(Config(config))
    if metadata['name'] == 'darcy':
        sys.path.insert(0, str(args.fm4pde))
        from sampling.pde_residuals import _apply_darcy_coef_positive
        original_load = solver.load_data
        def load_aligned_data():
            original_load()
            def transform(x):
                return torch.cat((_apply_darcy_coef_positive(x[:, :1],
                    {'coef_positive_mode': 'binary'}), x[:, 1:]), dim=1)
            solver.normalizer._transform = transform
        solver.load_data = load_aligned_data
    if selected:
        for _ in range(args.case_index):
            np.random.choice(128 * 128, 500, replace=False)
            np.random.choice(128 * 128, 500, replace=False)
    solver.generate()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    last_index = args.case_index if selected else args.count - 1
    for channel, expected_masks in enumerate((coefficient_masks, solution_masks)):
        actual_mask = solver.observations[0].masks[channel].flatten().nonzero().flatten().cpu().numpy()
        assert np.array_equal(np.sort(actual_mask), np.sort(expected_masks[last_index]))
    means = np.asarray(metadata["stats"]["mean"])[None, :, None, None]
    scales = np.asarray(metadata["stats"]["std"])[None, :, None, None] / 0.5
    failed = 0
    for index in range(args.count):
        prediction_path = args.output / f"results/batch_{index}.npy"
        prediction = np.load(prediction_path)
        assert prediction.shape == (1, 2, 128, 128)
        source_index = args.case_index if selected else index
        target = np.asarray(truth[source_index:source_index + 1], dtype=np.float64) * scales + means
        finite = bool(np.isfinite(prediction).all())
        errors = np.linalg.norm((prediction - target).reshape(2, -1), axis=1) / np.linalg.norm(target.reshape(2, -1), axis=1)
        finite = finite and bool(np.isfinite(errors).all())
        case = {"sample_id": int(ids[source_index]), "status": "ok" if finite else "failed",
                "prediction": str(prediction_path), "prediction_units": "physical",
                "relative_l2_coefficient": float(errors[0]) if finite else None,
                "relative_l2_solution": float(errors[1]) if finite else None}
        (args.output / f"case_{source_index:03d}.json").write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
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
    parser.add_argument("--fm4pde", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=["id", "smooth", "rough", "rough2", "rough3"], required=True)
    parser.add_argument("--task", choices=["forward", "both", "inverse"])
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=[1, 100], default=100)
    parser.add_argument("--profile", action="store_true")
    main(parser.parse_args())
