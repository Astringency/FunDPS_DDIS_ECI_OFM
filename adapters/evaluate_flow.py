"""Batch/observation adapter around official ECI and OFM sampling routines."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from train_flow import build_prior


def observation_indices(seed, count=100):
    # Match official DDIS/FunDPS SparseObservation with [500, 500], batch=1:
    # the inactive coefficient channel still consumes the first RNG draw.
    rng = np.random.RandomState(seed)
    indices = []
    for _ in range(count):
        rng.choice(128 * 128, 500, replace=False)
        indices.append(rng.choice(128 * 128, 500, replace=False))
    return np.stack(indices)


class ECIPriorShapeAdapter:
    """Expose the official OFM independent-channel GP to ECI's spatial API."""
    def __init__(self, gp):
        self.gp = gp

    def sample(self, grid, dims, n_samples=1):
        assert list(dims) == [2, 128, 128]
        return self.gp.sample(list(dims[1:]), n_samples=n_samples, n_channels=dims[0])


def ofm_sample(prior, truth, mask, args):
    from sampling_FSGLD.samplers import LangevinDynamics
    latent = prior.gp.sample([128, 128], n_samples=1, n_channels=2).detach().requires_grad_(True)
    observed = truth[0][mask[0]]

    def negative_log_posterior(x):
        # Same objective as OFM_GRF_3C_Regression.ipynb, cell 25. Only the
        # codomain count and observation selection change for this dataset.
        current, logp, _ = prior.data_likelihood_precise_codomain(
            x, n_channels=2, n_eval=4, n_repeat=args.hutchinson,
            forward=True, method="dopri5")
        loss1 = -0.5 * torch.sum((observed - current[mask[0]]) ** 2) / args.noise_variance
        loss = -(loss1 + logp)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite official OFM posterior objective")
        return loss

    # The upstream notebook's MAP stage is optional. Here the declared budget
    # is Langevin sampling from the GP, with no additional MAP optimization.
    sampler = LangevinDynamics(latent, negative_log_posterior, lr=1e-3,
        lr_final=8e-4, max_itr=args.langevin_steps, device=args.device,
        temperature=1, momentum=0)
    for step in range(args.langevin_steps):
        _, loss = sampler.sample(epoch=step)
        if step % 10 == 0:
            print(f"OFM Langevin {step}/{args.langevin_steps}, loss={loss}", flush=True)
    return prior.inv_sample(latent.detach(), n_eval=4, forward=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config_path = args.output / "run.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory belongs to a different evaluation config")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    manifest = json.loads((args.source / "manifest.json").read_text())
    data = np.load(args.source / f"{args.split}.npy", mmap_mode="r")
    ids = np.load(args.source / f"{args.split}_ids.npy")
    assert data.shape == (100, 2, 128, 128) and np.array_equal(ids, np.arange(100))
    masks = observation_indices(args.seed)
    np.save(args.output / "solution_observation_indices.npy", masks)
    torch.manual_seed(args.seed)
    model, prior = build_prior(args.ofm, 128, args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.eval().requires_grad_(False)
    if args.method == "eci":
        sys.path.insert(0, str(args.eci))
        from models.functional import FFM
        from models.constraints import DirichletCondition
        eci = SimpleNamespace(model=model, gp=ECIPriorShapeAdapter(prior.gp))
    means = np.array(manifest["stats"]["mean"], dtype=np.float64)[None, :, None, None]
    scales = np.array(manifest["stats"]["std"], dtype=np.float64)[None, :, None, None] / 0.5
    for index in range(args.offset, min(100, args.offset + args.count)):
        result_path = args.output / f"case_{index:03d}.json"
        if result_path.exists():
            continue
        torch.manual_seed(args.seed + index)
        truth = torch.tensor(np.array(data[index:index + 1]), device=args.device)
        mask = torch.zeros_like(truth, dtype=torch.bool)
        mask[0, 1].view(-1)[torch.as_tensor(masks[index], device=args.device)] = True
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        record = {"sample_id": int(ids[index]), "status": "ok"}
        try:
            if args.method == "eci":
                prediction = FFM.eci_sample(eci, 1, args.eci_steps, args.eci_mix, None,
                    [2, 128, 128], args.device, DirichletCondition(value=truth, mask=mask))
            else:
                prediction = ofm_sample(prior, truth, mask, args)
            torch.cuda.synchronize()
            prediction = prediction.detach().cpu().numpy().astype(np.float64) * scales + means
            target = np.asarray(data[index:index + 1], dtype=np.float64) * scales + means
            np.save(args.output / f"prediction_{index:03d}.npy", prediction)
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite prediction")
            errors = np.linalg.norm((prediction - target).reshape(2, -1), axis=1) / np.linalg.norm(target.reshape(2, -1), axis=1)
            record.update(relative_l2_coefficient=float(errors[0]), relative_l2_solution=float(errors[1]))
        except (FloatingPointError, RuntimeError, AssertionError) as error:
            numerical_failure = isinstance(error, FloatingPointError) or any(
                text in str(error).lower() for text in ("underflow in dt", "non-finite", "nonfinite", "nan", "infinite"))
            if not numerical_failure and not isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            record.update(status="failed", error=str(error), relative_l2_coefficient=None, relative_l2_solution=None)
            if isinstance(error, torch.cuda.OutOfMemoryError):
                result_path.write_text(json.dumps(record, indent=2) + "\n")
                raise
        record.update(seconds=time.monotonic() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["eci", "ofm"], required=True)
    parser.add_argument("--ofm", type=Path, required=True)
    parser.add_argument("--eci", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=["id", "smooth", "rough"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eci-steps", type=int, default=800)
    parser.add_argument("--eci-mix", type=int, default=5)
    parser.add_argument("--langevin-steps", type=int, default=100)
    parser.add_argument("--hutchinson", type=int, default=1)
    parser.add_argument("--noise-variance", type=float, default=1e-3)
    main(parser.parse_args())
