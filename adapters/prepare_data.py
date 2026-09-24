"""Data-only bridge: official DDIS normalization and FM4PDE split membership.

Run export beside the authoritative MATLAB files, then transfer the compact NPY
artifacts and run hf on the training server. No model or sampler is defined here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(args):
    sys.path.insert(0, str(args.ddis / "utils"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provenance"))
    from dataset_prop import STATS, load_helmholtz, load_poisson
    from fm4pde_split_reference import _train_val_split_indices

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed export: {manifest_path}")
    source_hashes = {}
    for line in (args.checksums.read_text().splitlines() if args.checksums else []):
        checksum, relative = line.split(maxsplit=1)
        source_hashes[relative.lstrip("*")] = checksum
    loader = {"poisson": load_poisson, "helmholtz": load_helmholtz}[args.pde]
    stats = STATS[args.pde]
    source_records, outputs = [], {}

    def load_verified(relative):
        path = args.raw / relative
        checksum = sha256(path)
        if source_hashes and checksum != source_hashes[relative]:
            raise ValueError(f"Source SHA256 mismatch: {path}")
        a, u = loader(path)
        if a.shape != (10000, 128, 128) or u.shape != a.shape:
            raise ValueError(f"Unexpected source shapes: {a.shape}, {u.shape}")
        source_records.append({"path": str(path), "sha256": checksum})
        return a, u

    def normalize(a, u):
        # Same float64 arithmetic and published constants as the official
        # dataset_process.data_generator; train.py converts its result to fp32.
        a = (a.astype(np.float64) - stats["mean"][0]) * (0.5 / stats["std"][0])
        u = (u.astype(np.float64) - stats["mean"][1]) * (0.5 / stats["std"][1])
        x = np.stack((a, u), axis=1).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite data")
        return x

    def finish(split, ids, temporary):
        final = args.output / f"{split}.npy"
        os.replace(temporary, final)
        np.save(args.output / f"{split}_ids.npy", ids)
        outputs[split] = {"file": final.name, "sha256": sha256(final),
                          "samples": len(ids), "ids": f"{split}_ids.npy"}
        print(f"Completed {split}: {len(ids)} samples", flush=True)

    # Test subsets are small, so make them available before the training export.
    for split in ("id", "smooth", "rough"):
        a, u = load_verified(f"{args.pde}/{args.pde}_test_10000-128-128_{split}.mat")
        temp = args.output / f".{split}.npy"
        np.save(temp, normalize(a[:100], u[:100]))
        finish(split, np.arange(100), temp)
        del a, u

    split_seed = args.seed + {"poisson": 1, "helmholtz": 2}[args.pde] * 1009
    train_ids, val_ids = _train_val_split_indices(50000, split_seed, 0.1)
    ids_by_split = {"train": train_ids.numpy(), "validation": val_ids.numpy()}
    arrays = {split: np.lib.format.open_memmap(args.output / f".{split}.npy", mode="w+",
              dtype=np.float32, shape=(len(ids), 2, 128, 128))
              for split, ids in ids_by_split.items()}
    for shard in range(1, 6):
        a, u = load_verified(f"{args.pde}/{args.pde}_10000-128-128_{shard}.mat")
        lower = (shard - 1) * 10000
        for split, ids in ids_by_split.items():
            positions = np.flatnonzero((ids >= lower) & (ids < lower + 10000))
            for start in range(0, len(positions), 256):
                where = positions[start:start + 256]
                source = ids[where] - lower
                arrays[split][where] = normalize(a[source], u[source])
        del a, u
        print(f"Completed training shard {shard}/5", flush=True)
    for split, ids in ids_by_split.items():
        arrays[split].flush()
        del arrays[split]
        finish(split, ids, args.output / f".{split}.npy")
    manifest = {"name": args.pde, "stats": stats, "shape": [2, 128, 128],
                "__version__": STATS["__version__"], "storage_dtype": "float32",
                "normalization": "DDIS official constants; (x-mean)*0.5/std, computed in float64",
                "seed": args.seed, "split_seed": split_seed,
                "split_reference": "provenance/fm4pde_split_reference.py",
                "sources": source_records, "outputs": outputs}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def hf_records(array_path, ids_path):
    array = np.load(array_path, mmap_mode="r")
    ids = np.load(ids_path)
    for idx in range(len(ids)):
        yield {"id": int(ids[idx]), "data": array[idx]}


def make_hf(args):
    from datasets import Array3D, Dataset, Features, Value
    manifest = json.loads((args.source / "manifest.json").read_text())
    entry = manifest["outputs"][args.split]
    array_path = args.source / entry["file"]
    if sha256(array_path) != entry["sha256"]:
        raise ValueError(f"Transfer checksum mismatch: {array_path}")
    ids = np.load(args.source / entry["ids"])
    if args.split in ("train", "validation"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "provenance"))
        from fm4pde_split_reference import _train_val_split_indices
        train_ids, validation_ids = _train_val_split_indices(50000, manifest["split_seed"], 0.1)
        expected = train_ids.numpy() if args.split == "train" else validation_ids.numpy()
    else:
        expected = np.arange(100)
    if not np.array_equal(ids, expected):
        raise ValueError("Sample membership/order does not match the reference protocol")
    if args.output.exists():
        raise FileExistsError(args.output)
    features = Features({"id": Value("int32"), "data": Array3D(tuple(manifest["shape"]), "float32")})
    dataset = Dataset.from_generator(hf_records, features=features,
        gen_kwargs={"array_path": str(array_path), "ids_path": str(args.source / entry["ids"])},
        cache_dir=str(args.cache))
    assert len(dataset) == entry["samples"]
    dataset.save_to_disk(str(args.output), max_shard_size="500MB")
    metadata = {key: manifest[key] for key in ("name", "stats", "shape", "__version__")}
    metadata.update(num_samples=len(dataset), split=args.split, source_manifest=str(args.source / "manifest.json"))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def diffusionpde(args):
    """Write the per-image HWC format expected by official ImageFolderDataset."""
    manifest = json.loads((args.source / "manifest.json").read_text())
    entry = manifest["outputs"][args.split]
    path = args.source / entry["file"]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Transfer checksum mismatch: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    array = np.load(path, mmap_mode="r")
    stats = manifest["stats"]
    mean = np.asarray(stats["mean"], dtype=np.float64)[:, None, None]
    scale = np.asarray(stats["std"], dtype=np.float64)[:, None, None] / 0.5
    # Invert exactly the published generate_poisson / generate_helmholtz scales.
    sol_scale = 1 / 36.5 if manifest["name"] == "poisson" else 0.028
    physical_scale = np.array([2.15, sol_scale])[:, None, None]
    for index, x in enumerate(array):
        physical = np.asarray(x, dtype=np.float64) * scale + mean
        normalized = (physical / physical_scale).transpose(1, 2, 0).astype(np.float32)
        np.save(args.output / f"sample_{index:05d}.npy", normalized)
    (args.output / "metadata.json").write_text(json.dumps({
        "name": manifest["name"], "split": args.split, "num_samples": len(array),
        "source_manifest": str(args.source / "manifest.json"),
        "physical_scales": [2.15, sol_scale],
        "note": "Physical fields recovered from the float32 compact export; float32 roundoff applies."
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--pde", choices=["poisson", "helmholtz"], required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--ddis", type=Path, required=True)
    p.add_argument("--checksums", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=export)
    p = sub.add_parser("hf")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--split", choices=["train", "validation", "id", "smooth", "rough", "rough2", "rough3"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.set_defaults(func=make_hf)
    p = sub.add_parser("diffusionpde")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--split", choices=["train", "id"], default="train")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=diffusionpde)
    args = parser.parse_args()
    args.func(args)
