# Unmodified split function from FM4PDE train.py at ec9e658f486dbf904e44ee7d0354a8beacea9ec2
import warnings
import torch

def _train_val_split_indices(
    num_samples: int,
    seed: int,
    val_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_samples < 1:
        raise ValueError("Cannot split an empty dataset")
    if num_samples == 1:
        warnings.warn(
            "Only one sample was loaded; reusing it for validation because a distinct 9:1 split is impossible.",
            RuntimeWarning,
            stacklevel=2,
        )
        index = torch.tensor([0], dtype=torch.long)
        return index, index
    val_count = max(1, int(round(float(num_samples) * float(val_ratio))))
    val_count = min(val_count, num_samples - 1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(num_samples, generator=generator)
    val_idx = permutation[:val_count].sort().values
    train_idx = permutation[val_count:].sort().values
    return train_idx, val_idx
