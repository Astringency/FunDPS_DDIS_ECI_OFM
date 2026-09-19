"""Cross-check saved predictions, masks, IDs and physical-unit metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(output, source, split, count):
    run = json.loads((output / "run.json").read_text())
    if run["count"] != count or run["split"] != split:
        raise ValueError("Run metadata does not cover the requested evaluation")
    manifest = json.loads((source / "manifest.json").read_text())
    data_path = source / f"{split}.npy"
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["outputs"][split]["sha256"]:
        raise ValueError("Evaluation source differs from the verified export")
    data = np.load(data_path, mmap_mode="r")
    ids = np.load(source / f"{split}_ids.npy")
    assert np.array_equal(ids, np.arange(100))
    masks = np.load(output / "solution_observation_indices.npy")
    rng = np.random.RandomState(0)
    for index in range(100):
        rng.choice(128 * 128, 500, replace=False)
        assert np.array_equal(masks[index], rng.choice(128 * 128, 500, replace=False))
    assert len(list(output.glob("case_*.json"))) == count
    means = np.asarray(manifest["stats"]["mean"])[None, :, None, None]
    scales = np.asarray(manifest["stats"]["std"])[None, :, None, None] / 0.5
    errors, failed = [], []
    for index in range(count):
        record = json.loads((output / f"case_{index:03d}.json").read_text())
        assert record["sample_id"] == int(ids[index])
        if record["status"] == "failed":
            assert record["relative_l2_coefficient"] is None and record["relative_l2_solution"] is None
            failed.append(index)
            continue
        assert record["status"] == "ok"
        prediction_path = Path(record.get("prediction", output / f"prediction_{index:03d}.npy"))
        prediction = np.load(prediction_path)
        assert prediction.shape == (1, 2, 128, 128) and np.isfinite(prediction).all()
        truth = np.asarray(data[index:index + 1], dtype=np.float64) * scales + means
        error = np.linalg.norm((prediction - truth).reshape(2, -1), axis=1) / np.linalg.norm(truth.reshape(2, -1), axis=1)
        reported = [record["relative_l2_coefficient"], record["relative_l2_solution"]]
        np.testing.assert_allclose(error, reported, rtol=1e-7, atol=1e-9)
        errors.append(error)
    values = np.asarray(errors)
    summary = {"verified_cases": count, "successful_cases": len(errors), "failed_cases": failed,
               "failure_rate": len(failed) / count, "relative_error_units": "ratio; uncapped",
               "source": str(source), "split": split,
               "mean_relative_l2_all_cases": values.mean(0).tolist() if not failed else None,
               "mean_relative_l2_successful_cases": values.mean(0).tolist() if errors else None,
               "median_relative_l2_successful_cases": np.median(values, axis=0).tolist() if errors else None,
               "channel_order": ["coefficient", "solution"]}
    (output / "verified_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=["id", "smooth", "rough"], required=True)
    parser.add_argument("--count", type=int, choices=[1, 100], default=100)
    args = parser.parse_args()
    verify(args.output, args.source, args.split, args.count)
