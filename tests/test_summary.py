"""Check that complete comparison tables preserve failures and uncapped errors."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
from summarize_evaluations import TASKS, await_results, summarize


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class SummaryTest(unittest.TestCase):
    def test_complete_results_reject_missing_splits_and_preserve_failed_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                await_results(root, False)
            # A physical coefficient error of 2.0 must remain above 100%.
            prediction = np.empty((1, 2, 128, 128), dtype=np.float64)
            prediction[:, 0] = 3
            prediction[:, 1] = 2
            np.save(root / "prediction.npy", prediction)
            rng, masks = np.random.RandomState(0), []
            for _ in range(100):
                rng.choice(128 * 128, 500, replace=False)
                masks.append(rng.choice(128 * 128, 500, replace=False))
            selections = {}
            for pde in ("poisson", "helmholtz"):
                source = root / "data/compact" / pde
                source.mkdir(parents=True)
                outputs = {}
                for split in ("id", "smooth", "rough"):
                    np.save(source / f"{split}.npy", np.zeros((100, 2, 128, 128), dtype=np.float32))
                    np.save(source / f"{split}_ids.npy", np.arange(100))
                    outputs[split] = {"sha256": hashlib.sha256((source / f"{split}.npy").read_bytes()).hexdigest()}
                save(source / "manifest.json", {"outputs": outputs, "stats": {"mean": [1, 2], "std": [.5, .5]}})
                for method in ("ddis", "fundps", "flow", "surrogate"):
                    job = root / "jobs" / f"{method}_{pde}"
                    state = job / "early_stopping"
                    state.mkdir(parents=True)
                    checkpoint = state / "fixture.pt"
                    checkpoint.write_bytes(f"synthetic {method} {pde}".encode())
                    best = {"checkpoint": str(checkpoint), "progress": 10}
                    save(state / "best.json", best)
                    save(state / "completed.json", {"selected_checkpoint": best})
                    (state / "best_checkpoint").symlink_to(checkpoint)
                    (job / "training_completed").touch()
                    selections[method, pde] = best, hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for method, pde, split in TASKS:
                output = root / "evaluation" / method / pde / split
                job = root / "jobs" / f"evaluate_{method}_{pde}_{split}"
                output.mkdir(parents=True)
                job.mkdir(parents=True)
                prior = "flow" if method in ("ofm", "eci") else method
                best, digest = selections[prior, pde]
                save(job / "prior_selection.json", best)
                (job / "evaluation_completed").touch()
                (job / "gpu_index").write_text("0")
                run = {"method": method, "split": split, "count": 100,
                       "source": str(root / "data/compact" / pde), "checkpoint_sha256": digest}
                if method in ("ddis", "fundps"):
                    run["official_config"] = {"pkl_path": best["checkpoint"]}
                    save(output / "completed.json", {"seconds": 26, "peak_reserved_bytes": 1024})
                else:
                    run["checkpoint"] = best["checkpoint"]
                if method == "ddis":
                    surrogate, surrogate_hash = selections["surrogate", pde]
                    save(job / "surrogate_selection.json", surrogate)
                    run["official_config"]["surrogate_path"] = surrogate["checkpoint"]
                    run["surrogate_sha256"] = surrogate_hash
                save(output / "run.json", run)
                np.save(output / "solution_observation_indices.npy", masks)
                for index in range(100):
                    failed = (method, pde, split, index) == ("ofm", "helmholtz", "rough", 7)
                    save(output / f"case_{index:03d}.json", {
                        "sample_id": index, "status": "failed" if failed else "ok",
                        "prediction": str(root / "prediction.npy"),
                        "relative_l2_coefficient": None if failed else 2.0,
                        "relative_l2_solution": None if failed else 0.0,
                        "seconds": .25, "peak_reserved_bytes": 1024})
            await_results(root, False)
            destination = root / "comparison"
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(root, destination)
            report = json.loads((destination / "summary.json").read_text())
            self.assertEqual((report["verified_splits"], report["verified_cases"]), (24, 2400))
            failed = next(row for row in report["results"] if row["failed_cases"])
            self.assertEqual(failed["failed_sample_ids"], "7")
            self.assertIsNone(failed["mean_relative_l2_all_cases_coefficient"])
            self.assertEqual(failed["mean_relative_l2_successful_cases_coefficient"], 2.0)
            self.assertEqual(failed["failure_rate"], .01)
            self.assertEqual(report["results"][0]["mean_relative_l2_all_cases_coefficient"], 2.0)
            with (destination / "cases.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2400)
            run_path = root / "evaluation/eci/poisson/id/run.json"
            run = json.loads(run_path.read_text())
            run["checkpoint_sha256"] = "wrong-checkpoint"
            save(run_path, run)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                summarize(root, root / "invalid-comparison")
            self.assertFalse((root / "invalid-comparison").exists())


if __name__ == "__main__":
    unittest.main()
