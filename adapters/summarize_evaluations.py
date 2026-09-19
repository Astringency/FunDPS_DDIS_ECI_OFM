"""Collect all verified formal splits into comparison tables after sampling."""
import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import time

from validate_evaluation import verify


TASKS = tuple(itertools.product(("ddis", "fundps", "ofm", "eci"),
                               ("poisson", "helmholtz"), ("id", "smooth", "rough")))


def read(path):
    return json.loads(path.read_text())


@lru_cache(None)
def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected(root, method, pde):
    job = root / "jobs" / f"{method}_{pde}"
    if not (job / "training_completed").is_file():
        raise ValueError(f"Training is not complete: {job}")
    state = job / "early_stopping"
    best = read(state / "best.json")
    if read(state / "completed.json")["selected_checkpoint"] != best:
        raise ValueError(f"Inconsistent validation selection: {state}")
    checkpoint = Path(best["checkpoint"]).resolve(strict=True)
    if (state / "best_checkpoint").resolve(strict=True) != checkpoint:
        raise ValueError(f"Best-checkpoint link mismatch: {state}")
    return best, checkpoint


def await_results(root, wait):
    while True:
        missing = []
        for method, pde, split in TASKS:
            job = root / "jobs" / f"evaluate_{method}_{pde}_{split}"
            if not (job / "evaluation_completed").is_file():
                missing.append(job.name)
                exit_path = job.with_suffix(".exit")
                if exit_path.exists() and exit_path.read_text().strip() != "0":
                    raise RuntimeError(f"Evaluation controller failed: {exit_path}")
        if not missing:
            return
        if not wait:
            raise FileNotFoundError(f"Formal evaluations still pending: {', '.join(missing)}")
        print(f"Waiting for {len(missing)}/24 formal evaluation splits", flush=True)
        time.sleep(60)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, destination):
    rows, cases = [], []
    for method, pde, split in TASKS:
        output = root / "evaluation" / method / pde / split
        job = root / "jobs" / f"evaluate_{method}_{pde}_{split}"
        run = read(output / "run.json")
        source = root / "data/compact" / pde
        if (run["method"], run["split"], run["count"]) != (method, split, 100):
            raise ValueError(f"Unexpected formal run: {output}")
        if run.get("profile_only", False) or Path(run["source"]).resolve() != source.resolve():
            raise ValueError(f"Profile or wrong data source: {output}")
        prior_method = "flow" if method in ("ofm", "eci") else method
        best, checkpoint = selected(root, prior_method, pde)
        if read(job / "prior_selection.json") != best:
            raise ValueError(f"Evaluation did not select the training best: {job}")
        used = run["official_config"]["pkl_path"] if method in ("ddis", "fundps") else run["checkpoint"]
        if Path(used).resolve(strict=True) != checkpoint:
            raise ValueError(f"Wrong evaluation checkpoint: {output}")
        prior_hash = checksum(checkpoint)
        if "checkpoint_sha256" in run and run["checkpoint_sha256"] != prior_hash:
            raise ValueError(f"Checkpoint changed after sampling: {checkpoint}")
        surrogate_hash = ""
        if method == "ddis":
            surrogate_best, surrogate = selected(root, "surrogate", pde)
            if read(job / "surrogate_selection.json") != surrogate_best:
                raise ValueError(f"Wrong surrogate selection: {job}")
            if Path(run["official_config"]["surrogate_path"]).resolve(strict=True) != surrogate:
                raise ValueError(f"Wrong surrogate checkpoint: {output}")
            surrogate_hash = checksum(surrogate)
            if run["surrogate_sha256"] != surrogate_hash:
                raise ValueError(f"Surrogate changed after sampling: {surrogate}")
        verified = verify(output, source, split, 100)
        records = [read(output / f"case_{index:03d}.json") for index in range(100)]
        for record in records:
            cases.append({"method": method, "pde": pde, "split": split,
                          "sample_id": record["sample_id"], "status": record["status"],
                          "relative_l2_coefficient": record["relative_l2_coefficient"],
                          "relative_l2_solution": record["relative_l2_solution"],
                          "seconds": record.get("seconds"), "error": record.get("error", "")})
        if method in ("ddis", "fundps"):
            timing = read(output / "completed.json")
            seconds, peak = timing["seconds"], timing["peak_reserved_bytes"]
            scope = "whole generation call, including solver setup and output I/O"
        else:
            seconds = sum(record["seconds"] for record in records)
            peak = max(record["peak_reserved_bytes"] for record in records)
            scope = "sum of case sampling times, excluding model setup"
        row = {"method": method, "pde": pde, "split": split,
               "cases": verified["verified_cases"], "successful_cases": verified["successful_cases"],
               "failed_cases": len(verified["failed_cases"]), "failure_rate": verified["failure_rate"],
               "failed_sample_ids": ";".join(map(str, verified["failed_cases"]))}
        for metric in ("mean_relative_l2_all_cases", "mean_relative_l2_successful_cases",
                       "median_relative_l2_successful_cases"):
            values = verified[metric] or (None, None)
            for index, field in enumerate(("coefficient", "solution")):
                row[f"{metric}_{field}"] = values[index]
        row.update(seconds=seconds, seconds_per_case=seconds / 100, timing_scope=scope,
                   peak_reserved_bytes=peak, gpu_index=int((job / "gpu_index").read_text()),
                   prior_checkpoint_sha256=prior_hash, surrogate_checkpoint_sha256=surrogate_hash,
                   result_directory=str(output))
        rows.append(row)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "summary.csv", rows)
    write_csv(destination / "cases.csv", cases)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "verified_splits": len(rows),
              "verified_cases": len(cases), "relative_error_units": "physical-field ratio, uncapped",
              "notes": ["An all-case mean is unavailable when any of its 100 cases failed.",
                        "Successful-case statistics are conditional and are not all-case results.",
                        "Shared-GPU wall times have different setup boundaries; see timing_scope.",
                        "These tables contain the four baseline methods; FM4PDE results are separate."],
              "results": rows}
    (destination / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (destination / "completed").write_text(result["created_utc"] + "\n")
    print(f"Verified {len(rows)} splits / {len(cases)} cases; saved {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    await_results(args.root, args.wait)
    summarize(args.root, args.output)
