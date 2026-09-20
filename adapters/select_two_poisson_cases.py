"""Select contrasting Poisson cases from the verified FM4PDE stress audit."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


TASKS = ("forward", "both", "inverse")
SPLITS = ("id", "smooth", "rough", "rough2", "rough3")
FIELDS = {"forward": ("u",), "both": ("a", "u"), "inverse": ("a",)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = {}
    for row in csv.DictReader(args.csv.open(newline="")):
        if row["method"] != "fm4pde" or row["num_obs"] != "500":
            continue
        if row["task"] not in TASKS or row["distribution"] not in SPLITS:
            continue
        if not row["sample_id"].startswith("poisson_test_"):
            continue
        index = int(row["index"])
        if index >= 100:
            continue
        key = (row["task"], row["distribution"], index)
        values = grouped.setdefault(key, {})
        if row["field"] in values:
            raise ValueError(f"Duplicate result: {key} {row['field']}")
        values[row["field"]] = float(row["rel_l2"])
    selected = []
    for task in TASKS:
        for split in SPLITS:
            candidates = []
            for index in range(100):
                fields = grouped[(task, split, index)]
                assert set(fields) == set(FIELDS[task]), (task, split, index, fields)
                score = sum(fields.values()) / len(fields)
                candidates.append((score, index, fields))
            candidates.sort()
            for label, (score, index, fields) in (("good", candidates[0]), ("poor", candidates[-1])):
                selected.append({"task": task, "split": split, "label": label,
                                 "index": index, "selection_score": score,
                                 "historical_fm4pde_relative_l2": fields})
    digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    report = {"source_csv_sha256": digest,
              "source_csv": "rough_stress_20260918/audits/final/per_sample.csv",
              "selection_cohort": "first 100 cases in each split",
              "criterion": "minimum/maximum task-field mean physical relative L2 over existing FM4PDE results",
              "observations_per_active_field": 500, "cases": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Selected {len(selected)} cases into {args.output}")


if __name__ == "__main__":
    main()
