"""Archive complete Git histories and record the two active Python environments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def run(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main(base, output):
    archive = output / "reproducibility"
    (archive / "code").mkdir(parents=True, exist_ok=True)
    (archive / "restore_checks").mkdir(exist_ok=True)
    records = {}
    repositories = {name: base / "official" / name for name in ("DDIS", "FunDPS", "OFM", "ECI", "neuraloperator")}
    repositories["orchestration"] = base / "orchestration"
    for name, repository in repositories.items():
        revision = run(["git", "rev-parse", "HEAD"], repository).strip()
        changed = run(["git", "diff", "--name-only", "HEAD"], repository).splitlines()
        if any(not path.endswith(".pyc") for path in changed):
            raise ValueError(f"Uncommitted source changes in {name}: {changed}")
        if run(["git", "rev-parse", "--is-shallow-repository"], repository).strip() != "false":
            raise ValueError(f"Incomplete Git history: {name}")
        stem = f"{name}-{revision[:12]}"
        bundle = archive / "code" / f"{stem}.bundle"
        if not bundle.exists():
            partial = bundle.with_suffix(".bundle.partial")
            run(["git", "-c", "pack.threads=2", "bundle", "create", str(partial), "--all"], repository)
            run(["git", "bundle", "verify", str(partial)], repository)
            partial.replace(bundle)
        restored = archive / "restore_checks" / f"{stem}.git"
        if not restored.exists():
            run(["git", "clone", "--bare", str(bundle), str(restored)])
        run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], restored)
        check = run(["git", "fsck", "--full", "--no-reflogs"], restored)
        (archive / "restore_checks" / f"{stem}.log").write_text(check)
        if (restored / "objects/info/alternates").exists():
            raise ValueError(f"Restore check unexpectedly depends on external Git objects: {name}")
        records[name] = {"revision": revision, "bundle": str(bundle), "sha256": digest(bundle),
                         "independent_git_restore_verified": True, "regenerated_bytecode": changed}
        print(f"Archived and verified {name} {revision[:12]}", flush=True)
    revision = records["orchestration"]["revision"][:12]
    environments = {}
    probe = ("import json,sys,torch,neuralop; print(json.dumps({"
             "'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,"
             "'torch_path':torch.__file__,'neuraloperator':neuralop.__version__,"
             "'neuraloperator_path':neuralop.__file__}))")
    for name in ("venv", "venv-flow"):
        python = base / name / "bin/python"
        freeze = run([str(python), "-m", "pip", "freeze", "--all"])
        freeze_path = archive / f"{name}-{revision}-freeze.txt"
        freeze_path.write_text(freeze)
        environments[name] = {"imports": json.loads(run([str(python), "-c", probe])),
                              "freeze_file": str(freeze_path), "sha256": digest(freeze_path)}
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "repositories": records,
              "environments": environments, "result_root": str(output),
              "limits": ["Git restore verification covers source history, not a fresh environment installation or retraining.",
                         "The flow venv imports shared dependencies from the base venv; both remain required while jobs run.",
                         "Source data and predictions are separate artifacts, not embedded in Git bundles.",
                         "Restore-check clones are task-created temporary files; clean them after final result archival."]}
    path = archive / f"runtime-{revision}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Runtime manifest: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.base, args.output)
