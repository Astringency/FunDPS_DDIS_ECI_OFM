"""Supervise unchanged official training processes using validation checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


class StopPolicy:
    def __init__(self, minimum, patience, delta):
        self.minimum, self.patience, self.delta = minimum, patience, delta
        self.best = math.inf
        self.anchor = math.inf
        self.bad = 0

    def update(self, progress, loss):
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        best = loss < self.best
        if best:
            self.best = loss
        significant = loss < self.anchor * (1 - self.delta)
        if significant:
            self.anchor = loss
            self.bad = 0
        elif progress >= self.minimum:
            self.bad += 1
        if progress < self.minimum:
            self.bad = 0
        return best, progress >= self.minimum and self.bad >= self.patience


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def candidates(method, root, log_text):
    if method in ("ddis", "fundps"):
        found = [(int(p.stem.rsplit("-", 1)[1]), p, None)
                 for p in root.glob("*/network-snapshot-*.pkl")]
    elif method == "flow":
        metrics = {int(ep): float(loss) for ep, loss in re.findall(
            r"te @ epoch (\d+)/\d+ \| Loss (\S+)", log_text)}
        found = [(ep, root / f"epoch_{ep}.pt", loss) for ep, loss in metrics.items()]
    else:
        metrics = {int(ep): float(loss) for ep, loss in re.findall(
            r"Epoch (\d+)/\d+ Summary:\s*Training Loss: [^\n]+\s*Average Batch Loss: [^\n]+\s*Test Loss: (\S+)", log_text)}
        found = []
        for ep, loss in metrics.items():
            for path in root.glob(f"*/checkpoint_epoch_{ep}.pth"):
                found.append((ep, path, loss))
    return sorted(found)


def stop_child(child):
    if child.poll() is not None:
        return child.returncode
    # Only the process group created by this supervisor is signalled.
    os.killpg(child.pid, signal.SIGCONT)
    os.killpg(child.pid, signal.SIGINT)
    try:
        return child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        return child.wait()


def interrupt_supervisor(signum, frame):
    raise KeyboardInterrupt(f"Supervisor received signal {signum}")


def main(args):
    settings = json.loads(args.policy.read_text())
    spec = settings["methods"][args.method]
    policy = StopPolicy(spec["minimum_progress"], settings["patience"], settings["relative_min_delta"])
    args.state.mkdir(parents=True, exist_ok=True)
    if (args.state / "supervisor.json").exists():
        raise FileExistsError("This supervised run already exists; use a new state directory")
    command = args.command[1:] if args.command[0] == "--" else args.command
    write_json(args.state / "supervisor.json", {"command": command, "method": args.method,
        "policy": settings, "training_root": str(args.training_root),
        "validation": str(args.validation), "selection": "minimum validation loss"})
    seen, reason, selected = set(), "budget_completed", None
    log_path = args.state / "official_training.log"
    with log_path.open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, interrupt_supervisor)
        write_json(args.state / "process.json", {"pid": child.pid, "process_group": child.pid})
        try:
            while True:
                returncode = child.poll()
                pending = False
                for progress, checkpoint, metric in candidates(args.method, args.training_root, log_path.read_text(errors="replace")):
                    if progress in seen or not checkpoint.is_file():
                        continue
                    if args.method in ("ddis", "fundps") and progress < spec["validation_interval"]:
                        continue
                    if time.time() - checkpoint.stat().st_mtime < settings["checkpoint_settle_seconds"]:
                        pending = True
                        continue
                    if metric is None:
                        result_path = args.state / f"validation_{progress}.json"
                        validation_command = [sys.executable, str(Path(__file__).with_name("validate_diffusion.py")),
                            "--repo", str(args.repo), "--method", args.method,
                            "--checkpoint", str(checkpoint), "--validation", str(args.validation),
                            "--output", str(result_path), "--seed", str(settings["validation_seed"])]
                        paused = child.poll() is None
                        if paused:
                            os.killpg(child.pid, signal.SIGSTOP)
                        try:
                            with (args.state / f"validation_{progress}.log").open("w") as validation_log:
                                subprocess.run(validation_command, stdout=validation_log, stderr=subprocess.STDOUT,
                                               check=True, timeout=3600)
                        finally:
                            if paused and child.poll() is None:
                                os.killpg(child.pid, signal.SIGCONT)
                        validation = json.loads(result_path.read_text())
                        if validation["samples"] != settings["validation_samples"]:
                            raise ValueError("Incomplete validation")
                        metric = validation["loss"]
                    else:
                        # Verify that the official writer has completed a readable checkpoint.
                        import torch
                        contents = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        if args.method == "surrogate" and contents["epoch"] != progress:
                            raise ValueError("Checkpoint/validation epoch mismatch")
                        del contents
                    best, should_stop = policy.update(progress, metric)
                    seen.add(progress)
                    record = {"progress": progress, "unit": spec["progress_unit"], "validation_loss": metric,
                              "checkpoint": str(checkpoint), "best_loss": policy.best,
                              "checks_without_significant_improvement": policy.bad, "early_stop": should_stop}
                    with (args.state / "validation_history.jsonl").open("a") as history:
                        history.write(json.dumps(record, allow_nan=False) + "\n")
                    if best:
                        selected = record
                        write_json(args.state / "best.json", record)
                        temporary = args.state / "best_checkpoint.partial"
                        temporary.symlink_to(checkpoint.resolve())
                        temporary.replace(args.state / "best_checkpoint")
                    print(json.dumps(record), flush=True)
                    if should_stop and child.poll() is None:
                        reason = "validation_early_stop"
                        returncode = stop_child(child)
                        break
                if reason == "validation_early_stop":
                    break
                if returncode is not None:
                    if returncode != 0:
                        raise RuntimeError(f"Official trainer failed with exit code {returncode}")
                    if not pending:
                        break
                time.sleep(settings["poll_seconds"])
        except BaseException as error:
            returncode = stop_child(child)
            write_json(args.state / "failure.json", {"error": str(error), "trainer_returncode": returncode})
            raise
    if selected is None:
        raise RuntimeError("Training ended without a validated checkpoint")
    write_json(args.state / "completed.json", {"reason": reason, "trainer_returncode": returncode,
        "selected_checkpoint": selected, "validation_checks": len(seen)})
    print(f"Completed: {reason}; selected {selected['checkpoint']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ddis", "fundps", "flow", "surrogate"], required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("An official training command is required after --")
    main(args)
