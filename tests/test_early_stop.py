"""Exercise checkpoint selection and stopping against a real child process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
from early_stop import StopPolicy, candidates


class EarlyStopTest(unittest.TestCase):
    def test_warmup_and_small_improvements(self):
        policy = StopPolicy(minimum=5, patience=2, delta=0.01)
        for epoch in range(1, 5):
            self.assertFalse(policy.update(epoch, 1)[1])
        self.assertFalse(policy.update(5, 0.999)[1])
        best, stop = policy.update(6, 0.998)
        self.assertTrue(best)
        self.assertTrue(stop)
        with self.assertRaises(FloatingPointError):
            policy.update(7, float("nan"))

    def test_official_fno_log_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run").mkdir()
            checkpoint = root / "run/checkpoint_epoch_10.pth"
            checkpoint.touch()
            found = candidates("surrogate", root,
                "Epoch 10/500 Summary:\n  Training Loss: 0.4\n  Average Batch Loss: 0.4\n"
                "  Test Loss: 0.12\n  Epoch Time: 2.0s\n")
            self.assertEqual(found, [(10, checkpoint, 0.12)])

    def test_cancelling_supervisor_stops_its_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"relative_min_delta": .01, "patience": 2,
                "poll_seconds": .05, "checkpoint_settle_seconds": .05,
                "methods": {"flow": {"minimum_progress": 3, "progress_unit": "epochs"}}}))
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "adapters/early_stop.py"),
                "--method", "flow", "--policy", str(policy), "--state", str(root / "state"),
                "--training-root", str(root / "models"), "--repo", str(root), "--validation", str(root),
                "--", sys.executable, "-c", "import time; time.sleep(90)"]
            supervisor = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 10
                pid_path = root / "state/process.json"
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                child_pid = json.loads(pid_path.read_text())["pid"]
                supervisor.terminate()
                supervisor.communicate(timeout=10)
                self.assertNotEqual(supervisor.returncode, 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertTrue((root / "state/failure.json").exists())
            finally:
                if supervisor.poll() is None:
                    supervisor.terminate()
                    supervisor.communicate(timeout=10)

    def test_checkpoint_selection_and_process_isolation(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Run the integration check in the server training environment")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            producer = root / "producer.py"
            producer.write_text(
                "import sys,time,torch\nfrom pathlib import Path\n"
                "root=Path(sys.argv[1]); root.mkdir()\n"
                "for epoch, loss in enumerate([1.0, .9, .899, .8995], 1):\n"
                "    torch.save({'weight': torch.tensor(epoch)}, root/f'epoch_{epoch}.pt')\n"
                "    print(f'te @ epoch {epoch}/100 | Loss {loss:.6f} | 1 (s)', flush=True)\n"
                "    time.sleep(2)\n"
                "time.sleep(60)\n")
            policy = root / "policy.json"
            policy.write_text(json.dumps({"relative_min_delta": .01, "patience": 2,
                "poll_seconds": .05, "checkpoint_settle_seconds": .05,
                "methods": {"flow": {"minimum_progress": 3, "progress_unit": "epochs"}}}))
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"], start_new_session=True)
            try:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "adapters/early_stop.py"),
                    "--method", "flow", "--policy", str(policy), "--state", str(root / "state"),
                    "--training-root", str(root / "models"), "--repo", str(root), "--validation", str(root),
                    "--", sys.executable, str(producer), str(root / "models")],
                    capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIsNone(unrelated.poll(), "Supervisor stopped an unrelated process")
                completion = json.loads((root / "state/completed.json").read_text())
                self.assertEqual(completion["reason"], "validation_early_stop")
                self.assertEqual(completion["selected_checkpoint"]["progress"], 3)
                self.assertEqual((root / "state/best_checkpoint").resolve(), root / "models/epoch_3.pt")
                self.assertEqual(completion["validation_checks"], 4)
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
