"""Check manual checkpoint stopping without GPUs or formal experiment data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ADAPTERS = Path(__file__).resolve().parents[1] / 'adapters'
sys.path.insert(0, str(ADAPTERS))
from plateau_priority import plateau, stop_validated_job, identity, same_live


class PlateauPriorityTest(unittest.TestCase):
    def test_plateau_eligibility_and_improving_diffusion(self):
        settings = {'relative_min_delta': .01, 'patience': 3}
        spec = {'minimum_progress': 50, 'after_progress': 40}
        history = [{'progress': i * 10, 'validation_loss': x} for i, x in enumerate(
            [.016605, .005447, .006333, .007231, .008286], 1)]
        self.assertFalse(plateau(history[:-1], spec, settings)[0])
        self.assertEqual(plateau(history, spec, settings), (True, 3))
        improving = [{'progress': i * 250000, 'validation_loss': x} for i, x in enumerate(
            [.58364, .18305, .12302, .0958, .07642, .06946], 1)]
        self.assertEqual(plateau(improving, {'minimum_progress': 1000000, 'after_progress': 1250000}, settings), (False, 0))

    def test_manual_stop_preserves_best_exit_evidence_and_unrelated_process(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Integration test requires the training environment')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / 'jobs/surrogate_poisson/early_stopping'
            models = root / 'models'
            old_policy = root / 'old.json'
            old_policy.write_text(json.dumps({'relative_min_delta': .005, 'patience': 8,
                'poll_seconds': .05, 'checkpoint_settle_seconds': .05,
                'methods': {'surrogate': {'minimum_progress': 100, 'progress_unit': 'epochs'}}}))
            producer = root / 'producer.py'
            producer.write_text("import sys,time,torch\nfrom pathlib import Path\n"
                "root=Path(sys.argv[1])/'run'; root.mkdir(parents=True)\n"
                "for epoch,loss in [(10,1.),(20,.5),(30,.6),(40,.7),(50,.8)]:\n"
                " torch.save({'epoch':epoch,'weight':torch.tensor(epoch)},root/f'checkpoint_epoch_{epoch}.pth')\n"
                " print(f'Epoch {epoch}/500 Summary:\\n  Training Loss: 1\\n  Average Batch Loss: 1\\n  Test Loss: {loss}',flush=True)\n"
                " time.sleep(.4)\n"
                "time.sleep(90)\n")
            command = [sys.executable, str(ADAPTERS / 'early_stop.py'), '--method', 'surrogate',
                '--policy', str(old_policy), '--state', str(state), '--training-root', str(models),
                '--repo', str(root), '--validation', str(root), '--', sys.executable, str(producer), str(models)]
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)'], start_new_session=True)
            supervisor = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 20
                while True:
                    path = state / 'validation_history.jsonl'
                    if path.exists() and len(path.read_text().splitlines()) == 5:
                        break
                    if time.monotonic() > deadline or supervisor.poll() is not None:
                        self.fail('Fixture did not reach five validated checkpoints')
                    time.sleep(.1)
                trainer = identity(json.loads((state / 'process.json').read_text())['pid'])
                policy = {'reason': 'test user-authorized stop', 'relative_min_delta': .01, 'patience': 3,
                    'jobs': {'surrogate_poisson': {'minimum_progress': 50, 'after_progress': 40}}}
                completion = stop_validated_job(root, 'surrogate_poisson', policy, ADAPTERS.parent, controllers=[])
                supervisor.communicate(timeout=5)
                self.assertFalse(same_live(trainer))
                self.assertIsNone(unrelated.poll())
                self.assertNotEqual(supervisor.returncode, 0)
                self.assertEqual(completion['selected_checkpoint']['progress'], 20)
                self.assertTrue((state / 'manual_plateau_stop/supervisor_signal_exit.json').is_file())
                self.assertFalse((state / 'failure.json').exists())
                self.assertTrue((state.parent / 'training_completed').is_file())
                self.assertEqual((state / 'best_checkpoint').resolve(), models / 'run/checkpoint_epoch_20.pth')
            finally:
                if supervisor.poll() is None:
                    supervisor.terminate()
                    supervisor.communicate(timeout=10)
                unrelated.terminate()
                unrelated.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
