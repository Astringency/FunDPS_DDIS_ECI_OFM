"""Only set the seed and relative data root for official surrogate training."""
import argparse
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--workdir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.chdir(args.workdir)
script = args.repo / "scripts/train/training_fno.py"
sys.argv = [str(script), "--config", str(args.config)]
runpy.run_path(str(script), run_name="__main__")
