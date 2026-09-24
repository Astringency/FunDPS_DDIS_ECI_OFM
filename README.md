# FM4PDE baseline experiment: FunDPS, DDIS, ECI, OFM

Adapters for the generative baselines in the revised FM4PDE manuscript.

| Method | Approach and official implementation |
| --- | --- |
| FunDPS | Guided diffusion on function spaces; [FunDPS](https://github.com/neuraloperator/FunDPS). |
| DDIS | Decoupled diffusion sampling with a learned forward surrogate; [DDIS](https://github.com/neuraloperator/DDIS). |
| ECI | Extrapolate–Correct–Interpolate sampling with direct observation replacement; [ECI](https://github.com/amazon-science/ECI-sampling). |
| OFM | Operator flow matching with a Matérn prior and likelihood-based reconstruction; [OFM](https://github.com/yzshi5/SPL_OFM). |

ECI-FM and ECI-OFM reuse the corresponding pretrained priors. FM4PDE-OFM uses
the OFM prior with the native FM4PDE sampler. DDIS/FunDPS cover Poisson and
Helmholtz; the flow-prior comparisons cover all five main equations.

## Structure and setup

- `adapters/`: data conversion, training/validation, sampling, and independent metric checks.
- `configs/`: training, appendix sampling settings, and pinned upstream revisions.
- `scripts/`: portable data, training, and comparison launchers.
- `official/`: external checkouts populated by the command below; excluded from Git.
- `provenance/`: shared training split implementation.

```bash
python scripts/setup_official.py
```

Use separate environments: DDIS/FunDPS require the environment specified in
`official/DDIS/environment_platform.yml` and its pinned neuraloperator revision; OFM/ECI
use `configs/flow-requirements.txt` with `configs/flow-constraints.txt`.
Keep the sibling FM4PDE checkout available, and set its `DATA_ROOT` and
`CHECKPOINT_<PDE>` variables before exporting shared physical inputs.

## Examples

```bash
python scripts/prepare_data.py --raw /path/to/PDEdata --hf --plan-only
python scripts/prepare_data.py --raw /path/to/PDEdata --hf
python adapters/prepare_shared_prior_assets.py --fm4pde ../FM4PDE \
  --data-root /path/to/PDEdata --output data/shared_prior_assets
python scripts/train/run.py --method ofm --pde poisson --plan-only
python scripts/train/run.py --method ofm --pde poisson
python scripts/sample/run.py --method eci-fm --pdes poisson --plan-only
python scripts/sample/run.py --method eci-fm --pdes poisson --device cuda:0
```

Select `--method ddis|fundps|eci-fm|eci-ofm|ofm|fm4pde-ofm`, `--pdes`,
`--tasks`, and `--splits`. Defaults use 100 inputs per setting and 500 observations
per active field. DDIS/FunDPS take the GPU selected by `CUDA_VISIBLE_DEVICES`.
`--checkpoints` defaults to `checkpoints/`, containing `ddis/<pde>.pkl`,
`fundps/<pde>.pkl`, `surrogate/<pde>.pt`, and `ofm/<pde>.pt`; use the
validation-selected weights from each training run. ECI-FM uses the exported
`data/shared_prior_assets/weights/<pde>.pth` links.

`configs/eci_sampling.yaml` distinguishes the selected 200-step/one-mix schedules
(with noise refresh every five steps) from the 800-step/five-mix schedules.
DDIS uses 100 outer levels; FunDPS uses 500 diffusion steps; OFM uses 100
Langevin iterations. Failed cases remain recorded separately from successful
predictions. For a recorded OFM numerical failure, run
`python adapters/retry_ofm.py --original /path/to/case_000.json --output /path/to/retry`.
Retries use step sizes 1e-4, 1e-5, then 1e-6 (final size: 0.8 times initial),
accepting the first finite verified prediction with the same checkpoint and observations.
The FunDPS Poisson forward numerical-failure retry uses the separate
`configs/evaluation/repair_fundps_poisson_rough_weight10000.yaml` configuration.
Upstream algorithm sources and their licenses remain in their
pinned checkouts; large datasets, weights, outputs, and synchronization bundles
are excluded from Git.
