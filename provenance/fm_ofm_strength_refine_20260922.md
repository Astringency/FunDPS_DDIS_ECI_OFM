# Guidance refinement after the first complete study

The first study completed 27 settings but contained 92 numerical failures and
17 additional finite predictions whose target relative-L2 error exceeded
1000%. Completion and controller exit status did not imply numerical success.

| Setting | Numerical failures | Finite errors above 1000% |
| --- | ---: | ---: |
| Poisson forward Rough | 9 | 4 |
| Helmholtz forward Rough | 8 | 6 |
| NS forward ID | 2 | 0 |
| NS forward Smooth | 1 | 0 |
| NS forward Rough | 63 | 4 |
| NS inverse Rough | 0 | 1 |
| Burgers trajectory completion Rough | 9 | 2 |

Five settings met the manuscript FM-FM mean: Helmholtz forward ID/Smooth and
Darcy forward ID/Smooth/Rough. The continuation keeps those five results and
targets the remaining 22 settings, prioritizing the unstable groups.

## Diagnosis and frozen search

Original step logs for Poisson and Helmholtz inverse sample 0 show clipping in
all 100 steps, with total gradient norm approximately 50. Multiplying the
observation zeta by 100 therefore barely changed their predictions. Aggressive
clipping thresholds, especially 200, also caused divergence on Rough inputs.

The new frozen plan is
`configs/evaluation/fm_ofm_strength_refine_20260922.json`. Per task it includes
18–19 candidates for native zeta values, PDE/observation balance, clipping
thresholds 20–150, and stochastic guidance coefficient 0.05–0.125. Network,
100-step budget, 500 noiseless observations, seeds, loss/gradient semantics,
sampler and official source revisions remain fixed. The adapter whitelist
adds the existing official `stochastic_guidance_coeff` option; no official
algorithm is edited. Existing concurrent adapter edits were preserved during
Git synchronization.

## Data separation and decision rule

- ID validation uses entries 100–199 of the existing held-out training
  validation partition; the earlier study used entries 0–99.
- Smooth and Rough validation use original source offsets 1000–1099, exported
  through the pinned official FM4PDE loader. The full source SHA256 is checked
  before export. Formal test offsets remain 0–99.
- New local validation IDs 0–7 screen candidates on all three distributions.
  IDs 32–47 independently confirm the top three stable candidates, plus the
  anchor and clip-20 conservative candidates when eligible.
- A candidate must have zero numerical failures and no finite error above
  1000% across all three validation distributions. Ranking uses the mean of
  the three validation means; retain the anchor unless improvement is at
  least 1% when the anchor is eligible.
- Freeze parameters before running each target's complete 100-case test set.
- If test instability remains, retry that entire 100-case setting with the
  predeclared confirmed anchor, then clip 20. Select the first stable fallback,
  never the lowest test error; record this test-triggered fallback explicitly.
  If none succeeds, preserve the failure and flag the group for review.

This second search was initiated after examining the original test benchmark.
Its candidate-error selection uses separate validation data, but the repeatedly
examined benchmark must not be described as a new untouched benchmark.

## Deployment and reports

Data export ran on server197 using one CPU thread with nice priority 19; its
GPUs were not used because CPU load was saturated. All ten OOD validation
exports completed, passed source and transfer hashes, and reached server216.

The native two-case NS/Rough preflight on216 completed with zero numerical
failures, no errors above 1000%, and peak reserved memory 9.30 GiB. Eight named
tmux workers (`ddis_fm_ofm_refine_gpu{0..7}_20260922`) then deployed to216.
Admission requires at least 32 GiB free, CPU load below110 and GPU utilization
at most90%; busy-GPU admission uses two shared compute locks, while idle GPUs
can run independently. Existing native OFM numerical recovery continues.

The user-facing refresh command is unchanged:

```bash
python3 ~/C01Python/DDIS_comparison_20260919/adapters/fm_ofm_strength_study.py --fetch
```

It now reports numerical failures, finite errors above1000%, stable settings,
FM-FM target attainment, and separate refinement progress. Completed verified
round-two settings update `reports/fm_ofm_guidance_20260922/comparison.csv`;
the original study remains in that directory's `round1/comparison.csv`.
Incomplete retries cannot replace complete previous results. Neither study
silently overwrites the main seven-method `metrics.csv`.
