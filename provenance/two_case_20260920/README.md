# Provisional Poisson two-case comparison

This pilot evaluates seven sampler/prior combinations on Poisson forward, both,
and inverse tasks across ID, Smooth, Rough, Rough2, and Rough3. Each task/split
uses two sample IDs, the lowest- and highest-error FM4PDE cases among the first
100 of the independently audited 1,000-case `rough_stress_20260918` result.
Forward ranks solution error, inverse ranks coefficient error, and both ranks
the mean of the two relative L2 errors. The exact IDs and historical errors
are in `selection.json`; the source audit CSV SHA256 is in that file.

Every new method sees the same physical test case and 500 noiseless point
observations per active field. A NumPy `RandomState(0)` draws separate
coefficient and solution masks for each sample ID. Forward uses the coefficient
mask, inverse the solution mask, and both uses both masks. The masks differ from
those in the historical FM4PDE run, which serves only to select contrasting
cases. The new FM4PDE+FM4PDE result is the paired control for this pilot.

The server216 output is
`/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919/evaluation_provisional/two_case_poisson_20260920`.
`plan.json` pins current validation-best DDIS, FunDPS, OFM, and DDIS surrogate
weights before sampling. The FM4PDE ordinary prior is the fixed pretrained
checkpoint with SHA256 `93e568b9957836776e3e5d17ad59fede6b2de47a3b17b8eeaaeadb5c1391f0`.
Rough2/Rough3 source MAT files are separately hashed against the authoritative
server193 files. The original official samplers and model implementations are
unchanged. Small data/run adapters select arbitrary IDs, task-specific masks,
and corresponding active observation-channel weights. DDIS/FunDPS use their
inverse setting's active-channel weight magnitude for forward and both, without
task-specific tuning. These are exploratory two-case diagnostics from weights
saved during ongoing training, not the final 100-case comparison.

The initial tmux workers used GPUs 4 and 6. On request, the remaining
task/split cells were repartitioned across GPUs 0, 2, 3, 4, 5, 6, and 7;
`parallel_shards.json` gives the nonoverlapping assignments. GPU 1 remained
available for existing work. Each worker uses a 40 GiB free-memory gate and
has its own log and exit code. Interrupted outputs from the transition are
preserved under `workers/interrupted_*` rather than treated as results.

The report generator independently verifies physical-unit relative L2 errors
from stored predictions. Official sampler NaNs are recorded as numerical
failures with their original prediction and log retained; no retry changes
the sampling algorithm or hyperparameters. A finalizer creates
`comparison.csv`, `summary.json`, `report.md`, 15 reconstruction figures,
and a hash manifest; a local collector copies and verifies the artifacts.
