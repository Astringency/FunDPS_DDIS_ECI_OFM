# Native OFM numerical-failure recovery, 2026-09-22

The user requested repair of OFM failed samples. Original successful cases and
all original failure records remain in `evaluation_v2/ofm/ofm`. Repaired cases
are separate in `diagnostics/ofm_numeric_repair_20260922` under the central
server216 result root.

At the initial snapshot (2026-09-22T07:43:41Z), OFM had 2,650 audited cases,
79 numerical failures, and four unstarted shards assigned to server197.
Numerical failures were in Darcy, reporting a nonfinite posterior objective or
adaptive ODE `underflow in dt 0.0`. For example, the original ID sample 5 loss
grew to `3.37047172634111e28` and sample 6 to `4.441080876716631e30` before failure.
These are separate from resource admission: four idle server197 recovery
controllers were waiting for CPU load below 110 while host load exceeded 950.
Their absence of children and GPU processes was checked before retirement.
Their four empty shards were reassigned to bounded queues on server216, using
the unchanged original parameters to complete the original baseline.

## Parameter-only retry protocol

- Official `LangevinDynamics` and `SGLD` implementations are unchanged.
- Original step-size endpoints: `1e-3`, `8e-4`.
- Retry initial step sizes, in declared order: `1e-4`, `1e-5`, `1e-6`;
  final step size is always 0.8 times the initial value.
- Each attempt keeps 100 Langevin steps, temperature 1, momentum 0,
  observation variance `1e-3`, one Hutchinson probe, `dopri5` with upstream
  tolerances, identical checkpoint, observation indices, and per-case seed.
- Only original, independently verified numerical failures are eligible.
- Accept the first finite verified prediction. Selection never compares
  ground-truth errors between candidates. All attempts retain logs/configs.
- Pilot cases: Darcy forward ID 2 and 3, forward Rough 2, inverse Smooth 12.
  Bulk workers wait for `pilot_accepted.json` before consuming the failure list.
- Workers rediscover failures from newly completed original shards, so the
  four migrated shards and the last running original shard are covered too.
- At most two bounded runtime slots per GPU, activation recomputation enabled,
  12 GiB allocation guard per process, and admission requires 30 GiB free.
  The existing recomputation equivalence checks still apply.

Reducing the step size can change mixing and accuracy at a fixed 100-step
budget. A finite prediction does not certify posterior convergence or a low
reconstruction error. This is failure-triggered test-set parameter adjustment,
not held-out hyperparameter selection.

All four pilots completed and passed independent verification with the first
retry (`1e-4` to `8e-5`). Target-field relative-L2 errors were:

| Darcy task | Split | Sample ID | Error (%) | Sampling seconds |
| --- | --- | ---: | ---: | ---: |
| Forward | ID | 2 | 36.1835557 | 672.21 |
| Forward | ID | 3 | 40.8326912 | 648.13 |
| Forward | Rough | 2 | 27.3676888 | 648.43 |
| Inverse | Smooth | 12 | 44.5613110 | 589.83 |

Peak allocated memory was at most 1.50 GiB, and peak reserved memory about
9.33 GiB per pilot. Twelve bounded repair workers were then released on
server216, with four additional workers completing the original empty shards.
At 2026-09-22T07:59:13Z, 2,660 original cases were audited: original numeric
failures had risen to 83 as the last older shard completed; the four audited
repairs reduced selected failures to 79. The unchanged count of 79 compared
with the earlier snapshot therefore does not mean the repairs were ignored.
Full repair completion remains a background task; later raw failures are
automatically discovered from newly verified original shards.

## Audit and reporting

`repair_native_ofm.py --audit --root ROOT` independently recomputes physical
relative-L2 errors and verifies source, checkpoint, IDs and observation masks.
The case audit pins original records and repaired prediction bytes by SHA256.
`verified_repair_results.py` checks those audit records before publishing only
the repaired failed IDs; original successes are retained exactly.

The existing command updates the usual report without rerunning sampling:

```bash
python3 ~/C01Python/DDIS_comparison_20260919/adapters/summarize_current_results.py
```

`metrics.csv` selects independently audited repairs as they become available;
`metrics_original.csv` preserves original results. `result_version`,
`sampling_parameters_json`, `selection_note`, and `original_n_failed` identify
the mixed configuration and original failures. Pending/unrepaired failures
remain counted. The verified-shard count measures coverage of the original
ten-case shards, not the number of repair folders.
