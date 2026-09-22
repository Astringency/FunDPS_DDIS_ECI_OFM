# Current protocol

See [the September 20 protocol revision](provenance/protocol_20260920_v2.md) and `configs/evaluation_matrix_v2.json` for the current authorized scope. Earlier operational notes below are historical.

# DDIS and generative baselines on FM4PDE data

## Expanded scope — 2026-09-20, supersedes the two-PDE scope below

The user now requests two comparisons on Poisson, Helmholtz, Darcy,
Navier–Stokes (`nsnonbounded`) and Burgers (`burger`): (1) ECI and FM4PDE
sampling with each existing FM4PDE ordinary-FM checkpoint; (2) ECI, OFM and
FM4PDE sampling with each OFM-trained operator-FM checkpoint. Prioritize
Poisson/Helmholtz OFM training, then train the other three OFM priors.
Reuse each group's exact pretrained weights; preserve the original algorithms.

The user confirmed ID/Smooth/Rough IDs 0–99, 500 noiseless observations,
inverse tasks for the first four PDEs, and FM4PDE's trajectory-completion task
for Burgers. Retain the original DDIS/FunDPS comparisons on Poisson/Helmholtz.
The expanded target is 87 formal split evaluations / 8,700 cases, with the
original OFM/ECI evaluations included in the shared-OFM-prior group.
See `provenance/scope-expansion-20260920.json`. These new runs are not complete.

DDIS has two independently trained components. The `ddis_*` jobs train the
diffusion prior over the unknown field; the `surrogate_*` jobs train the forward
FNO from true paired fields `(a, u)`. The FNO does not consume diffusion-generated
training data or wait for the diffusion checkpoint. Official DAPS combines both
components during posterior sampling. Report them as “DDIS diffusion prior” and
“DDIS forward FNO”; completion of the FNO alone does not complete DDIS training
or its inverse evaluation.

The provisional runner samples from the current validation-best checkpoint
while training continues. It freezes the checkpoint path and hash in a separate
`evaluation_provisional` result folder, uses the original full sampling schedule,
and independently verifies physical-unit predictions. It accepts
`PRIOR METHOD PDE SPLIT COUNT GPU OUTPUT_ROOT`, where `COUNT` is 1 (single-case
pilot) or 100 (the confirmed evaluation scale). For example, on server216:

```bash
BASE=/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919
OUT=/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919
RUN=$OUT/evaluation_provisional/ddis_poisson_id_$(date +%Y%m%d_%H%M%S)
bash "$BASE/orchestration/adapters/run_provisional_evaluation.sh" ddis ddis poisson id 1 6 "$RUN"
```

Change `1` to `100` for the full split. For an ordinary-FM checkpoint, use
`fm4pde eci poisson id 100`; for an OFM checkpoint, use
`ofm eci poisson id 100`, `ofm ofm poisson id 100`, or
`ofm fm4pde poisson id 100`. Select a GPU with at least 40 GiB free. A single
DDIS/FunDPS case takes several minutes at the published sampling schedule;
the 100-case run takes correspondingly longer. The runner does not publish
provisional results as completed formal comparisons.

The user also authorized stopping DDIS/FunDPS at validated checkpoints when
they plateau. `configs/plateau_priority.json` specifies three consecutive
checks without 1% significant improvement, eligible from 1M diffusion images
or epoch 50 for FNO surrogates, and only after the checkpoint current at the
request. OFM's existing early-stop policy is unchanged. For legacy running
supervisors, `adapters/plateau_priority.py` preserves the intentional signal
exit in an audit directory and publishes completion only after process exit
and validation-best checkpoint verification; it does not restart training.

Expanded implementation checkpoint, server216 2026-09-20 11:35:

- Both DDIS forward FNOs have stopped at confirmed plateaus. Poisson selected
  epoch 20 (validation loss 0.005447); Helmholtz selected epoch 70 (0.002899).
  Four diffusion priors and both priority OFM priors continue. Both priority OFM
  priors reached epoch 100 with five checks without significant improvement.
  OFM runs at about 206 seconds per epoch; stopping time depends on future validation.
- Darcy/NS/Burgers authoritative exports are complete on server197, with
  45,000/5,000 training/validation cases and three 100-case tests. Checksummed
  relays for Darcy/Burgers are complete; the NS relay and three training controllers
  are live. Burgers passed
  both profiles (batch 100 peak 23,544,725,504 bytes) and is training on GPU 0,
  trainer PID 583046 / supervisor PID 583039; first five epochs took about
  149.3 seconds each. Its first validation at epoch 10 is 0.015166. Darcy's
  five arrays passed checksum verification and both GPU profiles passed;
  batch 100 peak reserved memory is 23,544,725,504 bytes. Formal Darcy training
  is live on GPU 1, trainer PID 630977 / supervisor PID 630976. NS still waits
  for its checked transfer and then training capacity.
- The five existing ordinary-FM checkpoints are selected by the pinned current
  `configs/main` files, not by modification time. All are schema-3 checkpoints
  saved at epoch 299, with their own channel standardizers and no saved EMA.
  Native evaluation inputs, including NS viscosity/time parameters, were exported
  with unmodified `sampling.data.load_ground_truth`. Assets are still transferring.
- FM4PDE source is frozen at `cbe627cb85cd6cbba7083671d256f3bf418792e7` on both
  hosts. `evaluate_shared_prior.py` calls official ECI/OFM and native FM4PDE
  routines. Runtime adapters provide argument order, conditioning, normalization,
  and the trained noise prior. OFM's Matérn GP is used for both native FM4PDE
  initial noise and stochastic bridge refreshes; ordinary FM retains iid noise.
- Four CPU integration checks passed on server216 (1.007 seconds): gradients
  through frozen models, both noise draw sites and restoration, official ECI
  constraints/namespace isolation for one/two channels, and OFM normalization.
  The isolated runtime uses Python 3.13.7 / Torch 2.8.0, inheriting the existing
  FM4PDE environment without modifying it. Optional package imports and exact OT
  were exercised; unused optional dependencies are recorded in the setup log.
- All 87 evaluations are explicitly listed in `configs/expanded_evaluation_matrix.json`.
  The 24 original controllers remain; 63 added controllers are live and waiting
  for verified assets, completed prior selection when applicable, and GPU capacity.
  Each addition must pass a full-schedule single-case preflight before its 100 cases.
  `validate_shared_prior.py` independently recomputes physical errors and preserves
  failures without capping or substituting successful-case means for full means.
- Four complete-schedule GPU compatibility profiles passed and were independently
  verified: native FM4PDE on provisional Poisson OFM epoch 90; and ECI/OFM/FM4PDE
  on the Burgers resource-profile prior. These are not formal accuracy results.
  See `provenance/shared-prior-gpu-profiles.json`; selected-final-weight preflights
  are still mandatory. Ordinary-FM weight transfer and GPU checks remain pending.
- All 15 native truth splits match their compact exports to float32 roundoff.
  The first case of every split passes native PDE-loss/input-gradient checks and
  has exactly 500 active solution observations. NS retains the native approximate
  endpoint-secant residual. Evidence: `provenance/shared-prior-input-checks.json`.
- `summarize_expanded_evaluations.py` now waits for all 87 verified splits and
  checks identical weights within each prior/PDE group, all masks, native/compact
  truth alignment, and all 8,700 case records. It also creates 60 paired comparisons
  with failed cases retained. Four completeness/failure-accounting tests passed.
  The expanded summary controller is live; no formal evaluation has completed.
- The bounded Darcy transfer-priority controller completed normally after 23
  minutes and restored upload PIDs 653639 (NS) and 655881 (ordinary-FM assets).
  Both uploads are running again; no upload remains intentionally suspended.
  Audit: ignored task-local `transfers/darcy-priority.json`, exit 0.
- New OFM checkpoints are readable with finite parameters, and the Poisson and
  Helmholtz parameters changed between checkpoints. Best links match the minimum
  recorded validation loss. The first Burgers validation checkpoint also passed.
  Evidence: `provenance/flow-checkpoint-audit-20260920-1130.json`.
- The runtime archive now includes all six official source repositories, the
  orchestration source through `b1082a76fbbe`, and records for all three Python
  environments. Seven Git bundles passed independent restore checks; all bundle
  and environment-freeze hashes match. Server evidence:
  `reproducibility/runtime-b1082a76fbbe.json`. This archives runtime provenance;
  training, ordinary-FM GPU checks, formal evaluations and final results remain pending.

## Current checkpoint — 2026-09-20 (supersedes historical status below)

The user authorized minimal data/runtime adapters with official algorithms unchanged, then **removed DiffusionPDE and requested early stopping for every remaining model**. Active scope: DDIS, FunDPS, OFM and ECI on Poisson/Helmholtz. OFM and ECI share one official joint prior per PDE (paper H.3). Formal training started on 2026-09-20 around server time 01:23. No formal training or final evaluation has completed at this checkpoint.

- All official model/training/sampling source checkouts remain unchanged. Adapters are versioned separately. `adapters/early_stop.py` supervises child training processes and selects the minimum-validation-loss checkpoint. A plateau means eight validation checks without a 0.5% relative improvement. Small absolute improvements still update the saved best checkpoint. Configuration: `configs/early_stopping.json`.
- DDIS/FunDPS retain the official 10M-image maximum and 5M-image learning-rate warmup. Validation runs every approximately 250K images; early stopping only counts plateau checks after 5M images. It calls the loss function and EMA serialized by the official trainer on all 5,000 held-out validation cases, using fixed noise seed 20260920, no data augmentation and profiled validation batch 8. The supervisor temporarily suspends only its own trainer while validating. No final test distribution is used for early stopping.
- OFM/ECI retain the 300-epoch maximum, validate every 10 epochs with the official training implementation and enable early stopping after epoch 50. DDIS's FNO surrogate retains the 500-epoch maximum, validates and checkpoints every 10 epochs, and enables early stopping after epoch 100. Both use the official metrics on the separate 5,000-case validation split. Logs call it `test`; the adapter deliberately points that loader to validation data.
- `jobs/{method}_{pde}/early_stopping/` stores policy, official training log, validation history, best checkpoint link/metadata, and completion reason. `training_completed` is written only after a successful supervised run. Sampling must consume `best_checkpoint`, not assume the final numbered checkpoint is best.
- DiffusionPDE had only short resource probes. No full DiffusionPDE training was launched. Its formal launcher branch and 45K-image preprocessing are removed. Existing resource probes are historical diagnostics, not benchmark results.
- Authoritative exports from server197 contain the exact reference 45K/5K membership, normalized float32 values, original IDs and SHA256 hashes. Both exports, both checksummed transfers and all ten HF conversions (train, validation, ID, Smooth, Rough for each PDE) are complete. The two preprocessing exit codes are zero. Relay/preprocessing sessions ended normally. Authoritative compact/HF arrays are ready on server216; hidden partial arrays were never used for training.
- Persistent SSH socket: `/tmp/ddis216-recovered-20260919`. Reuse it. Base environment: PyTorch 2.7.1+cu126 and pinned neuraloperator 2.0.0. Separate flow environment: neuraloperator 0.3.0, torchcfm 1.0.5, torch-harmonics 0.7.2; it imports shared dependencies through a `.pth` file, so the base environment must remain available.
- Resource profiles completed: DDIS batch32 FP32 about 39.76 seconds/1K images; FunDPS batch32 FP32 about 20.08 seconds/1K images. Official FP16 flags fail a dtype assertion in both UNO trainers, so FP32 is retained. Shared flow prior batch100 uses about 23.5GB reserved; FNO surrogate batch32 about 19.5GB reserved. These are short profiles on shared GPUs, not convergence measurements.
- Formal GPU slots are DDIS Poisson/Helmholtz 3/2, surrogate 0/1, shared flow prior 5/7, and FunDPS candidate pool 4/6. GPU3 became idle before launch, so the heavier DDIS Poisson prior moved there and its surrogate moved to GPU0. GPU4 is occupied by unrelated jobs; memory gates and experiment locks prevent unsafe starts. The supervised jobs started at server time 01:23 after checks and data preparation completed. FunDPS Helmholtz acquired GPU4 at 01:31 when unrelated memory was released. A live PID check at 01:31:30 confirmed all eight official trainers running, with no supervisor failure records. Both shared flow priors completed epoch 1 with finite losses (Poisson 0.147292, Helmholtz 0.152102; about 453 seconds/epoch). DDIS and FunDPS diffusion logs also have finite losses; these early training losses are not validation or reconstruction results. Check actual processes and per-job logs for newer progress.
- Ground-truth PDE residuals and field normalization were checked; common masks match all 100 official DDIS/FunDPS masks. See `provenance/ground-truth-residuals.json` and `provenance/pde-parameters.json`. Official DDIS FNO currently has 34,653,889 parameters with its pinned dependency/config, differing from the paper's approximate table size; preserve the official implementation and disclose this difference.
- All four full-schedule single-case sampling probes passed with finite predictions, common masks and independently recomputed physical-unit metrics. These use resource-only weights and are not accuracy results. `provenance/full-sampler-checks.json` records runtime/memory. DDIS's generic entry imports missing optional modules; `adapters/evaluate_diffusion.py` calls the existing official multiresolution solver directly. Its config now selects `daps_multires`, so the paper's 64-to-128 resolution transition and phase-specific weights actually take effect. No official source was edited.
- The initial eight method/PDE evaluation controllers were superseded before sampling began by 24 independent `ddis_eval_{method}_{pde}_{split}_20260920` controllers. ID, Smooth and Rough can therefore use separate free GPUs once training slots are released. `adapters/formal_evaluate.sh` waits for completed early-stopping checkpoint selection (and DDIS's surrogate), profiles one case from its own distribution with trained weights, then calls the official sampler on all 100 cases. It prioritizes remaining training queues, checks free GPU/host memory and holds an experiment GPU lock. Logs/selection metadata are in `jobs/evaluate_{method}_{pde}_{split}/`; old controllers have a `superseded` marker.
- `adapters/validate_evaluation.py` verifies source hashes, exact IDs, common masks, all prediction shapes/finiteness and independently recalculated physical relative L2 errors. It reports uncapped errors and failures separately; it never substitutes a successful-case mean for an all-case mean. Each formal split must have `verified_summary.json` covering 100 cases before the corresponding method/PDE/distribution controller writes `evaluation_completed`. Completion of the requested evaluation requires all 24 verified 100-case splits.
- `adapters/summarize_evaluations.py --root RESULT_ROOT --output RESULT_ROOT/comparison --wait` waits for those 24 formal completions, checks validation-selected checkpoint identities/hashes and independently verifies all saved predictions before writing `summary.csv`, `cases.csv` (2,400 rows) and `summary.json`. It labels conditional statistics and differing timing boundaries explicitly; these are baseline results for a later FM4PDE comparison, not FM4PDE results. OFM/ECI now also record checkpoint SHA256 at sampler startup. The local synthetic integration check `python3 -m unittest discover -s tests -p test_summary.py -v` passed, covering missing splits, an uncapped error above 100%, a failed case and a mismatched checkpoint. Synthetic fixtures are temporary and are never used as formal results.
- Early-stopping verification passed four integration/unit checks, including child-process cancellation and isolation. The two batch-8 validation profiles repeated the same seeded loss within tolerance. DDIS validation reserved about 4.58GB; FunDPS about 0.71GB, both inside the training slots' headroom. Evidence: `provenance/early-stopping-checks.json`. Official source files are unchanged; OFM tracks some regenerated Python bytecode, recorded separately.
- At server time 2026-09-20 03:36, all eight formal trainers and their early-stopping supervisors were confirmed live after their first real 5,000-case validation. Each selected checkpoint/link matches the minimum recorded validation loss. All four diffusion trainers resumed after external EMA validation; both flow priors and both FNO surrogates used the intended held-out validation data. Helmholtz FNO has already updated its best checkpoint from epoch 10 to epoch 20. The inspectable snapshot is `provenance/first-formal-validation.json`, also copied to the result directory's `reproducibility/first-formal-validation.json`. This verifies the running validation/selection workflow, **not finished training or final sampling**; all eight training completions and all 24 evaluation completions were still pending.
- Source archival completed with exit code 0 at server time 01:50. The result directory's `reproducibility/runtime-49ce2fb265d4.json` records six complete Git bundles (four active methods, pinned neuraloperator and orchestration), SHA256 hashes and both Python environments. Each bundle restored independently and passed `git fsck`; this verifies source restoration, not a fresh environment installation or retraining. Keep both live environments until training/evaluation ends. A subsequent 01:51 PID check confirmed all eight official trainers still running, with no early-stopping failure records; formal training and evaluation completion counts remain zero.
- Current planning allowance excluding DiffusionPDE is about 5–7 days including sampling; early stopping may reduce this, but shared GPU contention and validation curves determine actual completion.

Flow protocol source: [DDIS Appendix H.3](https://arxiv.org/html/2601.23280v3#A8.SS3).

## Requested outcome

Train the official DDIS and generative baselines from arXiv:2601.23280 on server216, using FM4PDE data for two PDEs, then sample and evaluate 100 test cases on each of ID, Smooth and Rough for comparison with FM4PDE. Preserve official model, training and sampling implementations. Allocate GPUs concurrently after checking live jobs and measuring memory with small batches.

Working PDE selection: Poisson and Helmholtz, both supported by the paper and FM4PDE. Working task: inverse reconstruction from 500 noiseless solution observations, matching the FM4PDE inverse configs. Current target methods: DDIS, FunDPS (called FuncDPS in the request), OFM and ECI. DiffusionPDE was removed by the user on 2026-09-20. The scope is **not complete** until all required training and evaluation results are verified.

## Official source revisions

All five model repositories were cloned from GitHub locally and transferred unchanged through Git bundles to server216. `official/` contains separate repositories, not reimplementations.

| Method | Official repository | Revision |
|---|---|---|
| DDIS | https://github.com/neuraloperator/DDIS | `0af48666d5eb80b6a8b41b7e730db9d40a0b5cac` |
| FunDPS | https://github.com/neuraloperator/FunDPS | `59933932614df13a7be4796a1d9b815a792ed7c1` |
| OFM | https://github.com/yzshi5/SPL_OFM | `d4e99d229c8698f75298042ba22b567ba5eeb45d` |
| ECI | https://github.com/amazon-science/ECI-sampling | `8c3061a70559b239c1cb3044662ba5de991fc949` |
| DiffusionPDE | https://github.com/jhhuangchloe/DiffusionPDE | `1e2bc8b9e312f3a936630a30d2f49aedabf0cea7` |

DDIS explicitly requires neuraloperator commit `98cd305099f4a2b232ed85773984f3e5991f9b1a`; this dependency was also cloned and transferred through Git. Paper version inspected: https://arxiv.org/html/2601.23280v3 . HF paper metadata endpoint returned 404, so arXiv was used directly.

## Locations and initial state (2026-09-19)

- Local orchestration directory: `/home/tat512/C01Python/DDIS_comparison_20260919`.
- Server216 code and isolated environment: `/data1/zjinzxf2025/C01Python/DDIS_comparison_20260919`.
- Absolute long-term result directory: `/data1/zjinzxf2025/C01Python/DiffusionPDE/outputs/ddis_comparison_20260919`.
- Authoritative FM4PDE raw data: server197 `/large_storage/zhangxf/PDEdata`; manifest `provenance/data-files.txt` lists ten training shards and six test files.
- Local `tmux` session `ddis_data_pull_20260919` runs resumable rsync into `staging/raw`; logs and exit code are under `provenance/data-pull.*`. Revalidate the live process before taking any restart action.
- Server216 setup session `ddis_setup_mirror_20260919` installs Python 3.10 / PyTorch 2.7.1 dependencies in `venv`; logs/exit code under the result directory `setup/environment-mirror.*`. Initial direct PyPI setup failed with a verified download timeout (`setup/environment.exit = 2`); retry uses the Tsinghua PyPI mirror. Pinned neuraloperator must be installed from its local official checkout **after** the base installation succeeds.
- No model training or formal evaluation has started. No custom model or sampler code has been written.

Additional live preparation jobs:
- Local `ddis_data_push_20260919` uploads completed raw files concurrently with the pull, excluding rsync temporary dotfiles, into the result directory `data/raw/`. Logs/exit code: `provenance/data-push.*`.
- Server216 `ddis_dependency_20260919` waits for successful base environment installation, installs the pinned local neuraloperator checkout, records `setup/pip-freeze.txt`, and checks DDIS dependency imports. Log/exit: `setup/dependency.*`.
- Server216 `ddis_pilot_data_20260919` waits for dependency success and the first authoritative Helmholtz shard, verifies its source SHA256, and invokes the unmodified official `utils/dataset_process.py` in an isolated pilot working directory. It intentionally has only shard 1 (10,000 cases); official warnings for missing shards 2–5 are expected **only for this pilot dataset**. Do not use it as the formal 50,000-case pool. Log/exit: `setup/pilot-data.*`.
- `configs/pilot_{ddis,fundps}_helmholtz_b1.yaml` configure 1,000-image memory pilots with microbatch 1 and effective batch 32. They retain each upstream model configuration (DDIS 128, FunDPS 64), fixed seed 0, offline W&B and absolute output paths. These GPU pilots have **not** been launched yet. Recheck resources before launching.
- Config/runbook Git revision `9298ef8` was bundled as `orchestration-pilots.bundle` for a separate server216 `orchestration/` checkout. Config changes are versioned locally; official source files remain unchanged.
- A task-specific SSH multiplex socket `/tmp/ddis216-20260919-ssh` exists locally. Bulk transfer can delay other channels; `ssh -S none server216` is available for independent status checks. A transient SSH reset is not job failure.

The first resource check found eight A800 80GB GPUs, 920GiB available host RAM and 128 logical CPUs. GPUs 0 and 4 were temporarily idle; existing FM4PDE queues target both. Do not treat this snapshot as a reservation: inspect current GPU processes and queues again immediately before any pilot or training launch. Preserve all other jobs and sessions.

## Compatibility findings and pending clarification

An asynchronous question asks whether minimal data-loading/configuration/evaluation adapters are permitted by “不要自己编写代码”, while retaining all official algorithms. No answer has been received at this checkpoint. Do not treat elapsed time as approval. DDIS/FunDPS preparation can proceed using their official preprocessing and configurable entry points.

- DDIS and FunDPS provide official MATLAB-to-Hugging-Face conversion scripts and Poisson/Helmholtz training/generation entry points. Raw fields match FM4PDE: `f_data` with `phi_data` or `psi_data`. The converter uses fixed published normalization statistics and names raw files differently; path symlinks can accommodate this without model edits.
- DDIS FNO surrogate training hard-codes dataset paths relative to its checkout. Its YAML controls output path, batch size, epochs and checkpoints. Ensure its validation data does not accidentally substitute formal test results for an independent validation split.
- ECI's registry supports NS, Darcy, Stokes, diffusion, Stefan and PME; it lacks Poisson/Helmholtz loaders. Its Darcy loader reads only solution fields and is not a joint coefficient/solution inverse solver. Do not relabel a solution-only experiment as the requested baseline.
- OFM provides notebook entry examples and reusable `OFMModel.train`; no turnkey FM4PDE inverse evaluation CLI exists. `util/true_gaussian_process.py` imports `ofm_utils.util`, although the checkout directory is named `util`. Investigate an upstream-compatible namespace setup before modifying source.
- DDIS paper Appendix H.3 extends flow baselines to a two-channel coefficient/solution prior. Its public DDIS repository has no OFM/ECI pipeline. Paper flow settings: 300 epochs, batch 100, Fourier modes 32, hidden/projection width 128, Matérn lengthscale .01, variance 1, nu .5, sigma_min 1e-4. Do not claim the released ECI examples already reproduce this extension.
- Official DiffusionPDE `merge_data.py` is hard-coded for Darcy. Poisson/Helmholtz preprocessing requires adapting only its field names/scales or finding an official supported equivalent.

## Data difference verified

All ten server216 `/data0/zhangxf/PDEdata` training files differ in whole-file SHA256 from the same names on server197. A read-only numerical comparison of Poisson shard 1 confirmed that the actual arrays also differ (both hashes and standard deviations). Do not reuse the old server216 raw training data. Rsync is fetching the authoritative files.

Server216 Poisson shard 1 first 100 array hashes (float64 contiguous bytes):
- `f_data`: `cee6f37f5d87a96dcd580e1e46aa2b45605333ea68e9aca2f290c46a47de4e83`
- `phi_data`: `ac3010c6da4b47bb32ec1ea53175c133b7f0b637ddcd89175a19f90ee8bc4c8a`

Server197 authoritative Poisson shard 1 first 100 array hashes:
- `f_data`: `6051b8b824955b0c0a1d1aaebe2d47fcd8fb89a680ddb1a2620c27d18b1a096d`
- `phi_data`: `4c602ae81032e43affd06f4a2928a344fa675ebca643923b0ec5e0951d2be760`

FM4PDE's current training loader makes a 45,000/5,000 split from the five shards using a seeded `torch.randperm`, with PDE-specific seed offset. Resolve checkpoint-specific membership before claiming identical training/validation splits. Keep evaluation cases, observations and physical-unit metrics inspectable. Initial proposed evaluation IDs are 0–99 in each distribution; the requested 100 are test cases, not 100 posterior draws per case.

## Next actions

1. Revalidate setup, transfers, source array inspection and any user answer; finish environment installation and dependency import checks.
2. Verify exact data sources, sync data to server216, and use official preprocessing. Record normalization and sample/split membership.
3. Configure official training with absolute output directories and fixed seeds. Preserve full documented training budgets; memory pilots are not completed training.
4. Profile small batches, then allocate free GPU capacity across independent model/PDE jobs in identifiable tmux sessions. Keep logs, config, exit status and Git revisions.
5. Finish training and run ID/Smooth/Rough evaluation on the same 100 cases and observations per PDE. Report physical-unit relative errors, per-case results, time, finite-output checks and protocol differences.
6. Verify all method/PDE/distribution results and checkpoints before claiming completion. Clean only this task's temporary files and completed sessions, after preserving reproducibility in the main result directory.

## Continuation checkpoint (2026-09-19, about 23:55 CST)

- Base environment and pinned neuraloperator installation completed. The official DDIS dependency import check printed `2.7.1+cu126 12.6 2.0.0 3.6.0` (PyTorch / CUDA / neuraloperator / datasets). Read the exit files to reconfirm before launch.
- Local revision `6e3ffa6` prepared four full-duration diffusion training configs, two 500-epoch surrogate configs and twelve first-100 ID/Smooth/Rough generation configs for DDIS/FunDPS. Checkpoint paths are required CLI inputs, not guessed files. Batch sizes remain conservative pending profiling. Full duration and upstream architectures are preserved. These configs have **not** run.
- The original single-connection push was stopped deliberately after poor throughput (roughly 415MiB in 14 minutes). Its incomplete Helmholtz shard was moved to `data/raw/helmholtz/.initial-upload.partial`; it must not be treated as valid raw data. Its full source SHA256 is `30d91fc73fcf4d5b86bbaa2385718a8ed1dde90a2bf96bcb7bac8681a270b0ba`.
- A small parallel chunk-transfer experiment was stopped after server216 returned `Exceeded MaxStartups` and connection refusals. Do not restart the rapid new-connection transfer. Its local tmux session `ddis_pilot_chunks_20260919` and original push session `ddis_data_push_20260919` are stopped; at most 22 1MiB chunks were sent under `data/transfer_chunks/helmholtz_10000-128-128_1.mat/`. Keep these task-created temporaries until safe cleanup. The experimental transport script was removed from the working tree; Git retains its history.
- Original local data pull `ddis_data_pull_20260919` is still live and progressing; do not restart it. About 10GB of 40GB had arrived. Revalidate its process and progress before action.
- The old SSH socket `/tmp/ddis216-20260919-ssh` disappeared. Re-establish a persistent connection after the transient SSH startup limit clears, then resume upload using that connection and a hidden partial directory. Avoid creating many short SSH connections. Direct server197-to-server216 transfer was successfully authenticated using an ephemeral forwarded agent, but measured 32MiB / 246.94s; this did not solve throughput. No key was copied to either server.
- Recheck `setup/pilot-data.exit` and log: the waiting official preprocessing job appears to have exited during the brief interval when rsync left an incomplete file under its final name. The checksum guard must reject this file. Restart preprocessing only after inspecting the terminal status, and wait for an atomically published, checksum-verified complete source file.
- `orchestration-current.bundle` exists locally but its last scp failed with connection refusal. Server216 orchestration is still at `9298ef8` unless current Git state proves otherwise. Sync the latest local Git revision once connectivity recovers.
- The minimal-adapter clarification remains unanswered. Continue official DDIS/FunDPS work; do not infer permission for new OFM/ECI/DiffusionPDE adapters from elapsed time.
# Refresh current sampling statistics (2026-09-21)

From the local workstation, run this single command from any directory:

```bash
python3 ~/C01Python/DDIS_comparison_20260919/adapters/summarize_current_results.py
```

It reads the latest small JSON records on `server216` over SSH and writes
`reports/five_method_comparison_20260921/metrics.csv` (159 method/PDE/task/split
rows), plus `progress.csv`, `metrics.json`, `live_report.md`,
`latest_summary.json` and a reproducible `latest_source_snapshot.json.gz`.
The CSV covers ECI-FM, ECI-OFM, DDIS, FM-FM, FM-OFM, native OFM and FunDPS.
FM-FM is parsed from the **active, uncommented main tables** in
`~/C04Papers/fm4pde_jmlr/fm4pde_jmlr_revision.tex`. Its `n` and `expected_n`
come from each table caption's test sample count, not the sampler step counts
in the headers. Logical LaTeX rows may span multiple source lines; columns
are located by their headers. `source_table` and `source_line` identify each
FM4PDE metric. DDIS/FunDPS cover the two
trained PDEs, Poisson and Helmholtz. Each other included evaluation targets 100
cases. No model loading, sampling, GPU work or modification of remote results
occurs. The script uses only the Python standard library.

Check `status`, `n_saved`, `n_verified`, `n_finite` and `n_failed` before using
an error statistic. All error columns use **percent**. `mean_percent` is blank
unless all 100 cases are verified and successful. `finite_mean_percent` and the
other `finite_*` fields summarize only verified successful cases and may cover
a partial evaluation; extreme finite errors are retained. Resource/worker
errors are separate from verified numerical failures. Counts from unfinished
shards are saved but do not enter error statistics until verification finishes.
This command checks metadata, IDs and saved verification means; it does not
revalidate all prediction arrays.

`verified` counts audited records, including documented numerical failures.
It is not a success count or an accuracy threshold. The console also prints
`successful` for verified cases with finite predictions. Shared-prior workers
normally verify a complete 10-case shard; saved records from an unfinished or
resource-interrupted shard are not yet included in verified statistics.

The default report now selects six independently audited **whole-setting**
repair versions: ECI-OFM Poisson inverse/Rough (200 steps, one mixing iteration,
resample step 5), and FM-OFM Helmholtz inverse ID/Smooth/Rough, NS inverse/Rough,
and Burgers/Rough (gradient clipping threshold 50). All 100 cases in each
setting are replaced together, including cases whose error got worse.
These parameters were chosen after observing failures, not on an independent
held-out validation set. `result_version`, `sampling_parameters_json`,
`selection_note`, and `verification_sources` identify the selected versions.
The original configurations remain in **`metrics_original.csv`**, and
`original_n_failed` retains their failure counts in the current CSV.
Use `--original-only` to report the original configurations in `metrics.csv`.
Collection checks that repaired case records and configurations still match
the independent audit; it refuses a changed or incomplete repair.

At the user's subsequent request, FunDPS has one explicitly scoped exception:
only Poisson forward/Rough sample 6 is replaced. The original run reproduces a
NaN at step 325; the unchanged official solver with observation weight 10000
(previously 20000) completes all 500 steps, with solution relative L2 18.328854%.
The other 99 cases remain byte-for-byte original results. The affected CSV row
explicitly identifies this mixed configuration. The proposed full-setting rerun
was cancelled and its partial diagnostics are excluded from the report.

On 2026-09-22 the user also requested targeted ECI-OFM Poisson inverse repairs
for finite coefficient errors above 1000%: ID samples 6/8/13/15/66/80 and Smooth
samples 12/13/17/55/80 (zero-based). These eleven samples use 800 steps with
mixing reduced from 5 to 1, with the same weights, seeds and 500 observations.
The other 189 predictions and case records are retained. Independent auditing
recomputes saved prediction errors and checks observation locations; selection
requires the full audited eleven-case set. The report explicitly identifies
this test-outlier parameter tuning and mixed configuration. Original extreme
errors remain uncapped in `metrics_original.csv` and their original counts
remain in `original_finite_over_1000_percent`.

As of 2026-09-22, OFM's remaining work is Darcy forward. Server216 runs twelve
bounded sampling queues across its eight GPUs for offsets 60/70/80/90 in each
split (120 cases). Server197 retains offsets below 60 and runs four bounded
recovery queues across two GPUs. The former server197 primary queues were
retired after confirming they had no active GPU samplers: their old 76 GiB
free-memory admission rule blocked behind the recovery processes. The two
in-flight server197 samplers were preserved. Recovery queues resume missing
cases and prioritize nearly complete shards; existing results are retained.
Recovery processes require 30 GiB free at admission, use activation
recomputation and a 12 GiB allocated-memory guard. No server193 job is used.
Server197 results live under
`/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ofm_sampling_20260921`.
Thirty Darcy-forward shards were initially reserved centrally, preserving 18
previously successful cases. Twelve unstarted shards were reassigned to
server216; the server197 assignment now contains eighteen shards, including
completed ones. The local tmux session `ddis_ofm_sync197_20260921` returns
verified complete shards to server216; the single summary command remains the
same. Raw interrupted shards are archived before publication, not overwritten.
To run one manual synchronization after a local restart:

```bash
python3 ~/C01Python/DDIS_comparison_20260919/adapters/sync_ofm_server197.py \
  --cache ~/C01Python/DDIS_comparison_20260919/reports/resource_reallocation_20260921/server197_cache --once
```

Some Darcy-forward trajectories exceed an exclusive 80GB GPU. Server197 and
the explicit server216 recovery queue therefore enable PyTorch non-reentrant
activation recomputation around the
unchanged official FNO forward. Model outputs and nested gradients were verified
bitwise equal; two native one-step sampler probes were also bitwise equal.
Probe peak allocated memory decreased from 15,072,269,312 to 1,597,429,248 bytes.
The complete 100-step sampler, ODE tolerances, random seeds, observation masks,
and learned weights are unchanged. `memory_runtime.json` records this runtime
policy; this preflight measurement is not a bound on all future trajectories.

`complete settings` counts settings with all 100 cases audited, not settings
with some saved predictions. The 27 settings contain 270 ten-case shards;
the regular queue advances across settings, so many partial settings can
coexist with zero complete settings. The console now also shows verified
shards and unresolved errors split into recovering, queued, retry-failed and
unassigned/stale. `shard_recovery.csv` lists each unresolved shard, its host,
process and latest available sampling step. A recovery heartbeat older than
180 seconds is stale. Error records are archived only after the full shard is
independently verified; server197 results additionally require publication to
server216 before entering central verified counts.

An SSH/read/validation failure exits nonzero instead of silently using stale
data. Existing CSVs are replaced only after collection and validation succeed.
To reproduce a saved snapshot offline, add
`--snapshot ~/C01Python/DDIS_comparison_20260919/reports/five_method_comparison_20260921/latest_source_snapshot.json.gz`.
Use `--paper`, `--output`, `--host`, `--ssh-control-path` or `--local-root` to
override paths/access. The older `reports/.../analyze.py` reproduces only the
original frozen five-method snapshot; use the new command for current results.

## FM-OFM guidance-strength confirmation (2026-09-22)

`adapters/fm_ofm_strength_study.py` reuses verified guidance-screening results,
freezes one strength-only candidate for each of nine PDE/task combinations,
and compares it with the deployed baseline on validation IDs 64–79. These
IDs were excluded from the earlier tuning IDs 0–7 and checks 16–23. A candidate
must have no numerical failures and improve the confirmation mean by at least
1%; otherwise the baseline is retained. The chosen configuration is frozen
before evaluating ID/Smooth/Rough, 100 cases each. All settings use 100 sampler
steps and 500 observations; official sampling and model code are unchanged.
Original experiment results and the primary `metrics.csv` are retained.

Confirmation also rejects finite errors above 1000%. The initial Helmholtz
inverse clip-200 configuration failed this check on IDs 64–79. Its confirmation
and partial test files are preserved under `rejected_clip200_confirmation` and
excluded from the comparison. A second validation round on fresh IDs 80–95
used the previously audited clip 50 for both baseline and candidate; stronger
observation guidance did not improve it, so the stable clip-50 baseline was
retained. Test errors were not used to choose these parameters.

Server216 sessions `ddis_fm_ofm_strength_gpu0_20260922` through
`ddis_fm_ofm_strength_gpu7_20260922` wait for at least 32 GiB free GPU memory,
GPU utilization no greater than 65%, and CPU load below 110 before admitting
each sampling job. They do not preempt existing samplers. This study does not
use server193 or the heavily loaded server197 CPU.

Refresh the separate comparison from the workstation with:

```bash
python3 ~/C01Python/DDIS_comparison_20260919/adapters/fm_ofm_strength_study.py --fetch
```

This writes `reports/fm_ofm_guidance_20260922/comparison.csv` and
`comparison.json`. Incomplete settings have blank error means; failures are
retained. `reaches_fm_fm_mean` compares the complete 100-case tuned mean with
the manuscript mean, not a claim of statistical equivalence. Configuration,
confirmation evidence, predictions and logs live under server216's main
results root in `diagnostics/fm_ofm_strength_confirm_20260922/`.
