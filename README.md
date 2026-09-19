# DDIS and generative baselines on FM4PDE data

## Current checkpoint — 2026-09-20 (supersedes historical status below)

The user authorized minimal data/runtime adapters with official algorithms unchanged, then **removed DiffusionPDE and requested early stopping for every remaining model**. Active scope: DDIS, FunDPS, OFM and ECI on Poisson/Helmholtz. OFM and ECI share one official joint prior per PDE (paper H.3). No formal training or final evaluation has completed at this checkpoint.

- All official model/training/sampling source checkouts remain unchanged. Adapters are versioned separately. `adapters/early_stop.py` supervises child training processes and selects the minimum-validation-loss checkpoint. A plateau means eight validation checks without a 0.5% relative improvement. Small absolute improvements still update the saved best checkpoint. Configuration: `configs/early_stopping.json`.
- DDIS/FunDPS retain the official 10M-image maximum and 5M-image learning-rate warmup. Validation runs every approximately 250K images; early stopping only counts plateau checks after 5M images. It calls the loss function and EMA serialized by the official trainer on all 5,000 held-out validation cases, using fixed noise seed 20260920, no data augmentation and batch 1. The supervisor temporarily suspends only its own trainer while validating. No final test distribution is used for early stopping.
- OFM/ECI retain the 300-epoch maximum, validate every 10 epochs with the official training implementation and enable early stopping after epoch 50. DDIS's FNO surrogate retains the 500-epoch maximum, validates and checkpoints every 10 epochs, and enables early stopping after epoch 100. Both use the official metrics on the separate 5,000-case validation split. Logs call it `test`; the adapter deliberately points that loader to validation data.
- `jobs/{method}_{pde}/early_stopping/` stores policy, official training log, validation history, best checkpoint link/metadata, and completion reason. `training_completed` is written only after a successful supervised run. Sampling must consume `best_checkpoint`, not assume the final numbered checkpoint is best.
- DiffusionPDE had only short resource probes. No full DiffusionPDE training was launched. Its formal launcher branch and 45K-image preprocessing are removed. Existing resource probes are historical diagnostics, not benchmark results.
- Authoritative exports from server197 contain the exact reference 45K/5K membership, normalized float32 values, original IDs and SHA256 hashes. Both exports are complete. The two local relay sessions continue streaming train/validation arrays; the six first-100 ID/Smooth/Rough datasets are already converted on server216. Hidden partial arrays are never used for training. Preprocessing resumes completed test conversions and waits for checksummed train/validation arrays.
- Persistent SSH socket: `/tmp/ddis216-recovered-20260919`. Reuse it. Base environment: PyTorch 2.7.1+cu126 and pinned neuraloperator 2.0.0. Separate flow environment: neuraloperator 0.3.0, torchcfm 1.0.5, torch-harmonics 0.7.2; it imports shared dependencies through a `.pth` file, so the base environment must remain available.
- Resource profiles completed: DDIS batch32 FP32 about 39.76 seconds/1K images; FunDPS batch32 FP32 about 20.08 seconds/1K images. Official FP16 flags fail a dtype assertion in both UNO trainers, so FP32 is retained. Shared flow prior batch100 uses about 23.5GB reserved; FNO surrogate batch32 about 19.5GB reserved. These are short profiles on shared GPUs, not convergence measurements.
- Proposed formal GPU slots remain DDIS Poisson/Helmholtz 0/2, surrogate 3/1, shared flow prior 5/7, and FunDPS candidate pool 4/6. GPU4 is occupied by unrelated jobs; memory gates and experiment locks prevent unsafe starts. Queues were temporarily stopped before any formal job began so early stopping can be installed and tested. Restart only after the supervisor integration test and diffusion validation memory profiles pass.
- Ground-truth PDE residuals and field normalization were checked; common masks match all 100 official DDIS/FunDPS masks. See `provenance/ground-truth-residuals.json` and `provenance/pde-parameters.json`. Official DDIS FNO currently has 34,653,889 parameters with its pinned dependency/config, differing from the paper's approximate table size; preserve the official implementation and disclose this difference.
- Flow sampling adapters passed plumbing probes using resource-only weights. DDIS's top-level generation entry imports missing optional modules (`daps_zero`, `dsg`); a minimal entry adapter should import the required official solver directly. Full-schedule sampling, evaluation scheduling and final result verification remain pending. Do not present resource probes as final results.
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
