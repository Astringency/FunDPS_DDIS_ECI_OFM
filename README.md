# DDIS and generative baselines on FM4PDE data

## Current checkpoint — 2026-09-20 (supersedes historical status below)

The user explicitly authorized **minimal data and runtime adapters while keeping algorithms official**. The earlier adapter question is resolved; do not ask again. No formal training or final evaluation has completed.

- `adapters/prepare_data.py` uses official DDIS loaders/normalization and the unmodified FM4PDE split function from commit `ec9e658` (copied with provenance). Both June reference checkpoints have 45,000 training / 5,000 validation samples and seed 0. Historical source confirms PDE label offsets 1009/2018 and sorted split indices. Export stores the same normalized float32 values consumed by the diffusion trainers, explicit original sample IDs, and source/output SHA256 hashes.
- Both PDE exports completed on server197 under `/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/ddis_comparison_20260919/export/`. Each contains train, validation, and first-100 ID/Smooth/Rough arrays. Setup logs are in the adjacent `setup/` directory. Initial export launch failed because the bundle branch was not checked out; `export_*_r2.*` is the corrected attempt.
- Original raw-data pull/push sessions were intentionally stopped after compact exports became available. Their partial/raw files remain for later cleanup and must not be used for formal training. Local `ddis_relay_{helmholtz,poisson}_20260920` stream compact arrays through the workstation to server216 `data/compact/{pde}/`, retaining hidden partial files until SHA256 matches. Local logs: `provenance/relay_{pde}.*`. This replaces the former recovery controller. `adapters/relay_data.sh` supports resuming at the verified partial byte count.
- Persistent SSH control socket: `/tmp/ddis216-recovered-20260919`. Reuse it for server216. Avoid many fresh simultaneous SSH connections.
- DDIS/FunDPS environment installation succeeded (PyTorch 2.7.1, CUDA 12.6, pinned neuraloperator 2.0.0, datasets 3.6.0). A separate `venv-flow` is installing upstream neuraloperator 0.3.0 and flow dependencies; it shares read-only base dependency imports through a `.pth` file. Never remove the base venv while flow relies on it. Setup session `ddis_flow_env_20260920`, logs/exit `setup/flow-environment.*`.
- Resource-only legacy data at `data/resource-pilot/` are deliberately separate from authoritative compact exports. DDIS microbatch 1 ran for the 300-second profile limit (exit 124); microbatch 4 completed the 1,000-image diagnostic (exit 0). FunDPS microbatch 1 hit the same profile time limit (exit 124). Observed GPU totals were approximately 13 GB, 17 GB, and 8 GB, respectively, including existing unrelated jobs. Inspect `profiles/` logs and memory CSVs before scaling. These runs are **not** formal training; never reuse their weights.
- Paper Appendix H.3 explicitly says OFM regression and ECI sampling share the same pretrained OFM joint prior. Train one official OFM prior per PDE, then apply both official samplers. `adapters/train_flow.py` calls the unchanged `OFMModel.train`, official two-channel FNO and GP prior; it only provides data, CLI and an import-path alias. It preserves 300 epochs, batch 100, Adam 1e-3, StepLR(50, .8), Matérn(.01, 1, .5), sigma_min 1e-4, modes32/width128/projection128.
- Remaining work: validate exported values and memberships, finish compact transfer and HF conversion, finish memory profiles (including surrogate, flow and DiffusionPDE), launch full training with safe batches, complete official sampling adapters and all 2×3×100 evaluations per method, verify and archive results. The goal remains active.

Flow protocol source: [DDIS Appendix H.3](https://arxiv.org/html/2601.23280v3#A8.SS3).

## Requested outcome

Train the official DDIS and generative baselines from arXiv:2601.23280 on server216, using FM4PDE data for two PDEs, then sample and evaluate 100 test cases on each of ID, Smooth and Rough for comparison with FM4PDE. Preserve official model, training and sampling implementations. Allocate GPUs concurrently after checking live jobs and measuring memory with small batches.

Working PDE selection: Poisson and Helmholtz, both supported by the paper and FM4PDE. Working task: inverse reconstruction from 500 noiseless solution observations, matching the FM4PDE inverse configs. Target methods: DDIS, FunDPS (called FuncDPS in the request), OFM, ECI, and DiffusionPDE. The scope is **not complete** until all required training and evaluation results are verified.

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
