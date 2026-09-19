# DDIS and generative baselines on FM4PDE data

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

The first resource check found eight A800 80GB GPUs, 920GiB available host RAM and 128 logical CPUs. GPUs 0 and 4 were temporarily idle; existing FM4PDE queues target both. Do not treat this snapshot as a reservation: inspect current GPU processes and queues again immediately before any pilot or training launch. Preserve all other jobs and sessions.

## Compatibility findings and pending clarification

An asynchronous question asks whether minimal data-loading/configuration/evaluation adapters are permitted by “不要自己编写代码”, while retaining all official algorithms. No answer has been received at this checkpoint. Do not treat elapsed time as approval. DDIS/FunDPS preparation can proceed using their official preprocessing and configurable entry points.

- DDIS and FunDPS provide official MATLAB-to-Hugging-Face conversion scripts and Poisson/Helmholtz training/generation entry points. Raw fields match FM4PDE: `f_data` with `phi_data` or `psi_data`. The converter uses fixed published normalization statistics and names raw files differently; path symlinks can accommodate this without model edits.
- DDIS FNO surrogate training hard-codes dataset paths relative to its checkout. Its YAML controls output path, batch size, epochs and checkpoints. Ensure its validation data does not accidentally substitute formal test results for an independent validation split.
- ECI's registry supports NS, Darcy, Stokes, diffusion, Stefan and PME; it lacks Poisson/Helmholtz loaders. Its Darcy loader reads only solution fields and is not a joint coefficient/solution inverse solver. Do not relabel a solution-only experiment as the requested baseline.
- OFM provides notebook entry examples and reusable `OFMModel.train`; no turnkey FM4PDE inverse evaluation CLI exists. `util/true_gaussian_process.py` imports `ofm_utils.util`, although the checkout directory is named `util`. Investigate an upstream-compatible namespace setup before modifying source.
- DDIS paper Appendix H.3 extends flow baselines to a two-channel coefficient/solution prior. Its public DDIS repository has no OFM/ECI pipeline. Paper flow settings: 300 epochs, batch 100, Fourier modes 32, hidden/projection width 128, Matérn lengthscale .01, variance 1, nu .5, sigma_min 1e-4. Do not claim the released ECI examples already reproduce this extension.
- Official DiffusionPDE `merge_data.py` is hard-coded for Darcy. Poisson/Helmholtz preprocessing requires adapting only its field names/scales or finding an official supported equivalent.

## Data equivalence still being investigated

All ten server216 `/data0/zhangxf/PDEdata` training files differ in whole-file SHA256 from the same names on server197. This does not by itself prove numerical array differences. A read-only numerical comparison of the first Poisson file is underway; use exact array checks to decide whether existing raw data can be reused. Until verified, rsync is fetching the authoritative files.

Server216 Poisson shard 1 first 100 array hashes (float64 contiguous bytes):
- `f_data`: `cee6f37f5d87a96dcd580e1e46aa2b45605333ea68e9aca2f290c46a47de4e83`
- `phi_data`: `ac3010c6da4b47bb32ec1ea53175c133b7f0b637ddcd89175a19f90ee8bc4c8a`

FM4PDE's current training loader makes a 45,000/5,000 split from the five shards using a seeded `torch.randperm`, with PDE-specific seed offset. Resolve checkpoint-specific membership before claiming identical training/validation splits. Keep evaluation cases, observations and physical-unit metrics inspectable. Initial proposed evaluation IDs are 0–99 in each distribution; the requested 100 are test cases, not 100 posterior draws per case.

## Next actions

1. Revalidate setup, transfers, source array inspection and any user answer; finish environment installation and dependency import checks.
2. Verify exact data sources, sync data to server216, and use official preprocessing. Record normalization and sample/split membership.
3. Configure official training with absolute output directories and fixed seeds. Preserve full documented training budgets; memory pilots are not completed training.
4. Profile small batches, then allocate free GPU capacity across independent model/PDE jobs in identifiable tmux sessions. Keep logs, config, exit status and Git revisions.
5. Finish training and run ID/Smooth/Rough evaluation on the same 100 cases and observations per PDE. Report physical-unit relative errors, per-case results, time, finite-output checks and protocol differences.
6. Verify all method/PDE/distribution results and checkpoints before claiming completion. Clean only this task's temporary files and completed sessions, after preserving reproducibility in the main result directory.
