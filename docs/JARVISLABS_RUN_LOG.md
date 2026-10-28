# JarvisLabs real-data batch — 21 October 2026

This log records execution evidence. No Codabench submission, commit, push, or publication is authorized for this batch.

## Account and instance inspection

- Read README.md, docs/JARVISLABS.md, docs/COMPETITION_CONTRACT.md, docs/STATUS.md, requirements*.txt, both Jarvis shell scripts, run.py, submission.py, track2/data.py, track2/workflow.py, track2/packaging.py, and tests/test_contract.py. No applicable AGENTS.md was found in the project or its ancestors.
- Read the JarvisLabs skill at `C:/Users/Pratik/.agents/skills/jarvislabs/SKILL.md` and the active Ponytail skill. The user's interactive-auth requirement overrides the skill's token example; no credentials were requested or copied.
- `jl` was absent from this shell's PATH. Located the existing modern CLI at `C:/Users/Pratik/AppData/Local/Programs/Python/Python311/Scripts/jl.exe`; version **0.2.13**. No Python installation was modified.
- First account access: `jl status --json` (authenticated; currency **INR**). `jl list --json` initially returned `[]`. Inspected `jl --version`, `jl gpus --json`, `jl templates --json`, `jl create --help`, `jl run --help`, `jl run start --help`, `jl upload --help`, and `jl download --help`.
- Live new-instance catalog omitted L4. Asked once about available alternatives; launched none. Catalog showed RTX-PRO6000 containers at INR 207.36/hour in `india-chennai-01` and H200 at INR 377.46/hour in EU1. The catalog's Chennai region differs from the skill/CLI help's IN2/EU1 list.
- User then created **eeg-foundation-challenge-2026**, machine **542193**, and explicitly identified its notebook and SSH endpoint. Live `jl list --json` and `jl get 542193 --json` matched the name and SSH address: **one on-demand L4, IN2, PyTorch template, Running, 100 GB storage**. The agent neither created nor expanded it. Original requested storage was 40 GB; the supplied existing instance has 100 GB.
- First instance query reported two minutes of runtime. Use conservative active-start bound **16:20 UTC**, pause by **18:15 UTC**, five minutes before the conservative two-hour cutoff. Exact platform creation timestamp and L4 hourly tariff are not exposed in the parsed instance response; neither has been invented. First reported session cost was INR 2.11, which is not an hourly tariff.
- `jl exec 542193 -- nvidia-smi` succeeded: NVIDIA L4, 23,034 MiB reported memory, no GPU processes, driver 580.126.20, driver CUDA capability 13.0.
- Persistent mount `/home`: 100 GB, about 753 MB used, about 100 GB available. Remote shell HOME is `/home`, although initial working directory was `/root`; all project work will use explicit persistent paths.
- Template `python3 --version`: **3.10.20**, incompatible with the required 3.12+ neural stack. Compatibility setup is pending; a different interpreter must receive its own CUDA torch.
- A compound `jl exec ... sh -lc ... python3 -c ...` diagnostic failed with exit 1 because Windows argument quoting truncated the embedded Python command. No training ran. Subsequent diagnostics will use uploaded scripts or unambiguous argv.
- Local Windows OpenSSH/scp are present; rsync is absent. WSL reports not installed. Preferred transfer remains one source-only ZIP through `jl upload`.

## Local evidence

Machine-readable handoff: `runs/jarvis/20261021T162400Z/handoff.json`.
Source staging, transfer, setup, training, validation, and final lifecycle evidence will be appended as each is observed.

## Source transfer and compatibility repairs

- Live raw `misc/server_meta` through the installed `jarvislabs.Client` exposed the L4 container tariff in `india-noida-01` (IN2): **INR 41.31/hour**, **INR 82.62 estimated two-hour compute cost**. Storage and taxes are separate. `jl gpus` filters this catalog row out; zero free devices does not prevent access to the user's already-running L4. No new instance was launched.
- Staged current source under `.cache/jarvis/20261021T162400Z/source/`: **58 source files**, including uncommitted documentation and all 30 pinned vendor files. Excluded data, runs, artifacts, credentials, Git metadata, environments, caches, and generated vendor outputs. Shell scripts were normalized to LF in staging. Original and staged SHA-256 values are recorded per file; vendor bytes are identical.
- Windows .NET's default ZIP writer produced backslash names during local inventory inspection. That archive was not transferred. Built and asserted POSIX entry names in `source-posix.zip` instead (60 entries including two manifests).
- Source archive SHA-256: `23b98cadf6faead385d3388f68685ea7985464e3b9dbfc6c7388121845dcf0b1`. `SHA256SUMS` SHA-256: `6fee7252626006ad4c89464bf2d51c2f0a4bd88b5dfad085df85fd6dfdb28dce`.
- `jl upload 542193 .cache/jarvis/20261021T162400Z/source-posix.zip /home/track2-source-20261021T162400Z.zip --json` returned exit 0. An uploaded standard-library extractor asserted archive checksum, safe unique POSIX members, a new destination, complete inventory, all 58 extracted file checksums/sizes, and LF scripts. It passed at **`/home/eeg-track2-20261021T162400Z`**. The archive and manifests persist under `/home`.
- Started a hidden local deadline watchdog (PID 1984), restricted to machine 542193 and the batch's recorded run IDs. It stops tracked jobs and pauses the instance at 18:15 UTC; ordinary completion will pause earlier. Syntax checked its Python source. Handoff is updated immediately after managed launches.
- Managed setup command: `jl run --on 542193 --json --yes -- bash /home/track2-bootstrap-20261021T162400Z.sh`; handle **`r_9e3c85f8`**, local launch timestamp **16:29:11 UTC**, logs `/home/jl-runs/r_9e3c85f8/output.log`. No lifecycle flags were added to `--on`.
- Early status said running, but the log and PID files were absent. Rechecked the same handle and confirmed no setup process had started. `file .../run.sh` proved CRLF line endings in the CLI-generated wrapper. Installed SDK `_write_remote_text` uses Windows text-mode stdin, which converts LF to CRLF. Preserved the original launcher, normalized it remotely, and started that same managed wrapper; no duplicate training job was created. The original CLI-created record remains usable.
- During that early diagnosis, a process listing included a service command line containing the notebook token in the tool transcript. This was an avoidable credential exposure in diagnostic output. The value was not copied into project logs, handoff records, source snapshots or exports; subsequent process inspection avoided command arguments. SSH through `jl` already worked without the password supplied in chat.
- For later launches, `operations/jl_windows.py` calls the installed CLI entry point with only `_write_remote_text` patched in memory to send UTF-8 bytes. No installed SDK files were edited. The repair is transport-only and preserves the CLI's quoting/paths and run tracking.
- Setup now installs persistent Python **3.12.13** under `/home/track2-python-20261021T162400Z/`, with its own `torch==2.11.0` and `torchaudio==2.11.0` from the official CUDA 13.0 wheel index. The existing template Python 3.10 CUDA stack is not inherited across interpreters. Then the unchanged setup script, pytest, prepare, and audit run sequentially.
- Local bundled Python 3.12 currently lacks torch; downloaded-ZIP runtime validation is not yet available there. Remote validation remains required. No synthetic score will be reported as real-data performance.

## Setup and real-data audit

- Setup handle `r_9e3c85f8` reached **succeeded, exit 0**. The unchanged setup script's doctor and full synthetic EEGNet smoke passed; the subsequent `python -m pytest tests -q` reported **10 passed in 2.89 seconds**. `setup.exit`, `pytest.exit`, `prepare.exit`, `audit.exit`, and `bootstrap.exit` are zero under remote `evidence/`.
- The unchanged requirements installation resolved the project venv to **torch 2.12.1, CUDA build 13.0**, replacing the initially installed inherited torch 2.11.0 in that venv. CUDA is available and reports NVIDIA L4. This is a CUDA build, not a CPU fallback. Actual final versions, rather than the bootstrap's starting version, are recorded in `environment.json`, `evidence/doctor.json`, and `evidence/pip-freeze.txt`.
- Python **3.12.13**, NumPy **2.2.6**, scikit-learn **1.7.2**, benchopt **1.10.0**, neuralbench/neuralset/neuralfetch/neuraltrain **0.4.0**, Braindecode **1.8.1**, MNE **1.13.2**, MOABB **1.7.2**, exca **0.5.29**. The distribution is `nemar-py==0.3.3`; the existing doctor queries `nemar`, so its null field is a package-name mismatch, not proof the downloader is absent.
- Public download used nemar-py and the public nm000290 release, about 1.48 GiB. Audit found **10 participants, 14 participant-sessions, 112 runs**, 885 motor-imagery / 886 subtraction / 884 word events (2,655 total), one raw 41-channel order, 500 Hz recordings, microvolts, and mastoid M1 reference.
- Explicit pretraining split audit passed using the unchanged official loader: train **P01/P02/P03/P06/P07/P10**, validation **P05/P08**, test **P04/P09**; **1,521 / 379 / 755 windows**. Model-facing input is **41 × 480, 120 Hz, three classes**. Participant identities in CSV retain the upstream `Dreyer2026Proteus/` prefix. Only the audit summary strips that prefix for display.
- The unchanged tests/smoke check official metric parity, chronological single-window inference, shared state within a session, and fresh state for each session. No sealed data were loaded or used for fitting.
- Local Python 3.11 has torch 2.5.1+cu121 but lacks benchopt/Braindecode and the EEG stack. The new bundled Python 3.12 also lacks torch. No unrelated Python installation has been modified to force a local check.

## Real-data training

Every job explicitly activates the project `.venv` and invokes the unchanged `scripts/train_jarvis.sh MODEL SEED`. An additional validation-split replay and assert-based evidence check follows the script's test replay and ZIP validation. No architecture, hyperparameter, split, scorer, or inference reset behavior was changed.

- Linear seed 33: managed handle **`r_d1765af0`**, local launch timestamp **16:36:32 UTC**. `jl_windows.py run --on 542193 --json --yes -- bash /home/track2-train-one-20261021T162400Z.sh linear 33`. Initial logs showed all 30 epochs with finite loss and real public-test cell mean **0.3796610431343468**. Replay/export verification was still running at that observation; the training result alone does not mark the whole job complete.
- Linear seed 33 subsequently reached **succeeded, exit 0**, with exact validation/test saved-checkpoint replays and validated flat ZIP. EEGNet seed 33 (**`r_d640ebc3`**) also reached **succeeded, exit 0** before seed 34 (**`r_253894ca`**) was launched. Seed 34 then reached **succeeded, exit 0** before seed 35 (**`r_ccd07fd0`**) was launched. Each gate retained more than ten minutes for work/transfer before the pause deadline; actual remaining seconds are in the handoff.
- Linear settings: **30 epochs, Adam, learning rate 0.01, batch size 32, zero loader workers, initialization seed 33**. EEGNet settings: **20 epochs, Adam, learning rate 0.001, batch size 32, zero loader workers, initialization seeds 33/34/35**. Braindecode EEGNet architecture defaults were unchanged. All runs use CUDA, the same official participant split, and validation cell mean for checkpoint selection. These are initialization seeds, not held-out-participant folds; no seed or setting was selected by optimizing public-test scores.
- Added two-second `nvidia-smi` sampling around EEGNet commands; samples persist in `evidence/gpu-eegnet-seed*.csv`. Linear only has sparse before/after and polling observations. These are observations, not a guaranteed instantaneous memory peak or a speedup benchmark.
- Paired queries show the remote wall clock about **64 seconds behind local UTC**. Remote directory names and script timestamps retain that clock; local launch records and the watchdog use local UTC. `completed.json` uses monotonic elapsed time. This explains why a remote directory timestamp can precede its local CLI launch timestamp.

## Completed real-data results

All four jobs succeeded with exit **0**. For each run, the evidence checker required `completed.json`, training/environment/upstream manifests, inference source, model manifest, weights, complete finite history, selection, split inventory, validation/test metrics, cell breakdowns, predictions, confusion matrices, and both saved-checkpoint replays. Both replay metric JSONs exactly equal their original counterparts. The unchanged packaging command validated each flat ZIP in an isolated process, including single-window inference, deep copying and frozen state.

Scores below are participant × session × interface cell-averaged balanced accuracy on the fixed public development split. They are not Codabench scores. Checkpoints were selected on validation only.

| Model | Seed | Selected epoch | Validation | Public test | Train command seconds | Full pipeline seconds |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 33 | 2 | 0.445861335 | 0.379661043 | 30.792 | 62.538 |
| EEGNet | 33 | 20 | 0.453858647 | 0.352390098 | 30.903 | 67.019 |
| EEGNet | 34 | 7 | 0.454217070 | 0.382836157 | 30.839 | 67.242 |
| EEGNet | 35 | 17 | 0.467058692 | 0.363920946 | 30.483 | 66.267 |

“Train command seconds” is `completed.json.seconds`: it includes loader construction, optimization, validation selection and final scoring, not just GPU kernels. Full pipeline time uses the remote wrapper's start/verification timestamps and includes prepare, training, test replay, packaging and validation replay. No speedup was measured. The two-second GPU samples observed at most **318 MiB** for each EEGNet run, with maximum sampled utilization **29% / 27% / 30%** for seeds 33/34/35. These sampled maxima do not establish instantaneous peaks. Linear's sparse polling observed 3 MiB / 0% after optimization, so a training peak is unavailable.

| Model/seed | Managed run | Remote run folder below project `runs/` | Local CLI launch UTC |
|---|---|---|---|
| Linear/33 | `r_d1765af0` | `linear-seed33-20261021T163528Z` | 16:36:32 |
| EEGNet/33 | `r_d640ebc3` | `eegnet-seed33-20261021T163737Z` | 16:38:41 |
| EEGNet/34 | `r_253894ca` | `eegnet-seed34-20261021T163958Z` | 16:41:02 |
| EEGNet/35 | `r_ccd07fd0` | `eegnet-seed35-20261021T164141Z` | 16:42:45 |

## Retrieval, checksums and final lifecycle

- Exported only this batch's run folders, data audit, environment evidence, GPU samples, operational scripts, source archive/manifests and the five tracked managed-run logs/metadata. No dataset, venv, account configuration or credentials entered the result export. The exporter rechecked all **58 source files**, including **30 vendored official files**, unchanged after execution.
- Remote result archive: `/home/track2-results-20261021T162400Z.zip`, **454,684 bytes**, SHA-256 **`398df48909da0c89481a50e51d228ee71a31c8612709c115022e07931d04510f`**. Remote project remains `/home/eeg-track2-20261021T162400Z`; source archive remains `/home/track2-source-20261021T162400Z.zip`; Python installation and project venv also remain in persistent home.
- `jl exec 542193 --json -- python3 /home/track2-export-batch-20261021T162400Z.py r_9e3c85f8 r_d1765af0 r_d640ebc3 r_253894ca r_ccd07fd0` exited 0. Then `jl download 542193 /home/track2-results-20261021T162400Z.zip runs/jarvis/20261021T162400Z/remote-results.zip --json` exited 0.
- Local standard-library verification matched the archive SHA-256, safely extracted its inventory into a new `runs/jarvis/20261021T162400Z/remote/` directory, and matched **all 169 remote/local file checksums and sizes**. Required all four model/seed pairs. ZIPs were copied into a new `artifacts/jarvis/20261021T162400Z/` directory; every ZIP checksum and every flat member's bytes match the corresponding downloaded run. Existing runs and artifacts were preserved.
- Exact environment listing: [213-package pip freeze](../runs/jarvis/20261021T162400Z/remote/evidence/pip-freeze.txt). It records torch **2.12.1**, torchvision **0.27.1**, and torchaudio **2.11.0+cu130**. The requirements resolver upgraded torch/torchvision while torchaudio remained inherited. All requested imports, CUDA training, smoke, replay and export checks passed; a general `pip check` and unused audio functionality were not tested. Reusing this mixed torch/torchaudio environment for audio work needs a matching-version check.
- **Incomplete check:** local inference revalidation of the downloaded ZIPs was unavailable because the installed local interpreters lack the required dependencies. Local ZIP structure/member-byte/checksum checks passed, and remote inference ZIP validation passed for all four. No unrelated local Python installation was changed.
- After retrieval and checksum success, `jl pause 542193 --yes --json` returned exit **0**, then a live `jl get 542193 --json` verified **Paused** at **2026-10-21 16:45:25 UTC**. This is actual account state, not merely a pause request. The paused response resets runtime/cost fields to zero; those fields are not an invoice or evidence of zero historical cost.
- Conservative active-time upper bound from 16:20 to the verified pause: **25.43 minutes**, comfortably below two hours. At INR 41.31/hour, **INR 17.51 is an estimated upper bound on compute using that time bound**, not an invoice. The authorized two-hour compute estimate was INR 82.62. **100 GB persistent storage continues to incur charges while paused; taxes are separate.** The instance was not destroyed.
- No model remains pending. Nothing was submitted to Codabench, published, committed or pushed.

| Local submission ZIP under `artifacts/jarvis/20261021T162400Z/` | SHA-256 |
|---|---|
| [linear-seed33-20261021T163528Z.zip](../artifacts/jarvis/20261021T162400Z/linear-seed33-20261021T163528Z.zip) | `060a129fde05467149654e7b749af3835f2ce4026b2e3ed7267adc2723f06b15` |
| [eegnet-seed33-20261021T163737Z.zip](../artifacts/jarvis/20261021T162400Z/eegnet-seed33-20261021T163737Z.zip) | `eb474706f2186d074ef1ae15c6830b8305f094ad50e4ed517fc4797f6c93bf9e` |
| [eegnet-seed34-20261021T163958Z.zip](../artifacts/jarvis/20261021T162400Z/eegnet-seed34-20261021T163958Z.zip) | `19b901379cfc6cba32a923fe0538e1bb47ac7549315bac13c9ba2d0708d14e1b` |
| [eegnet-seed35-20261021T164141Z.zip](../artifacts/jarvis/20261021T162400Z/eegnet-seed35-20261021T164141Z.zip) | `c91ec7811a08471d962962cd02d421ca304e2f82f20cec22ccfbca0022c90518` |

Primary local handoff/evidence: [handoff.json](../runs/jarvis/20261021T162400Z/handoff.json), [verified-results.json](../runs/jarvis/20261021T162400Z/verified-results.json), [export manifest](../runs/jarvis/20261021T162400Z/remote/EXPORT_MANIFEST.json), [final instance query](../runs/jarvis/20261021T162400Z/final-instance.json). Full model directories and logs are under that batch's `remote/runs/` and `remote/managed_runs/`. Operational compatibility scripts are preserved under the batch's `operations/` and remote export's `operations/`.
