# Verification status

This file records evidence, not planned results. Initial implementation: 20 October 2026.

| Requirement | Current evidence |
|---|---|
| First warm-up package | `artifacts/warmup_constant.zip` built and exported-module validation passed |
| Official synthetic kit | Pinned official `starter.yml` completed with the upstream Constant solver |
| Unit/contract tests | All 10 tests passed in the deployed project with Python 3.12; includes upstream score parity, stream boundaries, channel order, incomplete-run rejection and isolated ZIP imports |
| Linear optimization/export/replay | Full offline synthetic smoke test passed; saved checkpoint replay produced identical metrics |
| Full EEGNet path | Explicit `--include-eegnet` synthetic optimization/checkpoint/replay/export smoke passed on Python 3.12 CPU |
| CUDA path | One-epoch synthetic linear run completed on the local RTX 4050; not an EEG score |
| Shell scripts | Both JarvisLabs scripts passed Bash syntax checks; remote execution remains separate |
| Real PROTEUS audit | Download complete; audit found 10 participants, 14 sessions, 112 EDF files, one 41-channel order, 500 Hz, microvolts, mastoid M1 reference, and 2,655 command events |
| Real preprocessing | Pinned official preparation passed: 1,521 train / 379 validation / 755 test windows, each 41 × 480 at 120 Hz; participant-disjointness check passed |
| Real linear training | Seed 33, 30 epochs, CPU: validation selected epoch 2 at **0.4458613**; held-out public test cell mean **0.3796610** on 755 windows / 8 cells |
| Real linear replay/export | Saved-checkpoint test metrics exactly matched the original JSON; `artifacts/linear-seed33.zip` passed isolated single-window/deep-copy/frozen-state validation |
| Official source preservation | All 30 vendored official files match the pinned upstream checkout byte-for-byte; generated Benchopt caches excluded |
| JarvisLabs deployment | Scripts/documentation prepared; no authenticated local account or remote training run |
| Codabench submission | User confirmed successful warm-up submission and score **0.33** on 20 October 2026; not independently read back from the platform |

Synthetic scores are contract evidence only. They are not performance estimates for real EEG. Final trained runs are proven complete by `completed.json` plus a successful inference replay and ZIP validation. A model trained on synthetic data must not be presented as a real-data baseline.

The initial real-data preparation encountered a 263-character Windows cache filename. The parent existed; the ordinary path failed when pandas opened the cache file. Preparation passed with the Windows extended-length data path documented in the README. The official filtering/extraction code was not changed.

## Reproduce and interpret the measured baseline

`runs/linear-seed33/` contains the complete run, selected checkpoint, exact inference source, split inventory and replay. Settings were the documented defaults: channel-mean linear model, Adam learning rate 0.01, batch size 32, 30 epochs, seed 33. Only validation selected the checkpoint; no adaptation was used.

| Split | Participants | Windows |
|---|---|---:|
| Train | P01, P02, P03, P06, P07, P10 | 1,521 |
| Validation | P05, P08 | 379 |
| Test | P04, P09 | 755 |

This is one public split and one seed, not a competition placement estimate or evidence of successful harmful-drift prediction. EEGNet has passed the synthetic workflow check; real-data EEGNet training and all JarvisLabs execution remain to be done on the cluster.

| ZIP | SHA-256 |
|---|---|
| `artifacts/warmup_constant.zip` | `1ddb08722d40aaedcc885fe782afe8672a707fd0ad39244b3ee9a8b903e8c1ab` |
| `artifacts/linear-seed33.zip` | `525431036c1b1567df2fc39d9770082b91a3a45d6e36377363920137043d9088` |

The constant ZIP's successful platform result is user-confirmed. The trained linear ZIP is locally validated and has not been submitted. No GPU instance was provisioned, and no repository commit or push was performed.
