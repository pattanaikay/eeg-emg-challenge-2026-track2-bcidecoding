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
| JarvisLabs deployment | 21 October: one on-demand L4 batch completed linear/33 and EEGNet/33,34,35, exact validation/test replay and remote ZIP validation; downloaded checksums matched; machine 542193 verified Paused. See [run log](JARVISLABS_RUN_LOG.md). |
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

This initial linear result is one public split and one seed, not a competition placement estimate or evidence of successful harmful-drift prediction. The subsequent [21 October GPU batch](JARVISLABS_RUN_LOG.md) completed real-data EEGNet seeds 33/34/35 on the same split; it includes full replay, export and retrieval evidence. Local inference revalidation of those downloaded ZIPs remains unavailable due to missing local dependencies; their remote inference validation and local byte/checksum checks passed.

| ZIP | SHA-256 |
|---|---|
| `artifacts/warmup_constant.zip` | `1ddb08722d40aaedcc885fe782afe8672a707fd0ad39244b3ee9a8b903e8c1ab` |
| `artifacts/linear-seed33.zip` | `525431036c1b1567df2fc39d9770082b91a3a45d6e36377363920137043d9088` |

The constant ZIP's successful platform result is user-confirmed. The original trained linear ZIP is locally validated and has not been submitted. The initial implementation did not provision a GPU; the later JarvisLabs batch used a user-created instance and paused it after retrieval. No repository commit or push was performed.



## Next fixed-corpus batch

The roadmap in NEXT_EXPERIMENTS.md is proposed work; none of these experiments has run. The new batch reserves P10 sessions 2–3 (385 windows) before fitting: 1,136 train, 379 validation, and 385 longitudinal diagnostic windows. The CPU phase compares a fresh linear control, four-band log power plus its separate 1–4 Hz ablation, and diagonal/full shrinkage covariance. The neural phase compares a freshly trained EEGNet with training-set z-score and current-window robust normalization on the same split. Fitted transforms use train windows only; validation selects neural checkpoints and test/P10 results remain diagnostics. Local code and synthetic replay checks passed, but the real-data batch is not runnable in the installed Python 3.10 environment (the project requires Python 3.12 and its NeuralBench stack). No GPU job or submission was launched.

## 5 November 2026 - all six Tier 1 paths implemented, Python 3.15 verified

This dated entry supersedes the implementation/environment limits in the proposed-batch note above. The user approved parallel implementation and deferred study/quizzes. Three workers implemented E1-E3, then E4-E6, with coordinator-owned shared integration and source review. All six folders, 21 named experiment presets and two matching control configs are present; [index](../experiments/README.md), [interface](../experiments/INTERFACE.md), and [execution guide](TIER1_EXECUTION.md) contain durable details and actual CLI commands.

- E1: four-band log power, logistic/shrinkage-LDA heads, separate 1-4 Hz extension. E2: centered OAS, fixed training arithmetic tangent reference, strict diagonal-only controls and both heads. E3: source/robust/causal z-score/causal EA modes with ten-window source prior, current inclusion, matched source replay and session reset. E4: regularized three-class OVR CSP/FBCSP plus exactly matching unprojected band controls. E5: maintained small ShallowConvNet/FBCNet. E6: matched Sinc/free compact decoder, valid sub-Nyquist cutoffs, no memory/calibration.
- Shared config runner, screen/four fixed outer folds, fit membership, participant/session/command/startup diagnostics, parameter counts, saved-source inference reconstruction and standalone single-Python-file ZIP export are integrated. P10 sessions 2-3 are reserved on every real config run; matching controls use that same protocol. Previously viewed P04/P09 have a separate `diagnostic` split in outer runs. Known-person interpretation is explicitly tagged.
- Review corrected the existing OAS squared-entry averaging divisor; the helper now matches an independent scikit-learn OAS calculation after unbiased covariance scaling and the documented positive floor. The strict diagonal control removes off-diagonal entries before shrinkage estimation. Legacy commands cannot silently ignore an outer-protocol request. Original saved-run source/checkpoints were preserved.
- **Requested runtime verified: Python 3.15.2 in `.venv`, CPU Torch 2.15.0, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1 and Braindecode 1.8.1.** The shell's default Python remains 3.10; explicit project-interpreter commands are documented. `.python-version` selects 3.15. Earlier 3.12 checks are historical evidence only; the old environment was moved to ignored `.cache/venv-py312` rather than overwritten.
- **88 tests passed on Python 3.15.2**, with five upstream deprecation warnings. Coverage includes all six architectures, split/fit isolation, outer-label permutation, class/channel/input contracts, independent copies/reset, causal future-prefix independence, numerical stability, probability/checkpoint parity and isolated exports. Tests first exposed/fixed the pre-existing P10 fixture argument mismatch; workspace scratch resolved sandbox TEMP restrictions.
- **Full synthetic smoke passed under Python 3.15.2:** existing linear and EEGNet optimization/selection/reload/export plus all **23 presets/controls**, each with fitting or one neural epoch, exact validation/test metric and prediction-CSV replay, and isolated flat-ZIP single-window/copy/frozen-parameter checks. These scores and measured synthetic runtimes are not EEG-performance evidence.
- **Real-data loading checked under Python 3.15.2:** screen membership is 1,136/379/755/385, all four outer protocols repartition correctly, and first batches have finite 41x480 inputs and three-class labels. This is loader/split/input verification, not real-data decoder training/evaluation.
- Both JarvisLabs shell scripts passed Bash syntax checks. The first sandboxed syntax attempt was blocked by MSYS initialization; the same read-only checks passed outside that restriction. Neither remote setup nor GPU training was executed.

Evidence: `runs/tier1-local-verification-20261105-py315/{environment.json,pytest.xml,real-data-protocols.json,synthetic-presets.json,smoke.log}`. The initial 3.12 verification is separately retained in `runs/tier1-local-verification-20261105/`. Runtime evidence records instantiated parameter counts and only executed synthetic timings. No paid resources, GPU experiments, real-data fitting or Codabench submissions were launched.

**Python 3.15 compatibility limit:** stable h5py/numcodecs wheels are not yet published for this runtime, so `requirements-python315.txt` pins two upstream scientific-nightly builds. PyYAML uses its supported Python implementation (`PYYAML_FORCE_LIBYAML=0`) to avoid an optional Windows C-extension build. The original dependency install failed on unavailable older wheels/build tools; the updated install and all checks above passed. Replace development pins only after stable 3.15 wheels pass the same checks. GPU and actual worker-image replay remain unexecuted; local export tests used the recorded environment.
