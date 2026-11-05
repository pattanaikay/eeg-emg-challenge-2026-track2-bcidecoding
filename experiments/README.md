# Tier 1 experiments

All six paths are implemented and reuse one preparation, split, training, official scoring, checkpoint/reload and flat-ZIP harness. Start with the [execution guide](../docs/TIER1_EXECUTION.md) for Windows and JarvisLabs commands. [INTERFACE.md](INTERFACE.md) describes code ownership and interfaces.

| Experiment | Primary preset | Selectable comparison | README |
|---|---|---|---|
| E1 spectral power | `e1_spectral/configs/logistic.json` | Shrinkage LDA; separate 1-4 Hz extension | [E1](e1_spectral/README.md) |
| E2 covariance | `e2_covariance/configs/tangent_logistic.json` | Strict diagonal-only covariance; LDA head | [E2](e2_covariance/README.md) |
| E3 normalization/alignment | `e3_normalization/configs/eegnet_source.json` | None, window robust, causal z-score, causal EA | [E3](e3_normalization/README.md) |
| E4 spatial projection | `e4_csp/configs/csp.json` | FBCSP and exact matching unprojected power controls | [E4](e4_csp/README.md) |
| E5 learned spatial/power | `e5_spatial_power/configs/shallowconvnet.json` | FBCNet | [E5](e5_spatial_power/README.md) |
| E6 temporal filters | `e6_sinc/configs/sinc.json` | Free temporal kernels with the same downstream decoder | [E6](e6_sinc/README.md) |

Paths in the table are relative to this directory; CLI paths start with `experiments/`. Newly fitted controls are `controls/linear.json` and `controls/eegnet.json`. Do not combine `--config` with `--model`. The old `experiment-batch` command is retained for compatibility, but the six-experiment comparison uses these configs.

Recommended real-data execution order: matched controls, E1, E2, E3, then E4, E5 and E6. This orders the scientific comparisons; it does not omit later code paths. Start with a primary preset, then its named control/ablation. No automatic real-data factorial sweep or winner selection is provided.

Every real-data config run automatically reserves P10 sessions 2-3. `--protocol screen` uses 1,136 fitting / 379 validation / 755 previously viewed P04/P09 / 385 longitudinal windows. `--protocol outer --fold 0..3` selects one fixed participant fold with a separate inner validation pair. In outer runs `--split test` is that fold's held-out pair, while `--split diagnostic` is P04/P09. P10 later-session scores are known-person transfer only when `split.json` says `longitudinal_known_person: true`. The eight-development-person fold results are robustness checks, not a pristine independently collected test set.

Configuration and fitted tensors are saved in each new run. The run's assembled `submission.py` is the source used for training and reload; exported inference never imports the experiment directories. Source normalization/reference/projections/scalers/classifiers fit only the fitting partition. Causal normalization includes the current window, uses a ten-window source prior and resets each participant-session; source training is replayed with matching boundaries. Do not combine E2 covariance features with causal whitening that erases them.

Actual instantiated counts at 41 channels x 480 samples:

| Architecture | Trainable tensor parameters | Other fitted/stored quantities |
|---|---:|---|
| Channel-mean linear / EEGNet control | 126 / 2,483 | EEGNet BatchNorm buffers |
| E1 four/five-band head | 495 / 618 | Feature scaler |
| E2 full/diagonal head | 2,586 / 126 | Source covariance reference and feature scaler |
| E3 EEGNet | 2,483 | Source priors; transient session state |
| E4 CSP/FBCSP head | 21 / 75 | 246 / 984 fitted spatial coefficients plus scaler |
| E5 ShallowConvNet / FBCNet | 3,475 / 899 | FBCNet also stores 1,148 fixed filter parameters (2,047 total) |
| E6 Sinc / free kernels | 899 / 1,131 | Same downstream stages; different temporal parameter capacity |

Statistical coefficients are fitted by scikit-learn, not gradient steps; counting them as tensor parameters does not count all fitted buffers as degrees of freedom. Each run records total/trainable counts and buffer scalars in `parameters.json`.

The project uses Python 3.15.2. On 5 November 2026, all 88 tests and all 23 synthetic preset/control fit-reload-export checks passed under this runtime. Initial Python 3.12.13 checks remain separate historical evidence. See [status](../docs/STATUS.md) for the final combined suite and all-preset smoke evidence. All real split protocols and their first input batches were checked locally. Real-data fitting, EEG performance comparisons, GPU execution and worker-image replay remain unexecuted. Synthetic timing in verification artifacts is not an EEG runtime benchmark.
