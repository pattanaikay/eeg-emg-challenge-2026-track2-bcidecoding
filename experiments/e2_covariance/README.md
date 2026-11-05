# E2: does cross-channel covariance add useful information?

The hypothesis is that relationships between electrodes help distinguish the three commands beyond channel power alone. A diagonal control keeps channel variances and removes cross-channel relationships before estimating shrinkage. Good synthetic execution is not evidence for this hypothesis on EEG.

## Exact representation and choices

Both variants reuse `FeatureDecoder` and `shrinkage_covariances` from root `submission.py`. Each window is centered across time and converted into a float64, OAS-shrunk covariance with a positive diagonal floor. The diagonal variant zeros off-diagonals before computing the OAS shrinkage moments; its shrinkage intensity therefore cannot encode cross-channel relationships. There is one covariance per supplied window; E2 adds no frequency bank or extra filtering to the official input.

- **Full tangent:** fit source reference `R` as the arithmetic mean of the fitting-window covariances. For each window covariance `S`, calculate `log(R^-1/2 S R^-1/2)` and vectorize the upper triangle, multiplying off-diagonals by sqrt(2). This gives 861 features for 41 channels. The reference is arithmetic, not a fitted affine-invariant geometric mean.
- **Diagonal control:** keep only centered sample variances, apply the shared shrinkage/floor math to that diagonal matrix, then take the natural logarithm of its diagonal: 41 variance-only features. Independently permuting the time samples of each channel leaves these features unchanged. It has no target-derived reference.
- Both fit a feature StandardScaler and the same head on fitting windows only. Primary head: L2 logistic regression with `C=1`. Selectable head ablation: shrinkage LDA, through the shared fitter. Heads use class order `0` imagery, `1` subtraction, `2` word generation.

| Config | Representation | Head |
|---|---|---|
| `configs/tangent_logistic.json` | Full tangent | L2 logistic |
| `configs/diagonal_logistic.json` | Diagonal | L2 logistic |
| `configs/tangent_lda.json` | Full tangent | Shrinkage LDA |
| `configs/diagonal_lda.json` | Diagonal | Shrinkage LDA |

Instantiation at `(41, 480)` confirmed 2,586 parameters for full tangent features and 126 for diagonal features (`3 * feature_count + 3`); fitted reference/scaling tensors are buffers, not additional trainable parameters. The shared runner also records instantiated counts. No runtime estimate is presented as a measurement.

## What is fitted, saved and reset?

`training.py` receives only the relevant fitting partition. It fits the source reference for the full variant, then calls `track2.experiments.fit_head` for scaling and the head. Validation can select explicit settings; outer evaluation must not select them. The shared split policy reserves P10 sessions 2-3 before all fitting, and records membership. P04/P09 are previously viewed diagnostics.

The reference, feature mean/scale and classifier coefficients are saved as tensors in the common checkpoint. The full reference is frozen throughout inference: no batch, current-session or whole-target update (`tsupdate=False` behavior). These presets have no inference-time state. Single-window prediction, independent copies and checkpoint reload all use the same source reference. Do not combine per-window covariance whitening with E2 as an implicit preprocessing change: it can erase the covariance feature being tested.

The coordinator assembles this folder's inference fragment with the shared inference source into one `submission.py`; exported inference does not import this experiment folder or scikit-learn estimators.

## Preparation, training, replay and export

Run from the project root with the Python 3.15 project environment active. On Linux, `data` is the normal data path. On Windows at this checkout's long path, use the extended path documented in the root README in place of `data` for every command. Data preparation is shared and cached; it is not repeated separately for each representation.

```bash
python run.py prepare --data-dir data
python run.py train --config experiments/e2_covariance/configs/tangent_logistic.json --protocol screen --data proteus --data-dir data --device cpu --output runs/e2-tangent-logistic
python run.py train --config experiments/e2_covariance/configs/diagonal_logistic.json --protocol screen --data proteus --data-dir data --device cpu --output runs/e2-diagonal-logistic
python run.py evaluate --run runs/e2-tangent-logistic --data proteus --data-dir data --device cpu --split val
python run.py evaluate --run runs/e2-tangent-logistic --data proteus --data-dir data --device cpu --split test
python run.py evaluate --run runs/e2-tangent-logistic --data proteus --data-dir data --device cpu --split longitudinal
python run.py package --run runs/e2-tangent-logistic --output artifacts/e2-tangent-logistic.zip
python run.py validate artifacts/e2-tangent-logistic.zip
```

`evaluate` reconstructs from the saved source/manifest/checkpoint without fitting and writes replay predictions and metrics. Repeat evaluation/export with the diagonal run to compare it. To select LDA, use the corresponding `*_lda.json` config and a new output directory. Existing run/ZIP paths are never overwritten.

One fixed outer participant fold, using the same config and independent inner validation:

```bash
python run.py train --config experiments/e2_covariance/configs/tangent_logistic.json --protocol outer --fold 0 --data proteus --data-dir data --device cpu --output runs/e2-tangent-fold0
python run.py evaluate --run runs/e2-tangent-fold0 --data proteus --data-dir data --device cpu --split diagnostic
```

Folds are `0` through `3`; apply the same fold to every control. Each is a separate deliberate command, not an automatic tuning sweep. The saved run determines replay membership. In the screen protocol, `test` contains P04/P09; after outer-fold training, `test` is the held-out pair and `diagnostic` separately contains P04/P09. Label P10 later-session results as known-person transfer only if that run trained on P10 session 1.

Dependency-light execution and focused tests:

```bash
python run.py train --config experiments/e2_covariance/configs/tangent_logistic.json --data simulated --device cpu --output runs/e2-tangent-synthetic
python run.py evaluate --run runs/e2-tangent-synthetic --data simulated --device cpu --split test
python -m pytest experiments/e2_covariance/test_experiment.py -q
```

## Comparisons, metrics and failure criteria

Compare full and diagonal covariance with the same head, tuning opportunities, split and fit allocation. Also refit E1, channel-mean linear and EEGNet controls on matching splits. Old results from the larger training allocation are not matched controls.

Primary metric is official equal participant x session x interface cell-balanced accuracy. Inspect command recalls, per-participant/session results, worst-session harm and paired full-versus-diagonal differences. Reject the spatial-information hypothesis if full covariance cannot beat the matched diagonal control across held-out people, gains are confined to one person, or gains conceal recurring subtraction/word harm. This small cohort may leave the answer inconclusive.

## Dependencies, tests and execution status

Uses existing Torch for inference and existing NumPy/scikit-learn for fitting; no pyRiemann or other new dependency. Real PROTEUS loading uses the project's Python 3.15 NeuralBench stack. CPU is sufficient for this path.

Focused tests cover opposite cross-channel relationships with equal variances, variance-only feature invariance to independent channel sample permutations, centering, positive definiteness on zero/rank-deficient/large-amplitude input, three-class probability ordering, training-only arithmetic reference, reference/weight immutability under a shifted evaluation window, independent copies and Solver checkpoint reload.

**Executed 5 November 2026:** Python 3.12.13, Torch 2.15.0+cpu: `.venv\Scripts\python.exe -m pytest experiments/e2_covariance/test_experiment.py -q --basetemp=.cache/pytest-e2-20261105-strict-diagonal` passed all eight cases in 4.94 seconds after making the diagonal control strictly independent of off-diagonals. Project-local scratch avoids the sandbox's operating-system temp-directory restriction. Parameter-count instantiation also ran. Shared CLI/export end-to-end execution remains coordinator-owned; no real-data E2 fit, benchmark, cloud job or submission has been run for this implementation.

Limitations: covariance discards temporal order and preserves possible artifact structure; source-reference tangent features do not guarantee session invariance. This implementation intentionally uses a single broadband covariance and an arithmetic source reference. It makes no EEG improvement claim.

Coordinator integration, 5 November 2026: this folder's focused tests were included in the passing **88-case Python 3.15.2 suite**, and all of its named presets passed shared synthetic fitting, exact validation/test replay and isolated ZIP validation. See [current status](../../docs/STATUS.md) and the [execution guide](../../docs/TIER1_EXECUTION.md) for environment details and the two required upstream development-wheel pins. Real-data fitting and GPU execution remain unrun.
