# E4 - CSP and filter-bank CSP

Hypothesis: a few spatial projections learned from labeled training EEG can expose task-related power more effectively than measuring each channel separately. Three-class one-vs-rest CSP fits each command against the other two commands; it does not assume this is a binary motor-imagery problem.

## Exact model and matched controls

`csp.json` uses one 4-45 Hz band. `fbcsp.json` uses the same four bands as the primary E1 comparison: 4-8, 8-13, 13-30 and 30-45 Hz. The fixed band operation centers each current window, applies a Hann taper, masks its Fourier bins in the half-open band, then reconstructs a real signal. Its power normalization is the same as E1's integrated FFT power. It uses no other window and no fitted temporal filter.

For each band, fit OAS-regularized covariance per training window using the existing `shrinkage_covariances`. Normalize each covariance by its trace, then average within each command and its rest group. Solve the generalized symmetric eigenproblem for command covariance versus command-plus-rest covariance. Keep the lowest and highest eigenvector per command by default: two components per command, six projections per band. Eigenvector signs are canonicalized for stable saved coefficients. `components_per_class` can select another even number no larger than the channel count; the presets fix two and do not launch a search.

Inference applies the saved spatial projections, takes log10 population variance with a `1e-12` floor, and uses training-only standardization plus L2 logistic regression (`C=1`). The shared shrinkage-LDA head is also supported by an explicitly edited config with `head: lda`; compare it only with an equally configured E1 control.

`csp_control.json` and `fbcsp_control.json` remove the spatial projection and keep identical filtering, bands, scaler fitting and classifier. Their unprojected features numerically match the shared E1 `band_power_features`, including channel-first/band-second ordering. The four-band control therefore matches E1's primary logistic preset, while the broad-band control is its one-band counterpart. These controls isolate the projection's value rather than a change of temporal filtering. Refit the controls and EEGNet/channel-mean linear references on each matching split; historical scores using 1,521 fitting windows are not matched controls.

At 41 channels, instantiated CSP has 21 linear-head parameters and 246 fitted spatial coefficients stored as buffers. FBCSP has 75 head parameters and 984 spatial coefficients. The one-band and four-band unprojected controls have 126 and 495 head parameters. These counts were checked by model instantiation; buffers are not counted as optimizer parameters. Shared run metadata records the instantiated parameter count. No real-data runtime benchmark has been executed.

## Fitting, state and dependencies

Reuse shared preparation, split safeguards, source assembly, tensor head fitting, official scoring, saved replay and flat-ZIP validation. `training.py` receives only the relevant training partition; neither spatial filters, scaler nor head sees validation or outer-evaluation recordings. Saved projection tensors, scaler/head tensors and the fitted flag are sufficient to reload without eigendecomposition or fitting. Calling a CSP model before spatial fitting is rejected. Evaluation has no adaptation state or target-statistic updates.

Dependencies are the already declared Torch, NumPy and SciPy; head fitting uses the existing scikit-learn dependency. Inference needs Torch only beyond the shared worker wrapper. No new package, external weights or separate CSP library is added.

## Commands

Run from the project root in its Python 3.15 environment. Preparation is shared and needed only once. At this long Windows checkout path, use the extended-length `--data-dir` path documented in the root README if ordinary cache paths fail; use the same data path for preparation, training and replay.

```text
python run.py prepare --data-dir data
python run.py train --config experiments/e4_csp/configs/csp.json --protocol screen --data-dir data --device cpu --output runs/tier1-e4-csp
python run.py evaluate --run runs/tier1-e4-csp --data-dir data --device cpu --split val
python run.py evaluate --run runs/tier1-e4-csp --data-dir data --device cpu --split test
python run.py evaluate --run runs/tier1-e4-csp --data-dir data --device cpu --split longitudinal
python run.py package --run runs/tier1-e4-csp --output artifacts/tier1-e4-csp.zip
python run.py validate artifacts/tier1-e4-csp.zip
```

`evaluate` reloads the saved source/config/checkpoint and performs inference without training. For FBCSP or a matched control, replace `csp.json` with `fbcsp.json`, `csp_control.json` or `fbcsp_control.json` and use a distinct run/ZIP name. Existing outputs are rejected. A predeclared participant fold uses `--protocol outer --fold 0` on training; fold indices 1-3 are separate matched runs, not tuning feedback. P10 later-session results are known-person transfer only if that fold trained on P10 session 1.

Synthetic execution and focused checks:

```text
python run.py train --config experiments/e4_csp/configs/csp.json --data simulated --device cpu --output runs/tier1-e4-csp-synthetic
python run.py evaluate --run runs/tier1-e4-csp-synthetic --data simulated --device cpu --split test
python -m pytest experiments/e4_csp/test_experiment.py -q
```

## Interpretation and verification

Primary metric is official equal participant x session x interface balanced accuracy. Inspect command recalls, paired participant changes and worst-session harm. Reject a claimed spatial-filter advantage that disappears against the unprojected matched control or does not survive different held-out participants. P04/P09 are previously viewed diagnostics; their scores must not select settings. A large training score can simply reflect supervised spatial overfitting.

On 5 November 2026, the focused suite passed **12 tests** in the project Python 3.12 environment. It checks exact E1/control feature parity, synthetic spatial discrimination, finite three-class probabilities, unchanged fitted tensors after held-out transformation, single-window and tensor-checkpoint parity, missing-class/unfitted-model rejection, zero/rank-deficient stability and invalid configuration rejection. One initial test-collection import-order failure was fixed within E4 before the passing run. CLI train help and model parameter counts were checked. Both CSP and FBCSP completed synthetic CLI training, byte-identical saved test-prediction replay and isolated exported-ZIP inference validation. The first packaging attempt encountered a sandbox TEMP permission error; validation passed with workspace-local scratch. Real-data training/evaluation and runtime benchmarking remain unexecuted. Synthetic discrimination is execution evidence, not EEG performance.

The FFT mask assumes periodic extension inside the supplied window and is not a claim about real-time sample-by-sample filtering. Only within-window power is modeled, and OVR projections can memorize participant-specific mixtures. Component count, broad-band choice and filter-bank choice are proposed controls, not validated optima.

Coordinator integration, 5 November 2026: this folder's focused tests were included in the passing **88-case Python 3.15.2 suite**, and all of its named presets passed shared synthetic fitting, exact validation/test replay and isolated ZIP validation. See [current status](../../docs/STATUS.md) and the [execution guide](../../docs/TIER1_EXECUTION.md) for environment details and the two required upstream development-wheel pins. Real-data fitting and GPU execution remain unrun.
