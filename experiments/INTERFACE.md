# Tier 1 implementation interface

Approved to proceed on 5 November 2026. Coordinator owns shared files; workers edit only their assigned experiment directory. Existing uncommitted work must be preserved.

## Configuration and inference

- A JSON config selects `experiment`: `e1_spectral`, `e2_covariance`, `e3_normalization`, `e4_csp`, `e5_spatial_power`, or `e6_sinc`; `model` selects its variant. Optional fields: `normalization` (default `none`), `bands`, `head` (`logistic` or `lda`), `C`, and small model-specific settings.
- Each experiment `inference.py` defines `build_e1(meta, config)`, etc., returning an `nn.Module` decoder with `forward(X) -> (B,3)` scores. `meta` contains `n_chans`, `n_times`, `n_classes`, `sfreq`, and optional ordered `ch_names`.
- Inference files are source fragments assembled AFTER root `submission.py` into the run's standalone `submission.py`. They may use its `torch`, `nn`, `FeatureDecoder`, `EEGNetModel`, `TorchLinear`, `BANDS`, `band_power_features`, `shrinkage_covariances`, and `robust_window_normalize`. Import other worker-provided packages normally, but no project-local imports or `__future__` statements. Prefix new global names with the experiment ID to avoid collisions.
- The shared wrapper validates input/outputs, owns normalization, exposes `predict` and `predict_proba` in class order 0/1/2, and delegates `reset_state`. Do not duplicate it in each experiment.
- All fitted quantities must be registered tensors so `state_dict` plus the config fully reconstructs prediction. No pickle/joblib estimator at inference. Preserve class ordering. Reject invalid config values.

## Statistical fitting (E1, E2, E4)

- `training.py` defines `fit(model, X, y, config)`, where `model` is the experiment decoder (not the wrapper), X is a finite CPU tensor containing ONLY the relevant fitting windows, and y contains command indices 0/1/2.
- Reuse `track2.experiments.fit_head(model, features, y, config)`: fit a training-only StandardScaler and logistic/shrinkage-LDA head, copying coefficients into `model.linear`, `model.feature_mean`, `model.feature_scale`.
- E1/E2 should reuse/subclass existing `FeatureDecoder` rather than copy spectral/covariance math. Custom bands can be set by the subclass, adjusting scaler/head dimensions at construction.
- E2 reference fitting belongs in training.py; reference never updates at evaluation. E4 stores fitted spatial projections as buffers. Fit receives no validation/outer labels.

## E3 normalization

- Define `InputNormalizer(meta, config)` in inference.py, with `forward(X)` transforming windows; causal modes process each window sequentially, updating exactly once before transforming it. Define `reset_state()` to restore the source prior.
- Modes: `none`, `source-zscore`, `window-robust`, `causal-zscore`, `causal-ea`; no combinations. Fixed prior strength: 10 windows by default. Source quantities are persistent buffers; running session quantities are non-persistent buffers. Include current window, reset each participant-session, shrink/floor numerically.
- `training.py`: `fit_normalizer(normalizer, X, config)` uses fitting data only. `build_e3` returns the base EEGNet decoder (small existing EEGNetModel). Configs also allow `model: linear` for dependency-light normalization smoke tests, clearly labeled as control.
- Coordinator replays training windows chronologically with session resets through the fitted normalizer before shuffled optimization, then resets it. Evaluation uses only normal online calls.

## CLI and tests

The shared CLI is `python run.py train --config <json> --data proteus|simulated --output <new-dir> [--protocol screen|outer --fold 0..3] [existing training flags]`. Config paths and legacy `--model` are alternatives. Experiment configs always reserve P10 later sessions on real data. Existing evaluate/package/validate remain; evaluate supports `--split val|test|longitudinal|diagnostic` and reads the saved protocol/config. `diagnostic` exists only in outer runs and contains P04/P09. Legacy `--model` rejects outer protocol requests rather than silently ignoring them.

Tests can call `track2.experiments.inference_module(config)` to get the assembled module and `module.build_model('experiment', meta, config)` to construct the wrapper; `.decoder` gives your decoder. `fit_head` and inference_module are being implemented by the coordinator. Until ready, report the dependency; do not edit shared files or invent passing checks.

Write README commands for preparation, train, validation/test/longitudinal replay, packaging and validation; specify controls, fit/state boundaries, failure criteria, parameter count reporting, dependencies, actual tests and execution status. Synthetic checks do not measure EEG performance. No real-data sweeps/cloud/submissions in worker tasks.
