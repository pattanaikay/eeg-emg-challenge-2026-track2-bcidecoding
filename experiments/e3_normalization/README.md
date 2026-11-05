# E3 - normalization and causal alignment

Hypothesis: some transfer failure comes from recording amplitude/contact changes. Normalization may reduce that nuisance, but it can also erase task information. Compare each mode separately on the same EEGNet, training windows and checkpoint policy; do not sweep combinations.

## Exact choices

| Config | Transformation and fitted quantities | Inference state |
|---|---|---|
| `configs/eegnet_none.json` | Existing EEGNet, no extra input transform | None |
| `configs/eegnet_source.json` | Per-channel mean and population variance over training windows/samples | Frozen training mean/variance |
| `configs/eegnet_robust.json` | Each current channel's median and `(Q75-Q25)/1.349`, scale floor `1e-6` | None; only current window samples |
| `configs/eegnet_causal_zscore.json` | Training mean/variance initialize an arithmetic running distribution, with prior weight 10 equal-duration windows | Running channel mean/variance and window count |
| `configs/eegnet_causal_ea.json` | Mean of training within-window centered OAS covariances initializes an arithmetic running covariance, prior weight 10 windows | Running covariance and window count |
| `configs/linear_causal_zscore_control.json` | Causal z-score with the existing channel-mean linear decoder | Same state as causal z-score; dependency-light workflow control |

Causal modes **include the current window before transforming it**. Causal z-score merges each window's mean/variance using stable mixture-moment updates. Causal EA merges each current OAS covariance, centers that window, and left-multiplies by the running reference's inverse square root. EA eigenvalues are floored at `max(1e-8, 1e-8 * largest_absolute_eigenvalue)`; per-window OAS already supplies shrinkage. It is Euclidean arithmetic alignment, not a geometric/Riemannian-mean estimator.

At the first prediction the reference is `(10 * source_reference + current_statistic)/11`, with the corresponding mixture-variance correction for z-score. Every participant-session starts from the saved source prior with count 10. State may continue across runs/interfaces in that session; it never carries into another session. No labels or inference-time identifiers enter the normalizer. Full supplied windows are available; future windows are not.

The coordinator fits source quantities from the fitting partition only, then replays that partition in session/run/time order with resets before shuffled neural optimization. Validation and evaluation use normal chronological single-window inference. Persistent buffers save source priors; non-persistent buffers hold transient session state. Reset after reloading a raw state dict; the shared Solver handles starting-state reconstruction. No inference-time gradients, GMM or pseudo-label memory are used.

## Reused components and controls

`EEGNetModel`, `TorchLinear`, `robust_window_normalize`, and `shrinkage_covariances` come from existing shared inference. The coordinator owns shape/channel/class validation, fit membership, source assembly, optimization, validation checkpoint selection, official streaming/scoring, reload and ZIP packaging. No new dependency is required beyond the project training environment.

EEGNet with no input normalization is the fixed control; source z-score is the fitted-source comparison. Use equal splits, seeds, optimizer/epoch budgets and checkpoint opportunities. The linear config checks execution; success on it does not establish an EEGNet normalization gain. All normalizers have **0 trainable parameters**. The 41-channel/three-class linear decoder has **126**; the shared runner records instantiated EEGNet/model counts. No EEG runtime or performance has been measured for these configs.

## Commands

Run from the project root with the project's Python 3.15 environment activated. Install `requirements-training.txt` there. On Windows use the same extended-length `--data-dir` for preparation, training and replay if the ordinary cache path exceeds Windows path limits; see the root README. On JarvisLabs these ordinary `data` paths apply.

```bash
python run.py prepare --data-dir data
python run.py train --config experiments/e3_normalization/configs/eegnet_source.json --data proteus --data-dir data --protocol screen --device cuda --seed 33 --output runs/e3-source-seed33
python run.py evaluate --run runs/e3-source-seed33 --data proteus --data-dir data --device cuda --split val
python run.py evaluate --run runs/e3-source-seed33 --data proteus --data-dir data --device cuda --split test
python run.py evaluate --run runs/e3-source-seed33 --data proteus --data-dir data --device cuda --split longitudinal
python run.py package --run runs/e3-source-seed33 --output artifacts/e3-source-seed33.zip
python run.py validate artifacts/e3-source-seed33.zip
```

`evaluate` reloads the saved inference source, manifest and checkpoint and performs no fitting. Replace the config and use a new output directory for each named comparison. To check an outer fold, train with `--protocol outer --fold 0` and a fresh output; validation chooses the checkpoint inside that fold and test is its outer evaluation. P10 later sessions are a known-person diagnostic only in runs whose fitting partition contains P10 session 1. Existing old scores are not matched controls.

Dependency-light synthetic execution check (not an EEG result):

```bash
python run.py train --config experiments/e3_normalization/configs/linear_causal_zscore_control.json --data simulated --epochs 2 --device cpu --output runs/e3-linear-synthetic
python run.py evaluate --run runs/e3-linear-synthetic --data simulated --device cpu --split test
python run.py package --run runs/e3-linear-synthetic --output artifacts/e3-linear-synthetic.zip
python -m pytest experiments/e3_normalization/test_experiment.py -q
```

## Metrics, failure criteria and status

Rank by the official equal participant x session x interface cell-balanced accuracy; inspect each command's recall, participant/session scores, first-window/first-ten/later errors, and natural class streaks. Reject future-window dependence, cross-session state carryover, nonfinite transforms, startup harm, class-streak collapse, or gains confined to one participant/class. Do not select a mode using outer-test or reserved P10 scores.

Alignment can erase information in covariance features; robust and z-score modes can remove useful amplitude. These E3 presets use EEGNet rather than automatically combining alignment with E2. P10 alone cannot establish population-level longitudinal reliability.

Implemented and focused-tested on Python 3.12.13 / Torch 2.15.0+cpu: `.venv\Scripts\python.exe -m pytest experiments/e3_normalization/test_experiment.py -q` passed **16 tests** in **4.04 seconds**. This is measured test-suite runtime, not EEG training/inference runtime. Tests cover causal prefix/batch independence, current-window inclusion, reset/deepcopy and tensor reload, frozen source statistics, constant/rank-deficient inputs, invalid fitting data/configuration, and linear-wrapper class ordering. End-to-end CLI/replay/export verification remains coordinator-owned. No real-data fitting, GPU job or benchmark was run by this experiment task.

Coordinator integration, 5 November 2026: this folder's focused tests were included in the passing **88-case Python 3.15.2 suite**, and all of its named presets passed shared synthetic fitting, exact validation/test replay and isolated ZIP validation. See [current status](../../docs/STATUS.md) and the [execution guide](../../docs/TIER1_EXECUTION.md) for environment details and the two required upstream development-wheel pins. Real-data fitting and GPU execution remain unrun.
