# E6 - constrained temporal frequencies

Hypothesis: learning a small bank of valid band-pass cutoffs may transfer better than learning each temporal tap freely. This isolates the frequency parameterization; it contains no GMM, confidence calibration, pseudo-label memory or online parameter update.

## Matched architecture

Both `configs/sinc.json` and `configs/free.json` use this exact path:

`source z-score -> channel-shared temporal filters -> grouped spatial convolution -> BatchNorm -> square -> four-bin average power -> log -> dropout -> three-class linear head`

Defaults: 8 temporal filters of 31 samples, 2 spatial filters per temporal filter, 4 temporal power bins, dropout 0.25. A temporal filter is shared over input channels. The spatial convolution spans all 41 channels. BatchNorm running statistics are learned during training and frozen at prediction. Log power has floor `1e-6`. Both temporal banks use a common L2 kernel normalization with floor `1e-6`; the free model has unrestricted learned tap values, without spectral cutoff constraints.

The two variants start from the same Hamming-tapered Sinc kernels. At 120 Hz, initial lower cutoffs are evenly spaced from 4.2 to 39 Hz and upper cutoffs are 6 Hz higher. The free variant then optimizes every tap independently. The Sinc variant optimizes two raw parameters per filter through bounded sigmoid maps:

- Lower cutoff is between `0.005 * Nyquist` and `0.970 * Nyquist`.
- Bandwidth is at least `0.020 * Nyquist`.
- Upper cutoff is at most `0.990 * Nyquist`.

Thus `0 < lower < upper < 60 Hz` at 120 Hz, including saturated finite raw parameters. `torch.sinc` handles the zero-time sample without a divide-by-zero branch. These are nominal filter cutoffs; a 31-tap windowed filter has finite transition width and is not an ideal rectangular passband.

The downstream layers, tensor shapes, initialization, preprocessing, training budget and checkpoint rule match. **Parameter capacity differs:** same topology is not a parameter-matched comparison. Measured by constructing the default 41-channel/480-sample models with Python 3.12.13 / Torch 2.15.0+cpu:

| Stage | Sinc parameters | Free parameters |
|---|---:|---:|
| Temporal bank | 16 | 248 |
| Spatial convolution | 656 | 656 |
| BatchNorm affine weights | 32 | 32 |
| Three-class head | 195 | 195 |
| **Total trainable** | **899** | **1,131** |

Source-normalizer buffers and BatchNorm running statistics are not trainable parameters. The shared runner writes actual counts to each run's `parameters.json`. Do not interpret equal epoch count as proof of equal realized compute; measure execution for real runs.

## Fit, inference and shared components

The coordinator fits source channel means/variances from the relevant training partition only, excludes P10 sessions 2-3, and uses inner validation for checkpoint selection. All neural parameters and BatchNorm running statistics are saved in the tensor checkpoint. Inference uses the frozen source normalizer and frozen neural model; no evaluation-session state is accumulated by these presets. Reload uses the saved source/config/weights, never new fitting or downloads. Outputs/probabilities follow command order 0 imagery, 1 subtraction, 2 word generation.

The experiment reuses E3 source normalization plus the coordinator's data preparation, split audit, optimizer/selection, official scorer, source assembly, single-window wrapper, checkpoint replay and isolated ZIP validation. Only PyTorch is used by the E6 decoder; no new dependency is added. The full real-data workflow still needs the project's Python 3.15 and NeuralBench environment.

## Commands

Run from the project root in the activated Python 3.15 environment. Install `requirements-training.txt` for the full real-data workflow. Use persistent project/data paths on JarvisLabs. Windows users should follow the root README's extended-length data-path example if the ordinary cache path is too long, supplying the same data path to each command.

```bash
python run.py prepare --data-dir data
python run.py train --config experiments/e6_sinc/configs/sinc.json --data proteus --data-dir data --protocol screen --device cuda --epochs 20 --lr 0.001 --seed 33 --output runs/e6-sinc-seed33
python run.py train --config experiments/e6_sinc/configs/free.json --data proteus --data-dir data --protocol screen --device cuda --epochs 20 --lr 0.001 --seed 33 --output runs/e6-free-seed33
python run.py evaluate --run runs/e6-sinc-seed33 --data proteus --data-dir data --device cuda --split val
python run.py evaluate --run runs/e6-sinc-seed33 --data proteus --data-dir data --device cuda --split test
python run.py evaluate --run runs/e6-sinc-seed33 --data proteus --data-dir data --device cuda --split longitudinal
python run.py package --run runs/e6-sinc-seed33 --output artifacts/e6-sinc-seed33.zip
python run.py validate artifacts/e6-sinc-seed33.zip
```

Repeat replay/export for the free run with its own paths. `evaluate` performs saved-checkpoint inference without fitting. For outer-fold checks, train each variant with identical `--protocol outer --fold 0` (then other predeclared folds) and new output paths. Do not choose cutoffs/configuration/checkpoints from outer or P10 diagnostic scores; refit the shared EEGNet and linear controls on the same allocation. Previously viewed P04/P09 scores are diagnostics.

Synthetic execution check, independently for each variant:

```bash
python run.py train --config experiments/e6_sinc/configs/sinc.json --data simulated --epochs 1 --device cpu --output runs/e6-sinc-synthetic
python run.py evaluate --run runs/e6-sinc-synthetic --data simulated --device cpu --split test
python run.py package --run runs/e6-sinc-synthetic --output artifacts/e6-sinc-synthetic.zip
python -m pytest experiments/e6_sinc/test_experiment.py -q --basetemp .cache/pytest-e6
```

Use the free config and distinct paths for its check. Existing run/ZIP paths are not overwritten.

## Metrics, failure criteria and status

Use official equal participant x session x interface cell-balanced accuracy, each command's recall and participant/session diagnostics. Reject nonfinite output/gradients, invalid cutoffs, reload differences, or a reported gain that depends on unmatched preprocessing/training opportunity. Inspect whether a gain survives participants and cognitive classes. Learned passband plots alone establish no decoding gain. If Sinc appears better, additional capacity/compute controls are needed before attributing the difference solely to its frequency constraint.

**13 focused tests passed** in **4.79 seconds** on Python 3.12.13 / Torch 2.15.0+cpu. Tests cover finite forward/backward, single-window probabilities/class order, optimized and saturated cutoff validity, zero signals/free kernels, identical downstream initialization, strict tensor reload/deepcopy, and invalid configs. This is test-suite runtime, not EEG execution runtime.

Both default configs also completed a one-epoch CPU synthetic fit, exact saved-test-metric replay, and isolated ZIP single-window/deepcopy/frozen-state validation through the shared workflow. These checks used temporary workspace `.cache` run directories and the same settings as the synthetic commands above; no synthetic score is an EEG-performance result. The first packaging attempt encountered a sandbox permission error in the operating-system temp directory; rerunning with Python's `tempfile.tempdir` set to project `.cache` passed for both variants. No real-data run, GPU training or EEG benchmark has been executed by this experiment task.

Coordinator integration, 5 November 2026: this folder's focused tests were included in the passing **88-case Python 3.15.2 suite**, and all of its named presets passed shared synthetic fitting, exact validation/test replay and isolated ZIP validation. See [current status](../../docs/STATUS.md) and the [execution guide](../../docs/TIER1_EXECUTION.md) for environment details and the two required upstream development-wheel pins. Real-data fitting and GPU execution remain unrun.
