# E5: small learned spatial/power models

The hypothesis is that learning spatial power filters improves transfer beyond the statistical representations. Start with the small ShallowConvNet; FBCNet is a selectable fixed-bank alternative. Both code paths are implemented, but no EEG performance gain is established.

## Models and meaningful choices

`inference.py` constructs the maintained **Braindecode 1.8.1** implementations directly; it does not copy their architectures or load external pretrained weights. Their installed interfaces were inspected and exercised locally.

| Config | Exact model at 41 channels / 480 samples / 120 Hz |
|---|---|
| `configs/shallowconvnet.json` | `ShallowFBCSPNet`: 8 learned temporal filters of length 25 samples, 8 spatial filters across all channels, BatchNorm, square, mean pooling over 75 samples with stride 15, safe log, dropout 0.5, and a three-class convolutional head spanning the remaining time positions. |
| `configs/fbcnet.json` | `FBCNet`: four fixed FIR bands (4-8, 8-13, 13-30, 30-45 Hz), 4 learned spatial filters per band, BatchNorm, SiLU, log variance in four 120-sample temporal segments, and a three-class linear head. |

Both configurations use **no additional input normalization**, the same four-second EEG windows and the shared training/checkpoint policy. For the first comparison, keep the explicit 20-epoch/Adam learning-rate 0.001 budget below for both models and the EEGNet control. ShallowConvNet's dropout and FBCNet's internal filter bank are architecture choices, not claimed parameter-matched controls. The shared E3 normalizer can be selected in a deliberate later comparison, but it must be matched across architectures and trained using source windows only.

FBCNet is passed the explicit list of band pairs via its verified `n_bands` argument; an integer would silently divide 4-40 Hz differently. Its bank uses the installed `FilterBankLayer` FIR/zero-phase implementation with MNE automatic transition widths, Hamming `firwin` coefficients and centered FFT convolution within each supplied window. Instantiated FIR lengths are 199, 199, 123 and 53 samples. The band edges match E1's four-band preset; FIR filtering is not numerically identical to E1's Hann-tapered spectral integration. Full current-window filtering is legal under the model-facing causal contract; no future window is read.

Actual instantiated parameter counts:

| Model | Trainable parameters | All registered parameters |
|---|---:|---:|
| Small ShallowConvNet | 3,475 | 3,475 |
| Small FBCNet | 899 | 2,047 |

FBCNet stores another 1,148 fixed FIR/denominator coefficients as parameters with `requires_grad=False`; these are not learned. BatchNorm running statistics are buffers. Report actual counts rather than treating these models as capacity matched. No training runtime or GPU throughput has been measured for this implementation.

## Shared components, fitted state and reload

The shared runner handles data preparation, participant splits/P10 reservation, gradient updates, validation cell-balanced-accuracy checkpoint selection, scoring, saved source, reload and flat-ZIP export. This folder needs no separate `training.py` or training loop.

Training fits network weights and BatchNorm source statistics. Evaluation uses `.eval()`, disabling dropout and freezing BatchNorm updates. The presets have no session-adaptive state. The checkpoint contains the learned tensors, BatchNorm buffers and FBCNet's fixed filter coefficients; the manifest contains the model configuration and ordered input description. The source assembler places this inference fragment in the saved standalone `submission.py`.

Braindecode 1.8.1 applies FBCNet's spatial/head max-norm constraints through functional weight parametrizations. Tests deliberately enlarge the underlying parameters and verify that evaluation changes no stored parameter or buffer. No constraint layer is bypassed and the shared frozen-state export check is retained. Pin the declared Braindecode version when reproducing these checkpoints; another version may use different state keys or inference behavior.

## Preparation, training, test inference and export

Run from the project root with the Python 3.12 environment active and `requirements-training.txt` installed. `python` below means that environment's interpreter. On Windows at this long checkout path, replace `data` with the extended data path documented in the root README; use the same path consistently. Linux/JarvisLabs uses the ordinary `data` path.

```bash
python run.py prepare --data-dir data
python run.py train --config experiments/e5_spatial_power/configs/shallowconvnet.json --protocol screen --data proteus --data-dir data --epochs 20 --lr 0.001 --device cuda --output runs/e5-shallowconvnet
python run.py train --config experiments/e5_spatial_power/configs/fbcnet.json --protocol screen --data proteus --data-dir data --epochs 20 --lr 0.001 --device cuda --output runs/e5-fbcnet
python run.py evaluate --run runs/e5-shallowconvnet --data proteus --data-dir data --device cuda --split val
python run.py evaluate --run runs/e5-shallowconvnet --data proteus --data-dir data --device cuda --split test
python run.py evaluate --run runs/e5-shallowconvnet --data proteus --data-dir data --device cuda --split longitudinal
python run.py package --run runs/e5-shallowconvnet --output artifacts/e5-shallowconvnet.zip
python run.py validate artifacts/e5-shallowconvnet.zip
```

Use `--device cpu` when CUDA is unavailable. The models support CPU; these commands do not launch or provision a GPU instance. Repeat replay/export with the FBCNet run. `evaluate` loads the saved source/manifest/checkpoint without fitting and writes predictions/metrics in `replay/`. P10 sessions 2 and 3 each start from the saved starting model. Existing output folders/ZIPs are rejected.

One fixed outer fold, with checkpoint selection restricted to that fold's inner validation people:

```bash
python run.py train --config experiments/e5_spatial_power/configs/shallowconvnet.json --protocol outer --fold 0 --data proteus --data-dir data --epochs 20 --lr 0.001 --device cuda --output runs/e5-shallow-fold0
python run.py evaluate --run runs/e5-shallow-fold0 --data proteus --data-dir data --device cuda --split diagnostic
```

Folds are `0` through `3`; run controls on the identical fold and saved allocation. In screening, `test` is P04/P09; in an outer-fold run, `test` is the held-out pair and `diagnostic` gives the separate previously viewed P04/P09 results. A P10 later-session result is known-person transfer only when the run trained on P10 session 1. Compare freshly fitted EEGNet and the selected statistical model at the same splits and tuning opportunities. The old EEGNet scores from the larger training allocation are not matched controls.

Synthetic workflow and focused checks:

```bash
python run.py train --config experiments/e5_spatial_power/configs/shallowconvnet.json --data simulated --epochs 1 --device cpu --output runs/e5-shallow-synthetic
python run.py evaluate --run runs/e5-shallow-synthetic --data simulated --device cpu --split test
python run.py train --config experiments/e5_spatial_power/configs/fbcnet.json --data simulated --epochs 1 --device cpu --output runs/e5-fbcnet-synthetic
python run.py evaluate --run runs/e5-fbcnet-synthetic --data simulated --device cpu --split test
python -m pytest experiments/e5_spatial_power/test_experiment.py -q
```

## Metrics, failure criteria and status

Use official equal participant x session x interface cell-balanced accuracy, command recalls and participant/session diagnostics. Reject a gain explained by extra training, a single participant, or recurring cognitive-command harm. A small model fitting the synthetic fixture does not establish EEG decoding or participant transfer.

**Executed 5 November 2026:** Python 3.12.13, Torch 2.15.0+cpu, Braindecode 1.8.1: `.venv\Scripts\python.exe -m pytest experiments/e5_spatial_power/test_experiment.py -q --basetemp=.cache/pytest-e5-20261105-a` passed all six cases in 8.02 seconds. Checks cover a finite backward/optimizer step for both maintained models, single-window output/probability contracts, exact Solver reload/deep-copy probability parity, zero/large input stability, physical frequency response of all four bands, fixed coefficients, invalid widths/bands and immutable max-norm evaluation. Two unrelated Braindecode import warnings concern MNE montage-name deprecation. Parameter-count/filter-length instantiation also ran.

No new dependency was added. Full shared CLI/export integration remains coordinator-owned; no real EEG fit, participant benchmark, GPU execution, cloud job or submission ran in this worker. FIR edge effects, nuisance power, cognitive-task timing loss and the small public cohort remain limitations. There is no improved-decoding claim.

Coordinator integration, 5 November 2026: this folder's focused tests were included in the passing **88-case Python 3.15.2 suite**, and all of its named presets passed shared synthetic fitting, exact validation/test replay and isolated ZIP validation. See [current status](../../docs/STATUS.md) and the [execution guide](../../docs/TIER1_EXECUTION.md) for environment details and the two required upstream development-wheel pins. Real-data fitting and GPU execution remain unrun.
