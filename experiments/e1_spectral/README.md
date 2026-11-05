# E1 - spectral band power

Hypothesis: how much EEG power lies in each frequency band may transfer across people better than channel means or a small learned decoder. This representation discards event order within the supplied four-second window; success would establish a useful feature comparison, not a physiological mechanism.

## Model and controls

Reuse `submission.py`'s centered, Hann-tapered, one-sided FFT band power. Integrate each half-open frequency band, floor power at `1e-12`, and take log10. Features are ordered channel first, then band. `logistic.json` uses 4-8, 8-13, 13-30 and 30-45 Hz, training-only standardization and L2 multinomial logistic regression with `C=1`. `lda.json` changes only the head to automatic-shrinkage LDA. `low_logistic.json` separately adds 1-4 Hz as a fifth feature per channel; it does not replace the primary four-band comparison.

There are 164 features and 495 fitted head coefficients at 41 channels with four bands; the five-band extension has 205 features and 618 coefficients. These are coefficient counts, not measured training times. The fixed FFT has no trainable parameters. No regularization search or band sweep is launched by these presets.

The matched controls are newly fitted EEGNet and channel-mean linear on exactly the same participant allocation. Compare E4 against an E1 configuration with the same bands and classifier; spectral/filter realization must also be recorded rather than attributing a filtering change solely to CSP. Old scores trained on all 1,521 original training windows are not matched controls.

## Shared components and fitted quantities

The shared CLI, preparation, split guards, official scorer, source assembler and packaging are reused. `training.py` calls `track2.experiments.fit_head`; the scaler and head see only the relevant fitting partition. No held-out data fit the scaler. Their means, scales and linear coefficients are tensors in `weights.pt`, and band/configuration choices are in `model.json`. Reload reconstructs them without fitting. There is no inference-time state, target reference update, pseudo-label memory or cross-session carryover. Head scores/probabilities use class order 0 imagery, 1 subtraction, 2 word generation.

## Execution

Run from the repository root in the project Python 3.15 environment. On Windows at this long checkout path, use the extended-length data path documented in the root README for every `--data-dir` argument if ordinary cache paths fail. Preparation is shared and needed only once.

```text
python run.py prepare --data-dir data
python run.py train --config experiments/e1_spectral/configs/logistic.json --protocol screen --data-dir data --device cpu --output runs/tier1-e1-logistic
python run.py evaluate --run runs/tier1-e1-logistic --data-dir data --device cpu --split val
python run.py evaluate --run runs/tier1-e1-logistic --data-dir data --device cpu --split test
python run.py evaluate --run runs/tier1-e1-logistic --data-dir data --device cpu --split longitudinal
python run.py package --run runs/tier1-e1-logistic --output artifacts/tier1-e1-logistic.zip
python run.py validate artifacts/tier1-e1-logistic.zip
```

`evaluate` is the saved-checkpoint reload and inference path, not retraining. For the head ablation or low-band extension, change the config to `lda.json` or `low_logistic.json` and use a new output/run name. Existing run directories and ZIPs are not overwritten. For one predeclared outer fold use `--protocol outer --fold 0` on `train`; folds 1-3 are separate matched runs, not a command to tune against their outer results. The config path selects E1 without another `--model` flag.

Synthetic CLI check (execution only, not EEG accuracy):

```text
python run.py train --config experiments/e1_spectral/configs/logistic.json --data simulated --device cpu --output runs/tier1-e1-synthetic
python run.py evaluate --run runs/tier1-e1-synthetic --data simulated --device cpu --split test
python -m pytest experiments/e1_spectral/test_experiment.py -q
```

## Metrics, failure criteria and execution status

Use official equally weighted participant x session x interface balanced accuracy, command recalls, participant/session differences and worst-session behavior. Treat P04/P09 as previously viewed diagnostics; reserve P10 later sessions before every fitting step. Reject an apparent gain that consistently harms subtraction/word recall or comes only from one participant. Interpret improvement against matched controls, not synthetic accuracy or unrelated historical allocations.

Dependencies are existing Torch, NumPy and scikit-learn, plus the shared Benchopt/NeuralBench stack for real-data execution. No new dependency is introduced. Focused tests cover frequency/band ordering, the separate low-band extension, source-only scaling, three-class ordering, finite probabilities, single-window calls, tensor-checkpoint parity and invalid bands. Static parsing passed for all three Python files and all three JSON presets on 5 November 2026. The E1 focused suite subsequently passed all nine tests in the initial Python 3.12 environment; current Python 3.15 integration status is recorded in docs/STATUS.md. No real-data E1 run, runtime benchmark or decoding gain is claimed.

Coordinator integration, 5 November 2026: this folder's focused tests were included in the passing **88-case Python 3.15.2 suite**, and all of its named presets passed shared synthetic fitting, exact validation/test replay and isolated ZIP validation. See [current status](../../docs/STATUS.md) and the [execution guide](../../docs/TIER1_EXECUTION.md) for environment details and the two required upstream development-wheel pins. Real-data fitting and GPU execution remain unrun.
