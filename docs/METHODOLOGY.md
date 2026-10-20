# What we are measuring and why

## Longitudinal transfer

Longitudinal transfer means using learning from earlier recordings to decode later recordings of the same person. If a decoder learns a user's command patterns on the first visit, does it still work when that user returns? Cap replacement, contact quality, reference/pipeline changes, eye/muscle activity, attention and strategy can change the observation. Passing a random window split from the same recording would not establish this ability.

The current competition also includes **cross-participant transfer**: a person was absent from the training release. The two problems overlap but are not interchangeable. We begin with the official participant-disjoint public split, while the limited repeated-session public data support only exploratory longitudinal analysis.

## Why these baselines?

1. **Constant prediction:** a contract reference. It checks whether the ZIP loads and runs. It learns no task information.
2. **Torch-Linear:** average each EEG channel over the window, then apply a linear classifier. It is small, easy to debug and intentionally loses timing information. A weak result can motivate richer features but is not evidence that all linear EEG decoding is ineffective.
3. **EEGNet:** temporal and spatial convolutional processing with the sample order retained. We use Braindecode's implementation and record its installed version. Ordered channel names are checked at reload. This is not yet an explicit coordinate-aware scalp geometry model.

The synthetic smoke fixture adds class-dependent channel means and independent noise. It is deliberately easy for the linear baseline. Near-perfect synthetic scores mean the test fixture was learned and the pipeline works; they do not predict the challenge score.

## The experiment flow

```mermaid
flowchart LR
 A[Public PROTEUS] --> B[Official preprocessing and participant split]
 B --> C[Train on training participants]
 C --> D[Select checkpoint on validation cells]
 D --> E[Evaluate selected checkpoint on test cells]
 E --> F[Freeze artifacts and replay inference]
 F --> G[Package one-file inference ZIP]
```

Only training data receive gradient updates. Validation chooses a checkpoint. Test labels are used to calculate the final development score, not to fit parameters or select epochs. Replaying the same test set checks reproducibility; it is not a second independent scientific test.

The initial implementation uses no inference-time adaptation. It therefore gives a fixed reference against which causal normalization or another permitted update can be assessed later. Existing output directories are rejected to prevent accidentally starting a new experiment from old weights.

## Drift versus forgetting

A fixed decoder can fail because the observations or their relationship to the commands changed. That is the transfer/drift problem. An update can then damage previously useful performance: forgetting or negative adaptation. Keep the two measurements separate.

For public development, compare the same earlier held-out trials before and after a proposed update. Also measure its gain/loss on later held-out data. The competition resets adaptation between sessions, so a model cannot carry a personalized updated state across visits in its sealed submission. Cross-session retention experiments are therefore separately defined research analyses.

## Calibration and label efficiency

The initial report discussed early labeled per-user calibration. Current sealed participants do not receive additional labeled calibration through `fit`. We can study label efficiency by reducing labeled examples from the public **training** partition, preserving validation/test boundaries and repeating the budget sampling. That asks whether a method needs fewer training labels, not whether we have permission to request sealed labels.

For a separate known-user calibration study, specify which public early-session trials are used and which later/earlier trials remain untouched. With only two public participants having multiple sessions, report individual outcomes and avoid broad population claims.

## What a harmful-drift monitor would need to show

A large recording change need not cause a large decoding error. Conversely, a small change in a decoder-critical direction can matter. A monitor must predict an operationally defined error increase or failure event on held-out data. Compare it against signal quality, prediction confidence and generic covariance/raw-distribution shift.

Detecting current deterioration differs from warning about future failure. Forecasting needs a predeclared horizon, valid temporal order and enough independent failures. A model can remain confident while wrong, and marginal P(X) may remain unchanged when P(Y|X) changes. No unlabeled novelty score has a universal guarantee.

For the first research extension, inspect saved confusion and cell scores. Then add a single controlled change and matched comparison, rather than simultaneously changing the encoder, pretraining, memory, retention and normalization. An improvement must survive the same splits, tuning budget, label budget and scoring definition.

## Mechanistic interpretation

A probe asks whether information is recoverable from a representation. An ablation asks whether the decoder depends on a component under that intervention. Neither alone proves that the component is a causal brain circuit. Use task/context-matched comparisons, matched random subspaces, and out-of-distribution intervention controls.

Stable decoding does not require stable numerical latent coordinates. For a linear readout, `z' = Rz` and `W' = W R^-1` preserve `W'z' = Wz`. Preserving useful computation is therefore a stronger research target than arbitrarily fixing every coordinate.

## Boundaries of the first milestone

This repository provides the baseline, provenance, data audit, test scripts and export path. It does not establish a new adaptation method, a foundation-model scaling law, clinical benefit, biological causality, or compatibility with proprietary BCI hardware. The paper route explains what evidence those later claims would require.
