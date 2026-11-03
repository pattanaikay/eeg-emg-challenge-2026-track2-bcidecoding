# Codabench warm-up leaderboard audit

## Bottom line

The live warm-up leaderboard contains many scores above 0.90, including 45 submissions at exactly 1.00. Several submitters' own model-name fields explicitly mention public lookup, all-public-data training, or inclusion of the public warm-up test participants. Those names are not proof of the code or of a rules violation, but they are strong evidence that the warm-up leaderboard is not a clean measure of hidden-session generalization.

Our score around 0.35 is therefore not directly comparable to a claimed warm-up score of 0.99 or 1.00. The sealed phase is the meaningful test: later sessions and new participants, with hidden labels and no labelled calibration.

## What Codabench exposes

The public Track 2 page exposes the leaderboard rows, submission IDs, participant names, duration, and a free-text fact-sheet field called **Model name**. It does not expose competitor ZIP files, source code, weights, training logs, prediction files, or detailed scoring output. The competition metadata currently reports `enable_detailed_results: false` and `make_programs_available: false`.

So we can catalogue **self-reported method clues**, but we cannot verify that a named model is the method actually used.

The captured public snapshot is phase **30961, “Warm-up Phase (update 1)”**, with **211 rows**. The primary leaderboard column is **Balanced accuracy cell mean**; the other visible columns are balanced accuracy, plain accuracy, and duration.

## What the score distribution looks like

In the captured 211-row snapshot:

| Warm-up primary score | Number of submissions |
|---:|---:|
| 1.00 | 45 |
| ≥ 0.90 | 74 |
| ≥ 0.95 | 69 |
| 0.35–0.38 | 13 |
| Minimum observed | 0.29 |

The large pile-up at 1.00 is unusual for a three-class, cross-participant/session EEG problem. It should trigger an audit of what information is available in the public warm-up evaluation—not a rush to copy a large neural architecture.

## Method clues visible in the leaderboard

The following table reports the **text entered by submitters**, not independently verified implementations.

| Score | Self-reported model name | What the name suggests | Evidence status |
|---:|---|---|---|
| 1.00 | `PROTEUS-PublicLookup-WarmupOnly` | Lookup or memorization of public warm-up examples | Self-report only |
| 1.00 | `BCI-PROTEUS-PublicLookup-Warmup-v2` | Same broad public-lookup strategy | Self-report only |
| 1.00 | `ni26 fast single ShallowConvNet+EA (all public data)` | ShallowConvNet plus Euclidean alignment, trained with all public data | Self-report only |
| 1.00 | `PROTEUS-WARMUP-ONLY-AllPublic-Shallow-v1` | Shallow convolutional model using all public data | Self-report only |
| 1.00 | `Causal-Covariance-Manifold` | Causal covariance/manifold adaptation | Self-report only |
| 1.00 | `riem_combo` | Riemannian feature combination | Self-report only |
| 0.99 | `full public data incl. P04/P09 - container test, not a skill estimate` | Explicit inclusion of public warm-up test participants | Self-report only |
| 0.98 | `HybridRiemannianDLEnsemble` | Riemannian features plus a deep-learning ensemble | Self-report only |
| 0.97 | `PSD+PhaseDiff+LDA` | Power spectral density, phase differences, and LDA | Self-report only |
| 0.97 | `fbc_net_ensemble` | Filter-bank convolutional ensemble | Self-report only |
| 0.38 | `Agnuxo Spectral Mu-Beta BCI Public PROTEUS Cross-Subject` | Mu/beta spectral features aimed at cross-subject transfer | Self-report only |
| 0.37 | Detailed EEGNet report | EEGNet, train-only normalization, no auxiliary data or online adaptation | Self-report; unusually reproducible description |
| 0.36 | `Proteus-EEGNet-Covariance-v3` | EEGNet plus covariance features | Self-report only |
| 0.33 | `Constant baseline (floor)` | Constant prediction baseline | Consistent with a chance-floor reference |

The detailed EEGNet row is useful as an anchor because it states its participant split and avoids public-test fitting. Its 0.37 score is close to our current result and far below the 1.00 cluster. That contrast is more informative than the leaderboard rank alone.

## Why public lookup can produce a spectacular warm-up score

The warm-up data are public. A submission can potentially exploit information that would be unavailable in the sealed phase: known participant/session structure, previously observed windows, public test participants, or a table of predictions for known examples. The public leaderboard itself cannot tell us which of these happened.

Some names openly advertise this boundary. `PublicLookup-WarmupOnly` and `full public data incl. P04/P09` should be read as “this score is a warm-up demonstration,” not as “this decoder will transfer to a new participant and later session.”

This is not an accusation of cheating. The official terms say that warm-up scores are indicative and do not count toward the final ranking. The sealed phase is specifically designed to remove this advantage.

## What we can and cannot conclude

We **can** conclude that the public leaderboard contains promising method families worth controlled experiments:

**Shallow convolution and filter-bank models.** These are plausible EEG baselines because they impose local temporal and frequency structure without requiring a huge model.

**Spectral and phase features.** Mu/beta power and inter-channel phase differences are physically motivated, but they must be evaluated with participant-separated preprocessing and the official cell-balanced metric.

**Covariance/Riemannian features.** These can represent spatial relationships under volume conduction and reference variation. They need shrinkage, careful training-only statistics, and a test for negative transfer.

**Small ensembles.** Combining genuinely different errors may help, but an ensemble can just average a leakage-prone predictor. It must be compared with a single model at the same data boundary.

We **cannot** conclude that any named method caused a 0.90+ score, because the actual code, weights, and data-use history are not public in the leaderboard response.

## What this means for our project

Do not try to reproduce a 1.00 warm-up score by building a lookup table. That would optimize the public board and tell us almost nothing about the sealed task.

The useful next comparison is a fair three-way baseline on the same participant split:

`EEGNet` versus `ShallowConvNet/filter-bank` versus `shrinkage covariance + Riemannian/LDA`

Use only training participants for normalization, feature fitting, model selection, and any self-supervised pretraining. Report the official participant × session × interface cell-balanced accuracy, per-class recall, and the score separately for the repeated-session participant. Keep public P04/P09 as a post-selection diagnostic, not as another tuning set.

If one of these families improves the held-out participant score without a class-recall collapse, it is a real lead. If it only reproduces 0.90 after using public test participants or lookup state, it is not a sealed-ready method.

## Sources and preserved snapshot

[Track 2 Codabench page](https://www.codabench.org/competitions/17982/) · [Competition repository](https://github.com/neural-interfaces26/2026-competition) · [Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md).

The raw public responses captured for this audit are stored beside this report:

- `codabench-17982-metadata.json` — competition, phases, leaderboard configuration, and public terms.
- `codabench-17982-warmup-leaderboard.json` — 211 warm-up rows and their visible scores/model-name fields.

Capture date: 2 November 2026. No competitor submission was downloaded, executed, or modified.
