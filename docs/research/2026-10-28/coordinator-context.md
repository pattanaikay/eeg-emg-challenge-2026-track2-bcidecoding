# Robust EEG decoding across sessions and participants

Literature investigation and static evidence audit — **28 October 2026, Asia/Kolkata**.

**Question:** What task information does a method preserve, which recording variation does it tolerate, and does its evaluation demonstrate both without information unavailable in this competition?

This report supports choosing experiments. It does not report a new PROTEUS experiment, submission, model improvement, or clinical result. All experiment suggestions remain proposals unless the user subsequently chooses one. Borrowing an architectural component and importing externally pretrained weights are separate decisions.

## Verified competition context

The coordinator inspected the current [track README](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/README.md), [track description](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/competition_bci.md), [scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md), [objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/objective.py), [streaming evaluator](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/streaming.py), [metric helper](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/metrics.py), data adapter, warm-up config, submission guide, Terms and timeline. Links to `main` are mutable; this is a dated inspection, not an assertion that all current files match the locally pinned source byte for byte.

| Boundary | Current inspected behavior |
|---|---|
| Observation | One `(1,41,480)` EEG tensor; four seconds at 120 Hz, starting one second after cue, in microvolts. |
| Prediction | One of motor imagery, mental subtraction, word generation. No label, participant/session/run/interface ID reaches `predict`. Input-description metadata can reach `load_model`. |
| Available context | Complete current window and earlier windows of the current session. Within-window bidirectional processing is compatible; future-window statistics are not. |
| State | Fresh deep copy of the submitted model at each participant-session; optional reset. Windows, statistics and parameter changes cannot cross sessions. |
| Score | Mean class recall within participant × session × interface cells, then equal mean across cells. Context is Graz/BrainHero. Executable helper uses the union of true/predicted classes, relevant when classes are absent. |
| Warm-up versus sealed | Public PROTEUS test participants versus private later sessions of represented people plus unseen people. No extra labeled sealed calibration. Warm-up performance is indicative, not final ranking. |

These core semantics agree with the project's 20 October contract snapshot. The phase config's `seed:42` is not evidence of a changed participant split: the NeuralBench config separately fixes validation/test split random states to 33. [Warm-up config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/bci_decoding/config.yaml), [loader](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/nb_task.py), [NeuralBench v0.4.0 config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml).

**Material ambiguity:** [repository Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md) and [timeline](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/timeline.md) say five warm-up uploads/day; Terms give a 60-minute evaluation limit. The [main website rules](https://neural-interfaces26.github.io/rules.html) say ten/day and 80 minutes. Both give five total sealed submissions, one/day, and warm-up closing 1 November 23:59 CET. Active platform enforcement was not inspected. Plan conservatively for five/day and 60 minutes until the organizer/platform resolves this; this does not assert which page has precedence.

Both rule sources allow declared, appropriately licensed public external training data. That permission does not remove the user's stricter fixed-corpus research preference. Our primary proposed comparisons use PROTEUS alone; pretrained-weight experiments would be a separately disclosed branch. Submission inference is self-contained, cannot download dependencies/data, and must fit the worker environment. [Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md), [submission guide](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md).

## Actual accessible project baseline

Read the [journal](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/RESEARCH_JOURNAL.md), status/methodology/contract, [run log](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/JARVISLABS_RUN_LOG.md), pinned-source manifest, and [verified-results.json](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/runs/jarvis/20261021T162400Z/verified-results.json). The pinned competition revision is `a66bc23d1a0e686b895c093a642b45c74970758b`, retrieved 20 October. The current upstream HEAD SHA was not obtained through the web API.

| Completed 21 October run | Validation cell BA | Public-test cell BA |
|---|---:|---:|
| Channel-mean linear, seed 33 | 0.445861335 | 0.379661043 |
| EEGNet, seed 33 | 0.453858647 | 0.352390098 |
| EEGNet, seed 34 | 0.454217070 | 0.382836157 |
| EEGNet, seed 35 | 0.467058692 | 0.363920946 |

These are three initialization seeds on one fixed split, not three participant folds. Checkpoints were selected by validation cell mean. They do not establish whether poor transfer arises from representation, normalization, optimization or data limitations. The frozen baseline has no inference-time weight-update forgetting.

The current official README's EEGNet/CTNet/REVE reference scores use a session-mean metric and different training recipes; they cannot directly rank our cell-mean runs. A statement that the mixed sealed protocol is easier is not a guaranteed ordering for our model or its hidden population.

Public partition: train P01/P02/P03/P06/P07/P10 (1,521 windows); validation P05/P08 (379); public test P04/P09 (755). The existing [window inventory](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/runs/jarvis/20261021T162400Z/remote/runs/eegnet-seed35-20261021T164141Z/split_windows.csv) identifies P10 (training) and P04 (public test) as the two people with sessions 1–3; everyone else has session 1. The public test has already been inspected, so it should not be relabeled a pristine untouched confirmation set.

Remote replay/export and local retrieval checks are documented; no runtime was rerun here. The only inspected platform score is the user's reported constant submission 0.33. No trained-model Codabench or sealed result was established. Raw EEG, cloud/account state, private data, and activities outside accessible records were not inspected in this literature task.

## Delegation and accounting

The session limit was four active agents including the coordinator. Wave 1: A/B/C. Wave 2: D/E, launched after B/C completed while A finished. Exactly five research subagents were used. The tool exposes neither per-agent numeric budgets nor a verified combined hard cap. No numeric budget was requested. Measured coordinator, per-agent and combined token usage were unavailable; no estimates are presented as measurements.

The five separate reports follow before the coordinator's synthesis. They distinguish paper claims, static release observations, interpretation and untested hypotheses.
