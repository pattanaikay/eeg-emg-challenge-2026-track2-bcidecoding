# Model-facing contract and evaluation boundaries

Contract snapshot: **20 October 2026**. Source authority is the active Codabench Track description/phase, the scientific FAQ, and executable objective. Older NeuralBench prose still contains incompatible descriptions. Recheck when the phase changes.

## Input and prediction

The current PROTEUS contract gives one `(1, 41, 480)` float EEG tensor at a time. The original 500 Hz recording is converted by the evaluator into a four-second, 120 Hz window beginning one second after a cue. Fixed preprocessing includes 50/60 Hz notch filtering, 0.1–75 Hz band-pass at the recorded rate and resampling; inference receives microvolts without learned scaling/clipping. The supplied window is the observation: a model cannot request raw sealed recordings.

Return a `(1,)` vector containing an integer in `{0,1,2}` for motor imagery, mental subtraction or word generation. Respect `meta["device"]`, the actual channel order and dimensions. The model may receive input descriptions through `load_model(meta)`, but `predict` gets neither subject/run identifiers nor labels.

## State and adaptation

One stream is one participant-session, spanning its runs/interfaces in acquisition order. A session starts from a deep copy of the submitted starting model; optional `reset_state()` is then called. Causal label-free transformations and updates may use current/past windows of that session. Neither data nor updated statistics/weights carry into the next session. These baseline models intentionally do no inference adaptation.

The fixed upstream preprocessing is performed before streaming. Therefore “causal” here describes the model-facing sequence, not a claim that the dataset's filtering is an online analog filter. A proposed real-world continuous BCI would need its own preprocessing/latency validation.

## Which split answers which question?

The public release has 112 runs, 14 sessions and 10 participants. Two participants have sessions 1–3; eight have session 1 only. The official public development task draws a 60/20/20 **participant** split with split random state 33. All sessions of one participant stay in one partition. Training RNG seeds do not replace the configured split seeds.

The private final includes later sessions of represented participants and recordings of unseen participants. Extra labeled sealed calibration is not supplied through `fit`; Codabench is inference-only. A labeled per-person calibration experiment must be a separately defined public-data analysis.

## Score exactly the right quantity

For each participant-session-interface cell, compute mean class recall, then average cell scores equally. The pinned upstream helper defines the class set as the union of true/predicted classes; a predicted-but-absent class contributes zero recall. This can differ from scikit-learn's balanced accuracy when a target class is absent.

The project records the official cell mean, a session-only mean, pooled balanced accuracy, ordinary accuracy, per-cell values and a confusion matrix. Only the cell mean is the current ranking statistic. Tests compare our aggregation against the unchanged upstream Objective on identical predictions.

## What the baseline training changes

Architectures follow the worked reference: channel-mean linear classification and Braindecode EEGNet. The project adds explicit validation-based checkpoint selection and run provenance. Consequently, published upstream neuralbench averages are **context**, not an exact expected score for this implementation. Never tune using the local test partition or leaderboard feedback as if it were independent validation.

## Authoritative links

- [Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md)
- [Track objective](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/bci_decoding/objective.py)
- [Streaming evaluator](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/streaming.py)
- [Metric helpers](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/metrics.py)
- [Active warm-up config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/bci_decoding/config.yaml)
- [Submission guide](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md)
- [Official rules and schedule](https://neural-interfaces26.github.io/rules.html)

The exact source revision used locally is recorded in `../UPSTREAM.json`; links to `main` can change after this snapshot.
