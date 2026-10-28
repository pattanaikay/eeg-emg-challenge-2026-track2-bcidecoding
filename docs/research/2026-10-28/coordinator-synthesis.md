## Five separate agent reports

The complete reports contain citations, participant/task descriptions, numerical evidence where verified, access requirements, static code audits, interpretations and falsifiable project proposals. Read them in assignment order before the synthesis below.

### Agent A — single-trial adaptation and frequency

[Full Agent A report](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/research/2026-10-28/agent-a-adaptation.md).

**Finding:** learnable frequency filters and compact prediction summaries are separable hypotheses; the released BTTA-DG pipeline is not a trustworthy drop-in causal benchmark. At pinned commit `5932d026bbd8a7de106d31a6d264f4f4924537e4`, the training script selects checkpoints using target labels; alignment uses the full target array before streaming; and the displayed online loop never inserts samples into its initially empty GMM memory. The coordinator independently read both core files. These observations establish limitations of the inspected release, not how every reported paper run was performed. [Training source](https://github.com/luo-huan-123/BTTA-DG/blob/5932d026bbd8a7de106d31a6d264f4f4924537e4/pretrain_SincAdaptNet.py), [adaptation source](https://github.com/luo-huan-123/BTTA-DG/blob/5932d026bbd8a7de106d31a6d264f4f4924537e4/BTTA_DG.py).

**Portable component:** source-trained constrained bandpasses, tested independently of adaptation. The reported binary-MI accuracy is not an expected three-command competition score. Gradient-free updates can still corrupt predictions through bad pseudo-label memory.

### Agent B — electrode geometry and channel adaptation

[Full Agent B report](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/research/2026-10-28/agent-b-geometry.md).

**Finding:** electrode identity and coordinates provide a meaningful spatial prior, but accepting multiple layouts is different from tolerating physical cap movement. REVE's published results do not isolate a low-data geometry benefit. Its released BCIC preprocessing differs from the paper, and the channel-adaptation benchmark uses a different subject split. Thus their scores cannot form a common leaderboard. Target-statistic chronology and one metric description in the adaptation preprint remain unresolved. [REVE](https://arxiv.org/abs/2510.21585), [channel-adaptation preprint](https://arxiv.org/abs/2604.23091).

**Portable component:** same-montage geometry-aware masking or cautiously regularized covariance normalization. Importing a large external encoder is a different data regime. A spatial projection or whitening operation can retain information mathematically while making an already trained classifier less effective.

### Agent C — masking and representation objectives

[Full Agent C report](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/research/2026-10-28/agent-c-masking.md).

**Finding:** mask design changes what can solve a prediction task. The controlled large-corpus study is informative, but it does not establish a good mask for our four-second, small-corpus regime. Its benchmark combines different transfer protocols, and repeated probe seeds do not measure encoder-training uncertainty. An appendix explicitly leaves anti-collapse regularizer rescue untested despite stronger wording in the discussion. [Preprint and integrated appendices](https://arxiv.org/html/2609.33487v1).

**Portable component:** compare objectives/masks under matched controls, and check that features respond to EEG content rather than merely electrode position. Do not use low reconstruction loss as a surrogate for command transfer. Same-corpus SSL remains a later hypothesis, not an established advantage.

### Agent D — drift and stable dynamics

[Full Agent D report](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/research/2026-10-28/agent-d-dynamics.md).

**Finding:** NoMAD distinguishes a reusable dynamics model from adaptable mappings into and out of it. Its monkey alignment uses whole target sessions; its human retrospective analysis instead uses preceding blocks. Neither establishes prospective single-window EEG adaptation or unseen-person transfer. Causal decoding after fitting an alignment on a complete target session does not make that alignment causal. [NoMAD](https://www.nature.com/articles/s41467-025-59652-y).

**Portable component:** hold a useful core fixed and test a small alignment separately. The assumption that all three of our tasks share one stable state distribution is not established. This remains a research extension after simpler statistical controls.

### Agent E — augmentation and generalization safeguards

[Full Agent E report](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/research/2026-10-28/agent-e-augmentation.md).

**Finding:** augmentation should be judged against real-only training, including how often it hurts. Teacher agreement is a selection mechanism, not proof that a synthetic waveform is physiological or its label correct. The inspected release relaxes one confidence gate to retain a minimum count, while the paper describes fail-closed behavior; a generator script trains on the outer train-plus-validation pool. The coordinator independently checked both paths. The provenance of published runs was not established, so these are release/protocol gaps rather than proof of published test contamination. [Study](https://www.nature.com/articles/s41746-026-02778-0), [gate code](https://github.com/danielchoi0315/TGA-repo/blob/main/tga/runner.py), [generator code](https://github.com/danielchoi0315/TGA-repo/blob/main/scripts/tga_train_motor_ddpm_unet_ema.py).

**Portable component:** modest real-data perturbations with an explicit real-only fallback, and matched-count random-gate controls if a teacher is introduced. Full held-out-subject normalization must be replaced by an eligible current-window/prefix procedure before competition fit is claimed.

## Coordinator synthesis: evidence map

These mechanisms address different failures; they are not mutually interchangeable remedies for one diagnosed bottleneck.

| Mechanism | Information it tries to retain | Variation it may tolerate | What limits the inference |
|---|---|---|---|
| Sinc/constrained temporal filters | Selected oscillatory components and their temporal evolution | Some irrelevant out-of-band activity; fewer free filter coefficients | Relevant bands vary by task/person; filtering can discard useful information. Binary-MI findings do not settle three-command transfer. |
| Coordinate-aware input/masking | Where a signal belongs on the scalp | Missing channels, coordinate uncertainty or layouts under tested conditions | Fixed geometry is not a calibrated physical model of electrode displacement. External pretraining and geometry effects are confounded. |
| Covariance normalization | Spatial fluctuation structure relative to a reference | Some gain/mixing differences | A marginal reference also changes with task mix; whitening can remove useful distinctions or destabilize an encoder. |
| MAE/latent prediction | Predictable signal or teacher-defined features | Only variations that the learned representation actually ignores | Predictable artifacts, interpolation shortcuts, identity-dominated features, collapse and pretraining overlap. |
| Frozen dynamics + small alignment | Reusable state transitions/behavioral mapping | Changes in the observation-to-state mapping | Different behavior, lost information, whole-target fitting, and ambiguity of unlabeled alignment. |
| Gated augmentation | Task-consistent variation according to a teacher/validation rule | Selected perturbations represented in training | Teacher bias, class-skewed acceptance, synthetic distribution errors, selection overfitting and mismatched data budgets. |

**Agreement:** keep task-discriminative structure, limit unsupported degrees of freedom, separate source learning from target adaptation, and judge utility on held-out people/sessions. **Disagreement or non-comparability:** papers optimize different objectives and metrics on different populations; some fit target distributions offline, some require target labels, and others only evaluate frozen representations. Their numerical gains cannot be added or transferred to our score.

There is no supported conclusion that covariance universally beats temporal networks, that JEPA universally beats reconstruction, or that geometry always improves decoding. A method can look stable because it removed both nuisance and task information. Conversely, a session-specific feature can be genuinely task-relevant.

For window features Z, even unchanged class-conditional distributions can have changing marginal covariance:

\[
\mathrm{Cov}(Z)=\sum_y\pi_y\mathrm{Cov}(Z\mid y)
+\sum_y\pi_y(\mu_y-\mu)(\mu_y-\mu)^\top.
\]

Here \(\pi_y\) is the class proportion. This identity explains why matching an unlabeled distribution is not automatically removing recording drift: it can instead respond to which tasks appeared recently. It also explains why a more confident or less novel representation need not be more accurate.

## Competition fit and exclusions

| Category | Candidate | Boundary |
|---|---|---|
| Compatible as a source-trained component | Band power, shrinkage covariance, constrained filters, static coordinates, small within-window temporal summaries | Fit all learned choices on training participants and use the supplied bandwidth/montage. No benefit is yet measured here. |
| Compatible in principle; needs causal evaluation | Session-prefix normalization; bounded statistical memory; restricted current-window adaptation | Reset every session, define cold-start fallback, exclude future windows/labels/IDs, measure early-stream harm. |
| Compatible training augmentation hypothesis | Modest gain/noise/dropout/masking, optionally teacher-gated | Preserve task information empirically. Fit teacher/generator only inside the relevant training fold and match accepted counts in controls. |
| Outside the primary fixed-corpus branch | External REVE or other pretrained weights | May be competition-eligible with licences/disclosures, but adds external training information. Do not silently mix with the fixed-corpus comparison. |
| Not eligible as deployed in the inspected path | Whole-target-session normalization/alignment; target-label checkpoint selection; across-session memory | Rebuild the information flow before evaluating it as a competition method. |
| Not an observed input | Sealed calibration labels; participant/interface tokens; environmental/cue labels; unavailable high-frequency features | Do not condition prediction on these. Metadata can define development folds/metrics without being model inputs. |

Specific leakage controls: split by participant before feature fitting, SSL or augmentation; keep all crops/overlapping windows of a trial together; do not use future target windows for normalization; select using inner validation rather than outer test or leaderboard; and report the official cell metric rather than pooled accuracy. A fixed current-window transform is not automatically leakage. Offline transductive protocols can be legitimate for their stated question but do not establish compatibility with ours.

## Prioritized experiments — proposals only

### Shared development protocol before choosing a winner

Use the eight existing training/validation participants for a frozen development plan: rotate held-out people in an outer grouped evaluation, and make preprocessing/hyperparameter/checkpoint choices using only inner training/validation people. Repeat initialization separately from participant rotation. Keep P04/P09 out of further selection, acknowledging that their previous results have already been inspected. Existing metadata identify P10 as the repeated-session person in development and P04 as the repeated-session person in public test. For an exploratory known-person P10 analysis, exclude sessions 2–3 from every fit, preprocessing reference and SSL stage; session 1 may join source training, and later sessions are evaluation only. P04 can provide a separately labeled post-selection diagnostic, not another tuning cohort. Neither single-person analysis supplies a population-level longitudinal guarantee.

Replay each validation session in the evaluator's order and reset exactly. Primary outcome: paired difference in official participant-session-interface cell BA. Also show per-class recall, per-person/interface effects, startup versus later-window performance, and measured inference cost when an experiment is actually authorized. Aggregate seeds within a participant before treating people as uncertainty units; do not turn overlapping windows or seeds into independent participants. Small participant counts limit confidence intervals and significance claims.

### Experiment 1 — identify what representation helps

- **Hypothesis:** explicit oscillatory power or spatial second-order structure is easier to estimate with our data than the current learned representation.
- **Minimal change:** compare the unchanged EEGNet and channel-mean control with a small band-power + regularized classifier and shrinkage-covariance + regularized classifier. These are diagnostic controls, not new 2026 techniques.
- **Comparison/ablation:** common windows, participant folds, label budget and selection policy; fit scaling, spatial filters, covariance reference and classifier within each fold. Predeclare a small representation/tuning menu rather than exhaustively searching frequency bands.
- **Metric:** shared cell BA and class recalls; inspect paired participant differences.
- **Against the hypothesis:** no consistent advantage over matched EEGNet across held-out people, or benefit concentrated in one person/interface. That would not prove the features contain no task information; it would reject this practical advantage under the tested recipe.

### Experiment 2 — isolate constrained frequency learning from BTTA-DG

- **Hypothesis:** restricting temporal filters to meaningful bandpasses reduces estimation error without removing command information.
- **Minimal change:** a Sinc-style front end in an otherwise matched small source-trained decoder, with valid cutoffs below the 120-Hz input's Nyquist limit. No imported checkpoints or GMM adaptation.
- **Comparison/ablation:** constrained versus unconstrained temporal filters; hold pooling, normalization, spatial stage and training schedule fixed as far as possible; record parameter counts. Compare with the strongest representation control from Experiment 1 using nested selection, not public-test choice.
- **Validation/metric:** the shared participant protocol; cell BA, class recall and seed variation.
- **Against the hypothesis:** equal/worse held-out performance, loss of subtraction/word-generation recall, or a gain explained by a changed normalization/training budget rather than filter constraints. Attractive learned passbands alone do not support the hypothesis.

### Experiment 3 — test whether causal normalization removes useful mismatch

- **Hypothesis:** a meaningful part of transfer failure is a nuisance change that current/past-window statistics can reduce.
- **Minimal change:** freeze a selected source decoder and compare its existing preprocessing against a regularized per-session prefix reference initialized from training data. Match any corresponding source preprocessing during training. Specify whether the current window is included; both the information boundary and ordering must be reproducible.
- **Comparison/ablation:** source-only fixed reference versus causal reference; no parameter updates, IDs or target labels; identical reset/cold-start policy. Test robustness to naturally occurring class mixtures; do not assume balanced prefixes.
- **Validation/metric:** chronological session replay; cell BA plus first-window/startup and class-specific losses. Group the exploratory later-session results by actual participant.
- **Against the hypothesis:** causal replay loses the apparent benefit, startup errors offset later gains, or normalization repeatedly collapses a command. This tests a statistical remedy, not the biological cause of drift.

### Experiment 4 — test one inexpensive robustness prior

- **Hypothesis:** modest label-preserving perturbations improve transfer without needing synthetic subjects or a new backbone.
- **Minimal change:** one gain/noise/dropout or spatial-temporal masking family on the selected small decoder; compare against real-only training. Magnitudes and budgets are new choices for inner validation, not imported paper optima.
- **Comparison/ablation:** for geometry, compare contiguous scalp neighborhoods with equal-budget random groups. If adding a teacher gate, compare no gate, teacher gate and matched-count random acceptance; include the teacher itself as a decoding baseline and a real-only fallback. Do not introduce a diffusion generator in this first comparison.
- **Validation/metric:** the same folds, label/tuning budgets and cell BA; show the distribution of paired participant changes and class-skewed acceptance.
- **Against the hypothesis:** matched random selection explains the gain, accepted examples increase class bias, or average improvement hides repeated participant/command harm. A subject-scarcity study is not evidence that more synthetic windows replace missing people.

**Deferred:** a full same-corpus SSL comparison, NoMAD-style dynamics alignment, and GMM output adaptation. If the earlier tests justify them, use compact models and separate each objective/adapter from the representation. For SSL, compare matched scratch/supervised compute, split before pretraining, and use only training participants' resting recordings if that addition is declared. For adaptation, keep an untouched starting model and compare the same held-out data before/after updates; better transfer and less forgetting are distinct outcomes. No specific experiment has been chosen by the user in this review.

## Open questions and evidence needed

1. **Which factor actually limits our baseline?** Only matched representation/normalization/training comparisons can distinguish plausible bottlenecks; current scores do not diagnose them.
2. **Can a three-command target-memory method start safely?** Need empty/insufficient-class behavior, chronological code, bounded runtime, no target-label selection and a clean reference against the frozen decoder.
3. **Which published outputs correspond to which released preprocessing and selection path?** Author versioned experiment configurations/checkpoints or clarification would resolve the BTTA-DG, REVE and augmentation release gaps. Static differences do not establish misconduct.
4. **Are masked features task-sensitive after controlling for identity?** Need held-out probes and content/position diagnostics; reconstruction or pooled variance is insufficient. The paper's pretraining overlap also needs recording/participant mapping.
5. **Does alignment handle recording drift or class composition?** Need controlled within-development comparisons with fixed label budgets and natural chronological task mixtures. Unlabeled marginal statistics cannot universally identify harmful conditional drift.
6. **What is the enforceable submission/runtime limit?** Resolve the Terms/site discrepancy from the active platform or organizer before execution/submission planning. This review does not require consuming an upload.

## Accountability and limits

All six named leads were located and investigated; publication/preprint status and access limits appear in the separate reports. Some primary code was pinned; other current-main code and unavailable artifacts are labeled. The coordinator independently checked the decisive BTTA-DG and augmentation code paths and the evaluator, not every agent-inspected file.

Measured token usage: **coordinator unavailable; A unavailable; B unavailable; C unavailable; D unavailable; E unavailable; combined unavailable**. No hard combined cap was verified. Source fetches, PDF extraction and static document/code reading are literature work; no decoder experiment, code repair, model training, cloud run or submission occurred.

The research journal records this investigation separately from experiments. Recommendations above remain proposals, not user-selected next steps or demonstrated competition gains.
