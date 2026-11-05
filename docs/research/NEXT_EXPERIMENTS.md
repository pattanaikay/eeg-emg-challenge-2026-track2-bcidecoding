# PROTEUS: next experiments

3 November 2026 · Proposed roadmap; no experiments executed for this analysis.

**Decision:** establish spectral, covariance and normalization controls first; then test learned spatial/frequency filters. Delay SSL and learned adaptation until their simpler controls reveal a useful signal. These priorities are engineering judgments, not demonstrated PROTEUS gains.

## Evidence and deployment boundary

The [verified runs](../../runs/jarvis/20261021T162400Z/verified-results.json) give EEGNet public-test cell BA **0.3524 / 0.3828 / 0.3639** (mean **0.3664**) and channel-mean linear **0.3797**. Seeds are initialization repeats on one participant split. The linear model is neither a spectral nor a covariance baseline. The inspected code applies no additional explicit channel scaling; frozen BatchNorm is part of EEGNet. Current evidence does not identify the bottleneck.

Inference receives `(1,41,480)` EEG at 120 Hz for imagery/subtraction/word generation, without IDs or labels. Current-window processing is permitted in full; causality prohibits future **windows**, not later samples inside the supplied window. Adaptation resets each participant-session. Training-domain IDs are available for splitting or auxiliary losses. External weights remain outside our primary comparison. [Official FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md).

## 1. Candidate inventory and failure modes

Each row gives the smallest experiment. Costs are qualitative; no new runtimes have been measured.

| Candidate | Smallest honest test | Main friction; priority |
|---|---|---|
| Filter bank + band power | Log power per channel → L2 logistic regression; shrinkage LDA as head ablation | Loses event order; include cognitive-task bands, not only mu/beta. CPU, **E1**. |
| Riemannian covariance + LDA | Shrunk covariance → training-reference tangent vector → regularized head | Spatial mixtures/artifacts survive; 861 features per band require regularization. CPU, **E2**. |
| Robust normalization | Fixed training mean/std; separately current-window median/IQR or clipped scaling | Can remove useful amplitude differences. CPU, **E3**. |
| Euclidean/Riemannian alignment | Training-initialized, shrunk running covariance; compare arithmetic/geometric reference only later | Cold start and class mixtures bias the reference. Whole-target fitting is ineligible. **E3**, geometric variant secondary. |
| CSP / FBCSP | Shrinkage one-vs-rest CSP, few spatial components, log variance; same bands/head as E1 | Valid for three classes; supervised spatial filters can memorize people. CPU, **E4**. |
| ShallowConvNet / FBCNet | Small learned spatial/power model; fixed bank for FBCNet | Oscillatory bias can miss cognitive-task timing. Small GPU, **E5**; test sequentially. |
| BTTA Sinc filters | Replace only temporal filtering in a matched small decoder | Cutoffs below 60 Hz; MI physiology does not establish subtraction/word benefit. Small GPU, **E6**. |
| Small ensemble | Equal-probability average of two complementary surviving models | Repeated validation weighting can overfit; no temporal label smoothing. Cheap, **E7**. |
| Mild augmentation | One training-only noise/gain/dropout family | Cropping, frequency perturbations and masks may erase the class cue. **E8**; test each separately. |
| Multi-scale CNN/TCN | One extra temporal scale, same classifier | Timing hypothesis, extra parameters/tuning. No online cold start. **E9**. |
| BTTA confidence summaries | Frozen per-time class probabilities → mean versus clipped mean-log/concentration summary | Summary loses order; current EEGNet output is window-level, so requires a matched temporal-output head. **E10**. |
| Causal confidence memory / averaging | Bounded, resettable memory; no-memory fallback | Single-class streaks and confidently wrong admissions corrupt memory. Never impose balanced prefixes. **E10**, conditional. |
| NoMAD-inspired adapter | Frozen classifier + identity-initialized adapter + weak moment/identity penalty | Can erase class separation despite lower alignment loss. Compare normalization alone. **E11**. |
| Normalization-layer TTA | Update only a small affine normalization component | Batch-one statistics are unstable; entropy minimization can collapse classes. **Tier 3**. |
| MAE/JEPA masking | Compact encoder, 2×2 objective × mask comparison | Position shortcuts, redundant scalp signals, teacher cost and few independent people. **E12**. |
| Contrastive SSL | Same encoder; two mild views from each training trial | False negatives and label-destroying invariances; adds another objective search. **Tier 3**. |
| Domain-adversarial training | Training-only subject adversary, single penalty sweep | No inference IDs needed; removing subject information can also remove task information. **Tier 3**. |
| Electrode coordinates / graphs | Fixed coordinate neighbourhood versus equal-size random neighbourhood | Preserve channel identity; volume conduction makes proximity an imperfect connectivity prior. Fixed masking in **E8**; learned graph **Tier 3**. |
| GMM calibration | Classwise density on useful confidence summaries | Empty/missing-class banks, overconfident errors, density estimation with short histories. **Tier 3**, after E10. |
| Entropy/drift monitoring | Record confidence/quality against later offline errors | Monitoring alone improves no predictions; low entropy is not evidence of correctness. Diagnostic alongside E3/E11. |

Spectral/CSP/covariance models have no pseudo-label cold start. Every stateful variant must separately expose first-window, first-ten-window and later performance; missing-class startup segments need class-specific error counts, not a forced three-class BA.

## 2. Four gates: correcting the proposed falsification protocol

**G1 — Fit and time boundaries.** Log participant/session/run membership for every `fit`: scalers, CSP, PCA/ICA, tangent references and SSL use inner training only; inner validation may select hyperparameters/checkpoints. Assert disjoint fitting/evaluation records. Permuting outer-test labels must leave every fitted parameter unchanged. Appending future windows must leave prior outputs/prefix states unchanged; test fresh-copy/reset equivalence. Current-window statistics are legal; whole-target-session fitting is not. These are implementation requirements, not checks already passed here.

**G2 — Shortcut controls.** Shuffling features *within a class/session preserves their label association*. Instead rerun the fitted pipeline under repeated label permutations at an exchangeable trial/block level, respecting dependence; compare to that empirical null. Temporal shuffling/phase scrambling asks narrower questions: covariance/power can survive, so above-chance decoding need not invalidate them. Pre-cue-only prediction and EEG-content/position swaps flag possible shortcuts. Above-chance pre-cue results require investigation: anticipation, previous-task activity and filter contamination are alternatives. Such windows require separate public extraction, with cue/filter separation; they never arrive at inference. No frequency band is automatically pure noise; >60 Hz is unavailable. Reject a transfer claim if its gain depends on an identified unavailable cue or disappears after matched confound removal.

**G3 — Transfer stress.** Split by participant with sessions/runs intact; separately fit earlier and evaluate later sessions for known-person transfer. Report cell BA, command recalls, worst session, session-score variance and participant/interface deltas. Gap chronological within-session splits to remove overlap/filter support; these supplement unseen-person testing. Stress TTA on natural single-class streaks and training-derived class-reweighted streams, without assuming equal task frequencies.

**G4 — Complexity.** Match data, checkpoint policy, tuning opportunities and relevant compute budgets. Report paired participant deltas, averaging seeds within participant. The requested “>1 seed SD” is only a screening heuristic; deterministic methods have zero seed SD. Predeclare a practical improvement margin, report participant-level uncertainty and worst-class/session harm. Windows/seeds are not independent people; this cohort may leave gains inconclusive. Retain experimental branches, but keep the simpler production model when evidence is weak.

## 3. Validation setup before new models

Screen on the existing six training/two validation people, retaining P04/P09 for post-selection diagnostics; their results have already been viewed. Confirm finalists through four fixed outer participant folds over the eight development people, with preprocessing/hyperparameters/checkpoints selected inside the remaining participants. Repeated use makes this a robustness assessment, not a pristine independent confirmation. Refit EEGNet and linear controls on every comparison split.

**Reserve P10 sessions 2–3 before any new fitting, selection or SSL.** They currently contribute 385 training windows. The prospective screen therefore has **1,136 train / 379 validation** windows, plus those 385 longitudinal diagnostic windows. Use P10 session 1 in training when appropriate; replay sessions 2 and 3 independently from the starting model. Only P10 supplies this development longitudinal test; it cannot establish population reliability. Existing 1,521-window scores cannot be directly compared with this reduced-training experiment. P04 remains an unseen-person diagnostic under the original split.

## 4. Ranked experiments

All rows require **G1–G4**. Primary metric throughout is official equal participant × session × interface **cell BA**; the last column adds experiment-specific sanity checks and rejection conditions. Gates below are proposed, not completed.

### Tier 1 — immediate work, in this order

| ID | Hypothesis and minimum change | Control | Additional sanity check / stop condition |
|---|---|---|---|
| **E1** | Explicit spectral power generalizes better. Log-power/L2 model; candidate bands 4–8, 8–13, 13–30, 30–45 Hz, then 1–4 Hz ablation. Proposed choices, not validated optima. | Refit EEGNet and channel-mean linear on identical windows. | Inspect recalls and band-removal effects. Reject gains hiding recurring subtraction/word harm or confined to one person; high-band benefit alone does not prove muscle contamination. |
| **E2** | Cross-channel covariance adds information beyond channel power. Shrinkage covariance → tangent space → regularized LDA/logistic head. | E1 plus diagonal-only covariance with matched head/tuning. | Center each window; ensure positive eigenvalues and fixed source tangent reference (`tsupdate=False`). Reject if full covariance cannot beat diagonal features or fails held-out people. |
| **E3** | Amplitude/contact variation explains part of transfer failure. Compare source mean/std, per-window robust scaling, then regularized running normalization/alignment individually. | Same selected representation/model with its matched fixed-source preprocessing. | Choose current-window inclusion in advance; match source training replay. Shrink startup toward source reference. Reject startup losses, class-streak collapse or future dependence. |
| **E4** | A few supervised spatial projections improve E1. One-vs-rest CSP; add filter bank only if useful. | Same bands, log-power head and budget as E1. | Fit covariance, spatial filters and component selection on inner training only. Reject person-specific gains that disappear across folds. |
| **E5** | Learning spatial power filters improves handcrafted features. Small ShallowConvNet first; FBCNet only after the bank comparison warrants it. | EEGNet and best statistical model. | Match preprocessing/checkpoint opportunities; record parameter count and time. Reject gains explained by extra training or loss of cognitive-class recall. |
| **E6** | Constrained frequency learning helps independently of BTTA calibration. Sinc versus unconstrained temporal kernels. | Same spatial stage, pooling, temporal head and normalization. | Keep 0 < lower < upper < 60 Hz. Reject if gain vanishes after matching capacity/budget; passband plots are insufficient. |

E1/E2 have the highest expected information per GPU hour because they need no GPU and distinguish power from spatial structure. E3 tests a cheap nuisance explanation. E4–E6 then ask which learned filters earn their complexity. Do not launch a factorial sweep of all normalizers and architectures.

### Tier 2 — conditional follow-ups

| ID / entry gate | Hypothesis and control | Metric sanity check / stop condition |
|---|---|---|
| **E7**, two useful branches | Complementary errors improve cell BA: equal-probability average versus each branch. Tune calibration only on inner validation. | Paired gain over the best single model; reject if subtraction errors remain identical or weighting needs repeated outer tuning. |
| **E8**, stable baseline | Mild gain/noise/dropout improves transfer versus real-only, same updates. Test local versus random channel masks at equal severity as a separate geometry ablation. | Class recalls, natural/controlled dropout; reject if perturbations remove task cues or geometry adds no benefit over random groups. |
| **E9**, timing evidence | A second temporal scale improves over a parameter-matched single-scale CNN. | Use timing-removal ablations to interpret the gain; reject if matched single-scale training performs as well or compute matching explains it. |
| **E10**, usable temporal head | Confidence dispersion adds information: mean probability versus clipped mean-log/Dirichlet summary, source-trained head. Only then add simple causal memory. | Content swaps and temporal-order removal; compare identical encoder outputs. Reject redundant summaries or wrong-label amplification. Missing-class fallback must return the base prediction. |
| **E11**, E3 helps | Small adapter improves beyond normalization. Compare no adaptation, E3 alone, and identity adapter with weak alignment/identity penalty, classifier frozen. | BA versus alignment loss; training-derived class-mixture stress. Reject lower alignment loss with worse class recall. Offline frozen pre/post snapshots on held-out anchors measure functional forgetting separately from drift. |
| **E12**, simpler tests exhausted | Fixed-corpus SSL improves a compact encoder: MAE/JEPA × scattered/contiguous masks. Compare random encoder, same encoder supervised from scratch, compute-matched scratch, and E1/E2/EEGNet. | Split before SSL; exclude P10 later sessions and held-out people. Match realized mask fraction, decoder, steps and data; include teacher cost in compute controls. Reject gains if embeddings ignore EEG content at fixed positions, or SSL fails to beat matched scratch. |

### Tier 3 — hold or discard

Hold GMM, contrastive SSL, subject-adversarial training, learned graphs and normalization-layer gradient TTA. Each adds an untested assumption or tuning axis; admit only one after a simpler ablation identifies the missing capability. A GMM needs E10 success, empty/insufficient-class fallback, confidence-error stress and a bounded runtime.

Discard full NoMAD/spike-Poisson reconstruction for this EEG entry, wholesale BTTA release reuse, full-network pseudo-label updates as a first adapter, whole-session target normalization, unavailable IDs/calibration, external-weight imports into the primary comparison, and public-window lookup as a transfer strategy. Confidence monitoring alone is not a decoder improvement.

## 5. What combines, and what conflicts

`window → normalization/alignment → spectral/spatial/temporal representation → classifier → optional causal calibration`

Train-only scaling can combine with any branch after separate validation. SSL is a training objective on a selected encoder; it is not an inference adapter. Ensembles combine successful representations; adapters/calibrators are tested on one frozen branch first. Use four conditions (base/A/B/A+B) before claiming synergy.

Sinc and fixed filter banks are alternative frequency priors for the first comparison. CSP and learned spatial convolutions are alternative spatial estimators. MAE and JEPA are alternative pretraining objectives. Most importantly, whitening each window by **its own covariance** makes that covariance approximately identity, destroying E2's feature; trace/unit-variance normalization can similarly erase E1's power. A stable tangent reference also does not automatically confer invariance to different session-specific mixtures. Keep channel identities/coordinates consistent throughout.

## First implementation batch and submission decision

Implement the split/reservation and G1 checks, then **E1/E2/E3 with fresh controls**. Existing integration points are `track2/data.py` (splits), `submission.py` (models/reload), `run.py` and `track2/workflow.py` (model selection/training), plus smoke/export and contract tests. No fold runner or candidate decoder is implemented yet. Reuse the official scorer and one-file ZIP path; do not fork the harness. CPU work comes first; run the small neural finalists together on one GPU allocation after export/replay checks.

Freeze configuration before refitting on all ten public participants for a final sealed candidate; disclose that refit and retain development scores from before it. The [official timeline](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/timeline.md) opens sealed evaluation **5 November, 04:31 IST**, with five total submissions per participant/track and hidden logs. Use local evidence to choose candidates; an unaudited warm-up 0.90–1.00 is not a target score for generalization. No sealed score can be predicted from this review.

### Evidence used

Read the [requested prompt](../../prompts/next-step-analysis-prompt.md), [methods note](different-methods-research.md), [project methodology](../METHODOLOGY.md), current model/training/contract code, saved split inventory and verified runs. The note was found in `docs/research/`, not the dated subfolder. [Prior paper/code audit](2026-10-28/REPORT.md) supports the transfer cautions. Primary checks: [FBCNet](https://arxiv.org/abs/2104.01233), [ShallowConvNet](https://arxiv.org/abs/1703.05051), [Euclidean alignment](https://arxiv.org/abs/1808.05464), [tangent-space API](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html), [BTTA-DG](https://proceedings.iclr.cc/paper_files/paper/2026/hash/36eeec256f89eba9ed60cbca73ea92ed-Abstract-Conference.html), [masking comparison](https://arxiv.org/html/2609.33487v1), [NoMAD full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12089531/fullTextXML). None demonstrates a gain on this exact competition. No cloud resources, experiments or submissions were run; code-level falsification remains future work.
