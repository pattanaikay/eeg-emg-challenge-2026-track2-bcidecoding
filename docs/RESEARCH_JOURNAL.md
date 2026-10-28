# BCI Decoding EEG 2026 Challenge — research chronology

Reconstructed and last updated: **28 October 2026 (Asia/Kolkata, IST)**.

Working notes for a future blog post, not a polished narrative. Dates below use individual chat-turn timestamps or explicitly dated project records; file modification times alone are not treated as experiment dates. A proposal, a user-confirmed platform result and a saved experiment result are different kinds of evidence.

## 4 August 2026 — prior research direction

- **Prior proposal:** The dated science brief distinguishes benign coordinate changes from changes that damage usable task information. It proposes harmful-drift monitoring, limited adapter updates, preservation of useful computation and a vendor-neutral stability layer. These are research aims, not demonstrated competition results. Its initial preference for invasive longitudinal data was later superseded by the user's EEG-first choice. [Science brief][brief]

## 6 October 2026 — formulate the EEG transfer problem

- **User direction:** Work sequentially through problem formulation, pretraining/tokenization, adaptation and validation, with physiological grounding and unknowns stated explicitly. Neuralink's pretraining update was a reference, rather than a requirement to reproduce its scale. [Chat C1]
- **Discussion:** Distinguished changes in recorded coordinates, task information, label priors and the conditional decoding relationship. Scalp mixing, referencing, contact quality, muscle/eye activity and cognitive state were competing explanations; none was diagnosed as the cause of this dataset's failures. [Chat C1]
- **Historical assumption, subsequently corrected:** Early discussion used the advertised 47-signal, six-session description and an older guide's restriction on per-session recalibration. Those descriptions must not be presented as the implemented model-facing contract. See 20 October. [Chat C1; contract][contract]

## 7 October 2026 — learning needs and research priorities

- **User preference:** Asked for less dense explanations and HTML visualizations; chose to preserve electrode identity and scalp geometry, and deferred the temporal-prior decision until covariance was clearer. Peripheral quality-only use was discussed, but the user's “EEG/EOG” wording prompted an unanswered EMG/EOG clarification. The later contract supplies EEG only. [Chat C2]
- **User decision:** Selected “Track 2 EEG transfer first; use mechanistic analysis to guide it” and “Reduce labeled recalibration while limiting forgetting,” then explicitly added “Also predict harmful drift.” [Chat C3]
- **Research discussion:** Connected label-free adaptation, stable/drifting decoder subspaces and a possible vendor-neutral layer. Decoder ablation was distinguished from proof of a causal biological circuit; commercial interoperability remained unverified. The supplied sprint plan and clinical review informed feasibility discussion, not implementation instructions. [Chat C3]
- **Completed learning artifact:** Created the illustrated competition report with covariance, drift versus forgetting, recalibration cost, harmful-drift monitoring and a reading route. Its early benchmark description is historical and must be checked against the later contract. [Learning report][learning]; Chat C4

## 8 October 2026 — clarify longitudinal transfer

- **Question; response gap:** User asked what longitudinal transfer means. The accessible turn is marked interrupted, so it does not establish a completed explanation at that time. The later methodology document defines earlier-to-later decoding for the same person and separates it from unseen-participant transfer. [Chat C5; methodology][method]

## 20 October 2026 — start the executable competition baseline

- **User intent:** Having read the report, asked how to begin the project. A dedicated Track 2 implementation and submission workflow followed. [Chat C6–C7]
- **Contract correction implemented:** The project snapshot specifies **41 EEG channels × 480 samples at 120 Hz**, a four-second window beginning one second after the cue, and three labels: motor imagery, mental subtraction, word generation. No auxiliary EMG/EOG or participant/run identifiers reach prediction. Current/past-window label-free adaptation is allowed within a session, with a fresh model/state for each new session; extra labeled sealed calibration is not supplied. This corrected both the original montage/calibration assumption and the blanket adaptation restriction. These are historical verified snapshots, not a fresh 28 October rules audit. [Contract][contract]; [upstream revision][upstream]
- **Completed implementation/checks:** Constant packaging, channel-mean linear classification, EEGNet training/export paths and provenance records were built. The status record reports ten passing tests and synthetic training/replay/export checks. Synthetic results establish pipeline behavior, not EEG accuracy. [Status][status]
- **Confirmed local experiment:** Real PROTEUS audit found 10 participants, 14 sessions and 112 runs. Official preprocessing yielded **1,521 train / 379 validation / 755 test windows**, split by participant. The initial real linear model achieved **0.4458613 validation / 0.3796610 public-test cell-averaged balanced accuracy**, with exact checkpoint replay. It averages each channel over time; it is not a band-power or covariance baseline. [Status][status]
- **User-confirmed submission:** User reported that `warmup_constant.zip` submitted successfully and scored **0.33**. This is the only platform score confirmed in the evidence reviewed; it was not independently read back from Codabench. [Chat C7; status][status]

## 21 October 2026 — narrow scope, study the papers, run real EEGNet

- **User decision and reasoning:** “I want to work on the basis of the challenge. The long term goal for a novel approach should be treated as an inspiration.” Competition baselines became the immediate priority; harmful-drift forecasting, mechanistic novelty and a general adaptation platform remained later research aims. [Chat C8]
- **Completed learning artifacts:** Created the literature review and then the HTML paper explainers covering 12 scientific papers and two company reports. They connect methods to our task and distinguish teaching examples from experimental observations. The study guide labels author-perspective prose as reconstruction. These are reading aids, not replications. [Literature review][literature]; [paper explainers][explainers]; Chat C9–C10
- **Compute proposal:** Recommended a single L4 for the small baseline models and prepared a CLI handoff covering real-data training, replay, export, retrieval and verified pause. Inspected EEGNet had 2,483 trainable parameters; the recommendation did not require an H100. [Handoff prompt][prompt]
- **Confirmed execution in project records:** A user-created L4 instance ran linear seed 33 and EEGNet seeds 33–35 sequentially. All four jobs completed; validation selected checkpoints. Saved-checkpoint validation/test metrics replayed exactly, remote submission ZIP inference checks passed, and 169 downloaded file checksums/sizes matched. A Windows launcher compatibility repair enabled execution; it was not a decoder-method change. [Run log][runlog]; [verification JSON][verified]
- **Confirmed lifecycle, historical only:** The instance was verified paused at **22:15:25 IST on 21 October** after retrieval. This is not evidence of its present state or an invoice. Local runtime revalidation of the downloaded ZIPs was unavailable in that recorded environment; remote inference and local byte/checksum verification passed. [Run log][runlog]

Actual fixed-split results from that batch:

| Model | Initialization seed | Selected epoch | Validation cell BA | Public-test cell BA |
|---|---:|---:|---:|---:|
| Channel-mean linear | 33 | 2 | 0.445861335 | 0.379661043 |
| EEGNet | 33 | 20 | 0.453858647 | 0.352390098 |
| EEGNet | 34 | 7 | 0.454217070 | 0.382836157 |
| EEGNet | 35 | 17 | 0.467058692 | 0.363920946 |

- **Interpretation:** EEGNet's three-run public-test mean was **0.366382400** (36.64%). It had not demonstrated a consistent advantage over this linear reference. These seeds change initialization on **one fixed participant split**; they are not participant folds. Scores are local public-data evaluations, not Codabench scores or sealed-test results. [Verification JSON][verified]

## 24 October 2026 — inspect the data and turn reading into diagnostic experiments

- **Completed inspection/explanation:** Opened the NEMAR signal viewer and explained channels, sampling and window duration: 120 samples/second × four seconds = 480 values per channel. Distinguished raw 500 Hz acquisition, the viewer's 250 Hz preview and the 120 Hz model tensor. A viewer screenshot was saved. [Viewer screenshot][viewer]; Chat C11
- **Confirmed diagnostic observation:** The saved EEGNet seed-35 confusion matrix gives subtraction recall **61/252 = 24.21%**, and word-generation predictions **416/755 = 55.10%**. This suggested checking class bias; it did not identify its cause. [Confusion matrix][confusion]
- **Proposed next experiments, not completed:** Rotate held-out participants; compare explicit band-power and covariance features with EEGNet; then test normalization/alignment as a separate change. Explain cross-participant folds separately from earlier-to-later session tests. The core reading question became what task information a method preserves and whether it survives a change of person/session. [Chat C11–C12]
- **Interpretation clarified:** A frozen EEGNet cannot suffer weight-update forgetting during inference; weak transfer alone does not diagnose catastrophic forgetting. [Chat C12; methodology][method]

## 25 October 2026 — fixed-dataset constraint and adversarial triage

- **User constraint and reasoning:** Explicitly described a fixed, single-corpus setting, with no new subjects or 50,000-hour collection. Asked to focus on feature extraction, context conditioning and statistical modeling, and to scrutinize scale dependence, leakage and unnecessary complexity. [Chat C13]
- **Working interpretation:** The reply treated externally pretrained weights as additional data under that constraint; borrowing architectural ideas remained permissible. This was an experimental-scope interpretation, not a claim that competition rules ban external public pretraining. [Chat C13]
- **Research review, not a new experiment:** Separated Neuralink's event/channel tokenization, Perceiver input, Mamba dynamics and spatially masked Poisson prediction from the scale of its recordings. Noted that reported live results used participant-specific pretraining and that pooled models had not improved online performance. The proposed EEG transfer changes tokenizer/likelihood; it does not treat scalp voltages as spikes. [Chat C13; Neuralink source](https://neuralink.com/updates/pretraining-on-50000-hours/)
- **Proposed priorities:** Constrained frequency extraction, spatially meaningful inputs, shrinkage covariance/regularized classifiers, causal statistical alignment and modest augmentation. Recent BTTA-DG, REVE, masking, NoMAD and augmentation/alignment studies were discussed as evidence or inspiration with modality/protocol limits. No reproduction or PROTEUS gain was established. [Chat C13]
- **Validation decisions discussed:** Do not use whole-future-session normalization, overlapping-window splits, unavailable IDs/task labels, or repeated test/leaderboard tuning. Use the official equal-cell balanced-accuracy metric. These were evaluation requirements and cautions, not allegations that a completed run used those shortcuts. [Chat C13]

## 28 October 2026 — establish this journal

- **Completed documentation:** Reconstructed this chronology from 16 accessible chat turns spanning 6–28 October and the cited local records, including raw saved result summaries/confusion data. Earlier background is limited to the dated science brief and the user's account; residency experiments were not independently reconstructed.
- **Maintenance instruction:** Update this same file when meaningful work/discussion occurs in this chat or when asked. Add concise dated bullets with evidence and status. Keep historical proposals/results distinct, date corrections explicitly, preserve existing material, and omit credentials and raw EEG. This is turn-based maintenance, not continuous monitoring of other chats, cloud resources or outside activity.

## Open evidence gaps

- No trained-model Codabench submission or sealed score was established by the reviewed evidence. The 0.33 constant score is user-confirmed only.
- No completed participant-fold, band-power, covariance, same-corpus SSL, test-time adaptation, forgetting, harmful-drift forecasting or mechanistic-ablation experiment was found in the records inspected. This does not establish that no such work exists elsewhere.
- The available repeated-session public data are limited to two participants; broad longitudinal claims require more evidence than the fixed public participant split supplies.
- Current cloud state, charges, leaderboard standing and post-25-October rule changes were not checked. Historical schedules/prices should not be reused as live facts.

## Evidence index

Chat source: **BCI Decoding EEG 2026 Challenge**, thread `01a11248-c4e5-77a3-91f3-ca23cf4d7bb8`. Individual turn timestamps were retrieved on 28 October. The thread-level creation date is later than some preserved turns; it was not used to date those earlier discussions.

| Chat reference | IST date/time | Turn ID / topic |
|---|---|---|
| C1 | 6 Oct 23:05 | `01a11248-c79f-7751-87d7-baf5ce690376` — Phase 1 |
| C2 | 7 Oct 09:28 | `01a11483-86c6-7713-8036-9657cc6dc6d7` — explanation needs |
| C3 | 7 Oct 16:17 | `01a115f9-a9d1-7271-bfe6-a35ee44258ec` — research priorities/replies |
| C4 | 7 Oct 17:24 | `01a11637-5198-7d73-b4c3-b301a752eae7` — HTML report |
| C5 | 8 Oct 12:28 | `01a11a4e-9b8b-77e1-a77d-f387a63fb538` — longitudinal question, interrupted |
| C6 | 20 Oct 21:27 | `01a15a08-71f4-7a91-9396-08111a82c69c` — begin project |
| C7 | 20 Oct 21:44 | `01a15a17-efa7-7482-af9c-095b91c45aa4` — implementation/submission reply |
| C8 | 21 Oct 11:29 | `01a15d0b-55c6-7663-9f49-95b8cddffef9` — challenge-first scope |
| C9 | 21 Oct 10:56 | `01a15cec-cd78-76f2-8241-31ffbaf9cebf` — literature review |
| C10 | 21 Oct 18:30 | `01a15e8c-a2de-77e2-9598-94e2f0d29bcd` — paper explainers |
| C11 | 24 Oct 11:12 | `01a16c6e-b91a-7d10-9d4b-d7ef73a4c471` — results/data viewer |
| C12 | 24 Oct 11:47 | `01a16c8e-ba21-7eb3-a476-edd59eda5108` — folds/bottleneck |
| C13 | 25 Oct 09:51 | `01a1714a-aa5a-7e01-8f40-5c3ac8b51d36` — fixed-data survey/triage |

[brief]: C:/Projects/Brain2Text2025/BCI_DRIFT_SCIENCE_AND_PARTNERSHIP_BRIEF.md
[learning]: C:/Projects/Brain2Text2025/output/track2-bci-drift-learning-report.html
[literature]: C:/Projects/Brain2Text2025/output/track2-literature-review.md
[explainers]: C:/Projects/Brain2Text2025/output/track2-paper-explainers.html
[prompt]: C:/Projects/Brain2Text2025/output/track2-jarvislabs-codex-cli-prompt.md
[viewer]: C:/Projects/Brain2Text2025/output/nemar-proteus-signal-viewer.png
[status]: C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/STATUS.md
[contract]: C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/COMPETITION_CONTRACT.md
[method]: C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/METHODOLOGY.md
[upstream]: C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/UPSTREAM.json
[runlog]: C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/JARVISLABS_RUN_LOG.md
[verified]: C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/runs/jarvis/20261021T162400Z/verified-results.json
[confusion]: C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/runs/jarvis/20261021T162400Z/remote/runs/eegnet-seed35-20261021T164141Z/test_confusion.json


## 28 October 2026 — five-agent evidence and competition-fit review

- **Completed literature investigation, not an experiment:** Coordinated five agents around the question of which task information a method preserves, which recording variation it tolerates, and whether evaluation uses only competition-available information. Reviewed BTTA-DG, REVE, the channel-adaptation preprint, the masking-geometry preprint, NoMAD and trust-gated augmentation, including primary methods and selected release code. [Full report](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/research/2026-10-28/REPORT.md)
- **Contract recheck:** Current README/FAQ/evaluator retain the 41×480 EEG, chronological single-window, session-reset and equal-cell balanced-accuracy contract. A material unresolved conflict remains: repository Terms/timeline give five warm-up submissions/day and Terms give 60 minutes; the website gives ten/day and 80 minutes. Active platform enforcement was not inspected. External public pretraining is officially permitted subject to licensing/disclosure, distinct from our fixed-corpus preference.
- **Major evidence limits:** BTTA-DG's pinned release uses target-label checkpoint selection and whole-target alignment, and its shown adaptation path does not populate memory. Other reviews found paper/release preprocessing, gate and data-partition discrepancies, heterogeneous transfer protocols and incomplete target-statistics chronology. These are static source/provenance findings, not proof about every published experiment or allegations of misconduct. NoMAD's monkey whole-session alignment and human preceding-block analysis were kept distinct.
- **Recommendations, not user decisions:** Proposed matched representation controls, isolated constrained-frequency learning, causal statistical normalization and modest augmentation as candidates. Full SSL, dynamics alignment and GMM adaptation were deferred in the recommendation. **The user has not yet selected a next experiment in this review.**
- **Execution/accounting boundary:** No decoder code was changed, no model or GPU experiment was run, and no submission was made. Reports and this journal were the only project-document changes. Per-agent, coordinator and combined token counts were unavailable; no hard combined cap was verified. Published-paper results remain unreplicated here.

