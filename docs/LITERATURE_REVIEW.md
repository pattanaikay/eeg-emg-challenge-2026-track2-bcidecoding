# Track 2 literature review: stable decoding, cautious adaptation, and harmful-drift prediction

Research review for the EEG/EMG Foundation Challenge 2026. Prepared 21 October 2026 from the current `docs/READING.md` in the Track 2 repository. Numbers below are reported results unless explicitly labeled as a derivation, proposal, or local project result.

## How to use this review

The reading list contains **12 scientific papers**, including one clinical perspective/review; **two company research reports**; and a masked-pretraining tutorial plus supporting software/data documentation. All 12 papers receive the requested four-part breakdown. The company reports receive the same treatment but retain their different evidence status. Supporting documentation is covered separately rather than treated as experimental research.

**Verification boundary:** full primary texts were inspected for MINDFUL, EWC, EEGNet, LaBraM, EATA, NoMAD, Haufe, PSID, DPAD, and the attached Saez review. For **RPA**, the primary abstract, author repository, and transformation implementation were accessible, but the complete journal tables were not. For **Nolte**, the institutional primary abstract was accessible; the full experimental methods were not independently verified. These two entries explicitly identify what remains unverified. No inaccessible accuracy table is reconstructed from secondary summaries.

Each entry separates **what the authors demonstrate** from **reviewer analysis for our project**. Algebraic examples and complexity estimates marked as reviewer derivations are explanatory results, not extra experiments from the paper. Absence of a latency or GPU-hours figure means “not verified here,” not zero cost. Percentages, percentage-point differences, AUROC, correlation, R², and communication throughput are different quantities and must not be pooled into a ranking.

The central research opportunity is to connect three questions while measuring each separately: **Does the decoder still work? Does an update help without damaging useful behavior? Can we identify impending harm early enough to act?** None of the reviewed sources establishes all three for the present scalp-EEG benchmark.

## Project setting and a shared mathematical vocabulary

The current official contract supplies **41 EEG channels**, one chronological `(1, 41, 480)` window at a time, representing four seconds at 120 Hz. The public release contains 10 participants and 14 sessions, not the earlier 47-signal calibration scenario. Sealed inference provides no extra labeled calibration, and each participant-session begins from a fresh model copy. Keep this distinction when adapting papers that assume labeled target calibration or whole-target-session access. [Official scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md).

For participant u and session s, let X be an EEG window and Y its mental-command label. A useful observation model is:

```text
X(u,s) = A(u,s) S_task + B(u,s) S_other + E(u,s).
P(u,s)(X,Y) = P(u,s)(Y|X) P(u,s)(X).
```

This is a **reviewer abstraction**, not an identifiable decomposition of the recorded EEG. A represents spatial mixing and reference effects; B and E include other neural activity, muscle/eye contamination and acquisition noise. Cap placement, contact impedance, reference, cognitive strategy and vigilance can change their contributions. It does not follow that every observed change reflects neural reorganization.

| Term | Meaning in this project | What would establish it? |
|---|---|---|
| Covariate shift, strictly defined | P(X) changes while P(Y\|X) stays fixed | A modeling assumption to test; changing EEG histograms alone cannot establish the unchanged conditional |
| Conditional/concept drift | P(Y\|X) changes | Labels or additional assumptions are generally needed to identify it |
| Longitudinal transfer | Earlier recordings support later recordings of the same person | Chronological held-out sessions; a random within-recording split is insufficient |
| Cross-participant transfer | Generalization to a person absent from training | Participant-disjoint evaluation |
| Forgetting/negative adaptation | Updating parameters worsens previously useful performance | Same held-out reference evaluated before and after the update |
| Drift detection | Recognizing a change that is present now | A defined detection target and false-alarm measure |
| Harmful-drift forecasting | Predicting later loss of useful decoding from information available now | A predeclared future horizon, held-out failures, and temporal separation |

A small impossibility example explains why the monitor is a separate scientific claim. Suppose X is equally likely to be −1 or +1. Initially Y=1 when X=+1; later the labels swap. **P(X) is identical, but a previously perfect decoder can become entirely wrong.** This is a reviewer counterexample: an unlabeled marginal-shift score cannot universally detect every harmful conditional change. Useful warning is possible under restricted, empirically validated conditions.

## Paper 1 — Riemannian Procrustes Analysis (RPA)

**Rodrigues, Jutten & Congedo, IEEE TBME, 2019 issue / online 2018.** [Primary record](https://pubmed.ncbi.nlm.nih.gov/30596565/), [author repository](https://github.com/plcrodrigues/RPA), [author transformation code](https://github.com/plcrodrigues/RPA/blob/master/rpa/helpers/transfer_learning/utils.py). Evidence is access-limited as described above.

### 1. Core Hypothesis & Problem Statement

EEG covariance distributions differ across people and sessions, making useful source training data difficult to reuse. The hypothesis is that part of this mismatch can be removed by geometrically appropriate recentering, rescaling and rotation of symmetric positive-definite (SPD) features. The contribution concerns matching distributions; it is not a proof that all biological drift is an invertible spatial transform.

### 2. Architectural & Methodological Innovations

For a centered C-channel window with T samples, a covariance descriptor is `C_x = XXᵀ/(T−1)`. Positive definiteness must be checked; shrinkage or another regularizer is an implementation choice when the estimate is singular or poorly conditioned. The affine-invariant distance is `d_R(C₁,C₂) = ||log(C₁⁻¹ᐟ² C₂ C₁⁻¹ᐟ²)||_F`.

The author code implements:

```text
Recenter: C̃ = G⁻¹ᐟ² C G⁻¹ᐟ², with G a domain Riemannian mean.
Dispersion: d = mean_i d_R(C̃_i, I)².
Stretch target: C̃_target → C̃_target^sqrt(d_source/d_target).
Rotate target: C̃_target → R C̃_target Rᵀ.
```

Class-wise source/target means determine the rotation. **The target rotation implementation uses target training labels.** Covariance features and manifold distances predate RPA; the distinctive contribution is the coordinated Procrustes-style domain matching. [Transformation implementation](https://github.com/plcrodrigues/RPA/blob/master/rpa/helpers/transfer_learning/utils.py).

**Reviewer interpretation:** covariance compresses a window into channel powers and cofluctuations. It retains no general ordering of events: centered X and a common time-permutation XΠ have the same XXᵀ. This is why covariance can be robust to waveform timing yet cannot replace a temporal model for every task. At 41 channels, the symmetric matrix has **861 unique entries**; that is an algebraic dimension count, not 861 independent biological connections.

### 3. Key Quantitative Results & Benchmarks

The primary abstract reports simulated data plus **eight public BCI datasets, three paradigms, 243 subjects**, with improvements over geometry-aware transfer comparators and benefits to ensembles. **Exact original per-dataset effect sizes, uncertainty intervals, and wall-clock costs remain unverified.** The author repository supplies MI and SSVEP demonstrations; these are not a substitute for the original result tables. [Primary abstract via HAL API](https://api.archives-ouvertes.fr/search/?q=halId_s:hal-01971856&fl=title_s,abstract_s,uri_s&wt=json).

### 4. Critical Evaluation & Limitations

**Reviewer analysis:** recentering can correct a useful nuisance displacement but can also erase task information if class prevalence changes. If the average changes because the user performs more of one task, treating the whole difference as sensor drift is unsafe. Covariance estimates from autocorrelated samples have fewer effective observations than their raw sample count suggests. Eigensolvers scale roughly cubically in channel count per decomposition; this is an algorithmic cost estimate, not a measured RPA latency.

Full class-wise RPA is not directly eligible for label-free sealed target adaptation. A causal recentering-only reference is a defensible first experiment, explicitly named as a restricted variant. Fit source statistics on training data; estimate target statistics using only permitted current/past windows. Compare no alignment, recentering, and any further permitted transform separately. Use the same reference and class-mix controls before claiming physiological invariance.

The author helper also scores one-hot hard predictions using ROC-AUC. That implementation detail is another reason to reproduce the current competition's cell metric rather than comparing an old repository score directly to our result.

## Paper 2 — MINDFUL: monitoring instability associated with performance

**Pun et al., Communications Biology, 2024.** [Full primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11494208/), especially Results, Fig. 2 and Methods.

### 1. Core Hypothesis & Problem Statement

A deployed intracortical decoder may deteriorate when user intention is unavailable as a label. MINDFUL tests whether changes in observed neural-feature distributions can nevertheless track deterioration well enough to inform recalibration. It is a monitoring method rather than a decoder adaptation algorithm.

### 2. Architectural & Methodological Innovations

The study uses fixed kinematic decoders and 20-ms binned threshold-crossing features, with spike power for one participant. Reference samples are drawn from early good-performance periods. Comparison distributions are updated with **60-second sliding windows**, with neural PCA features and alternative decoder-output features. KL divergence measures reference/current distribution mismatch. The strongest configuration combines reduced neural inputs, decoded velocity and a one-bin lag of decoded velocity. [Methods and Fig. 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11494208/).

The paper uses the Gaussian KL closed form on empirical first and second moments:

```text
D_KL(N₀ || N₁) = ½[tr(Σ₁⁻¹Σ₀) + (μ₁−μ₀)ᵀΣ₁⁻¹(μ₁−μ₀)
                         − d + log(det Σ₁ / det Σ₀)].
```

This expression explains why both shifts in the mean and changes in spread matter; KL direction must be specified. As a moment-based score it can be useful without truly Gaussian data, but it does not capture every higher-order distribution change. Covariance regularization still needs checking. The main combined configuration uses five neural PCs plus two current and two lagged velocity components: nine dimensions in total.

### 3. Key Quantitative Results & Benchmarks

| Evaluation | Reported result |
|---|---|
| Two BrainGate2 participants, fixed decoders | Longitudinal spans of **142 and 28 days** |
| Neural features only versus contemporaneous median angle error | Pearson **r=0.91 and 0.59** |
| Neural features + decoded velocity + lagged velocity | Pearson **r=0.926 and 0.719**, rounded in the abstract to 0.93/0.72 |

These are correlations with continuous cursor angle error, not command balanced accuracy, forecasting AUROC, or proof of an intervention benefit. [Fig. 1c and Fig. 2c](https://pmc.ncbi.nlm.nih.gov/articles/PMC11494208/).

### 4. Critical Evaluation & Limitations

**Documented boundary:** the online score does not need current intention labels, but selecting low-angle-error reference samples in this experiment uses performance information. Adaptive mean/bias corrections were already part of online operation; “fixed decoder” does not mean an entirely unchanging signal-processing pipeline.

**Reviewer analysis:** sliding windows overlap heavily, so many score/error pairs are not independent subjects or failures. Two participants provide a strong motivating case, not a population-level EEG validation. A monitor can respond to changed task use or output distribution without a recording fault. If the decoder adapts, its own output distribution also moves, confounding a monitor that uses those outputs.

For our study, compare a frozen-reference monitor against signal-quality scores, entropy, and generic covariance distance. Evaluate **current deterioration** separately from **future failure**. A 60-second window is not automatically suitable for short EEG task streams; its horizon and evidence requirements need validation. Preserve a fixed teacher/reference during monitoring so an adaptive model cannot silently redefine what counts as normal.

For a d-dimensional moment score, covariance formation and inversion introduce roughly O(Nd²) and O(d³) work, respectively. These reviewer complexity estimates explain the appeal of the reduced feature set; they are not a measured end-to-end latency or a guarantee that a shorter window provides enough samples.

## Paper 3 — Elastic Weight Consolidation (EWC)

**Kirkpatrick et al., PNAS, 2017.** [Author manuscript](https://arxiv.org/abs/1612.00796), [full PDF](https://arxiv.org/pdf/1612.00796), especially Eq. 3 and Figs. 2–3. The inspected arXiv version is v2.

### 1. Core Hypothesis & Problem Statement

Sequential gradient updates can overwrite parameters needed for earlier tasks. EWC hypothesizes that retaining old expertise requires protecting important parameters more strongly than unimportant ones, while allowing sufficient plasticity for the next task.

### 2. Architectural & Methodological Innovations

EWC approximates the previous parameter posterior locally with a diagonal Gaussian. A Fisher-based importance estimate gives the penalty:

```text
L_total(θ) = L_new(θ) + (λ/2) Σ_i F_i (θ_i − θ_i*)².
F_i ≈ E[(∂ log pθ(Y|X) / ∂θ_i)²] evaluated near θ*.
```

The distinctive ingredient is importance-weighted consolidation rather than uniform distance from old weights. It requires the old parameter anchor and importance estimates. The Atari system also uses task recognition, task-specific gains/biases and short-term replay; its results must not be credited to the quadratic penalty alone. [Full manuscript, Methods and Atari experiment](https://arxiv.org/pdf/1612.00796).

**Reviewer interpretation:** this is a penalty on changing the model, not a monetary or clinical “cost of recalibration.” A separate deployment objective could charge for labeled trials, user time or latency. Those costs need explicit units and cannot be replaced by an arbitrary λ. The diagonal approximation measures parameter sensitivity in one coordinate system; it is not a direct measure of preserved cognitive information.

### 3. Key Quantitative Results & Benchmarks

The paper studies permuted MNIST and sequential sets of **10 Atari games**. Its Atari summary clips each human-normalized game score at 1, giving a maximum total of 10; the plain sequential-gradient comparator remains below 1 while EWC learns multiple games. EWC consolidation begins only after **20 million frames per game**. The paper also reports a gap to training separate networks. Exact plotted endpoint scores are not transcribed here. [Fig. 3 and text](https://arxiv.org/pdf/1612.00796).

### 4. Critical Evaluation & Limitations

The authors identify limitations of the parameter-uncertainty approximation. **Reviewer analysis:** local quadratic protection can be inaccurate after large updates; correlated parameter directions are omitted, and a confidently wrong model can protect the wrong computation. Maintaining separate anchors/Fishers for many tasks adds storage; approximately O(P) per retained task for P parameters, before optimizer state. Aggregating them is a different design with different assumptions.

EEG sessions do not necessarily correspond to distinct semantic tasks. If a session changes the observation map, a successful adjustment may require altering parameters that were important before. Excessive protection can therefore preserve the very mismatch we need to remove. Compare no update, unconstrained update, uniform anchoring and Fisher anchoring with identical update budgets. Measure both later-session gain and reference retention. Under the competition reset rule, cross-session accumulated EWC state cannot simply be carried from one test visit into the next.

## Paper 4 — EEGNet: a compact spatial–temporal reference

**Lawhern et al., Journal of Neural Engineering, 2018.** [Primary manuscript](https://arxiv.org/abs/1611.08024), [full PDF](https://arxiv.org/pdf/1611.08024), especially Tables 1–3.

### 1. Core Hypothesis & Problem Statement

EEG pipelines were often tailored to a single paradigm, while large general CNNs risked overfitting small datasets. EEGNet asks whether one compact architecture can express useful EEG features across both evoked-potential and oscillatory paradigms.

### 2. Architectural & Methodological Innovations

The network first learns temporal filters, then depthwise spatial filters spanning channels for each temporal feature map. Separable temporal/pointwise processing summarizes and combines maps, followed by pooling and classification. Batch normalization, nonlinearities, dropout and norm constraints support training. The innovation is the EEG-specific factorization and compact use of depthwise/separable convolutions, not the invention of convolution or spatial filtering. [Architecture, Table 2](https://arxiv.org/pdf/1611.08024).

**Reviewer interpretation:** a temporal filter can emphasize a frequency range; a following spatial filter can weight the scalp pattern in that range. This is a learned analogue of filter-bank spatial processing, not a guarantee that a feature is cortical rather than ocular or muscular. Channel order matters, but a learned weight at an index is not an explicit model of electrode coordinates. “General architecture across paradigms” also differs from “one pretrained weight set transferred unchanged across paradigms.”

### 3. Key Quantitative Results & Benchmarks

The datasets cover P300, ERN, MRCP and SMR, with **15, 26, 13 and 9 subjects**, respectively. Comparators include DeepConvNet, ShallowConvNet and paradigm-specific traditional methods.

| Table 3 example | EEGNet-8,2 parameters | DeepConvNet parameters |
|---|---:|---:|
| P300 | **2,258** | **174,127** |
| SMR | **1,716** | **152,219** |

These counts belong to the paper's inputs/configurations, not automatically to our 41-channel implementation. Performance comparisons use different metrics across paradigms; there is no single universal “EEGNet accuracy.” [Tables 1 and 3, Results](https://arxiv.org/pdf/1611.08024).

An interpretation ablation in one P300 fold reports AUROC **0.9054** with all filters and **0.7108** after removing filters 1, 3 and 4. This shows model dependence in that example; it is not a multi-subject intervention estimate. [Table 4](https://arxiv.org/pdf/1611.08024).

### 4. Critical Evaluation & Limitations

**Reviewer analysis:** fewer parameters reduce one source of overfitting and deployment cost but do not guarantee longitudinal robustness. Temporal kernel duration depends on sample rate; a kernel configured for another rate has a different physical meaning. Pooling can remove timing detail. Batch-normalization behavior at one-window inference must be deliberately controlled; updating statistics is itself an adaptation intervention.

This is our essential supervised reference before expensive SSL. Record the exact Braindecode version, architecture settings and channel order. Compare frozen inference first; then test normalization changes separately. For explanation, transform linear spatial filters into activation patterns where appropriate and inspect artifact sensitivity. A neat scalp map or a compact model does not establish a causal neural mechanism.

## Paper 5 — LaBraM: EEG-specific self-supervised token learning

**Jiang, Zhao & Lu, ICLR, 2024.** [Full primary text](https://arxiv.org/html/2405.18765v1), [PDF](https://arxiv.org/pdf/2405.18765), [author code](https://github.com/935963004/LaBraM).

### 1. Core Hypothesis & Problem Statement

EEG datasets vary in montage, duration, sampling and task, restricting reusable representations. LaBraM proposes a common channel-patch representation and a learned discrete target so masked pretraining can combine heterogeneous EEG collections.

### 2. Architectural & Methodological Innovations

Signals become per-channel temporal patches with spatial and temporal embeddings. A vector-quantized tokenizer learns codes by reconstructing Fourier amplitude and phase. A Transformer then predicts the discrete codes of masked patches, with complementary/symmetric masking. The paper uses **one-second patches at 200 Hz**, up to 256 patches per example. The distinctive element is the spectrum-trained tokenizer plus masked-code learning; Transformer blocks and VQ techniques are established ingredients. [Methods §§2–3](https://arxiv.org/html/2405.18765v1).

```text
Tokenizer code: z_i = argmin_j ||normalize(p_i) − normalize(v_j)||².
Masked objective: L_mask = −Σ_(masked i) log p(z_i | visible patches).
```

**Reviewer interpretation:** predicting a code differs from reproducing every waveform sample. It may encourage a more learnable target under noisy EEG, but the codebook can still represent equipment signatures or artifacts. A lower masked loss does not prove that the representation is invariant to session drift. Electrode identity embeddings support montage handling; they are not automatically a physically faithful inverse solution.

### 3. Key Quantitative Results & Benchmarks

Pretraining uses about **2,500 hours**; Base/Large/Huge have **5.8M/46M/369M** parameters. The main comparison excludes downstream datasets from pretraining and reports five-seed means/standard deviations.

| Benchmark | BIOT balanced accuracy | LaBraM Base | LaBraM Huge |
|---|---:|---:|---:|
| TUAB | 0.7959±0.0057 | **0.8140±0.0019** | **0.8258±0.0011** |
| TUEV | 0.5281±0.0225 | **0.6409±0.0065** | **0.6616±0.0170** |

Base's TUEV difference is **11.28 percentage points**, an arithmetic comparison, not 11.28% relative improvement. These are clinical EEG tasks, not PROTEUS command decoding. [Tables 1–2](https://arxiv.org/html/2405.18765v1).

The tokenizer ablation is task-dependent: raw-patch reconstruction reaches TUEV balanced accuracy **0.5630±0.0313** versus **0.6409±0.0065** for LaBraM, but reaches TUAB **0.8261±0.0030**, above LaBraM's **0.8140±0.0019**. The discrete target is therefore not uniformly best. [Appendix G, Table 7](https://arxiv.org/html/2405.18765v1).

### 4. Critical Evaluation & Limitations

The inspected setup uses eight A800 GPUs; a standardized total GPU-hours budget is not established here. Scaling curves are not uniformly monotonic: the Base model with 500 hours can exceed its 2,500-hour TUAB result. The paper's extrapolation to larger data is a hypothesis, not a validated universal scaling exponent. [§§3.2, 3.6](https://arxiv.org/html/2405.18765v1).

**Reviewer analysis:** at our 120 Hz, a 200-sample patch would last 1.67 seconds, so copying the sample count would change the temporal prior. Reusing weights needs a compatible sampling/patch/normalization path and verification of permitted preprocessing. Self-attention costs grow quadratically with token count; channel independence at tokenization does not remove cross-channel compute. Pretraining datasets require overlap/license checks, particularly when studies have multiple releases. First compare frozen features, then limited fine-tuning, against EEGNet at matched label budgets. Test whether pretrained embeddings improve worst-cell performance and retention, not just average accuracy.

## Paper 6 — EATA: filtered adaptation with anti-forgetting protection

**Niu et al., ICML/PMLR, 2022.** [Primary proceedings record](https://proceedings.mlr.press/v162/niu22a.html), [full paper](https://proceedings.mlr.press/v162/niu22a/niu22a.pdf), especially Eqs. 3–9 and Tables 2–3.

### 1. Core Hypothesis & Problem Statement

Test-time entropy minimization can waste updates on unreliable/redundant samples and damage performance on familiar data. EATA tests whether filtering update samples and protecting important weights can improve both efficiency and retention.

### 2. Architectural & Methodological Innovations

Only low-entropy samples sufficiently different from a moving average of prior predictions contribute to adaptation. A Fisher regularizer anchors selected parameters to the original model; the experiments update batch-normalization affine parameters. Importance is estimated once from a small **unlabeled in-distribution sample set** with hard pseudo-labels. This combines sample selection with EWC-like protection; neither entropy minimization nor Fisher anchoring originates here. [Methods](https://proceedings.mlr.press/v162/niu22a/niu22a.pdf).

```text
S(x) = exp(E₀ − H(pθ(x))) · 1[H(pθ(x)) < E₀]
                           · 1[cos(pθ(x), m_previous) < ε].
L = mean_x S(x) H(pθ(x)) + β Σ_i ω_i (θ_i − θ_i⁰)².
```

Here m is a moving average of selected predictions. This is selective sharpening, not a calibrated estimate that the predicted label is correct. The reference Fisher set's origin matters: “source-data-free at adaptation” does not mean “no reference-distribution assumptions anywhere.”

### 3. Key Quantitative Results & Benchmarks

| Setting, ResNet-50 | Base | Tent | EATA |
|---|---:|---:|---:|
| ImageNet-C severity 5, mean corruption error (%) | 68.4 | 47.7 | **45.0** |
| Table 2 backward-example count | 0 | 50,000 | **25,150** |
| ImageNet-R error (%) | 63.8 | 57.7 | **54.8** |

The implementation uses 2,000 samples for Fisher estimation, adding forward/backward work. Counts are the paper's compute proxy, not a measured EEG wall-clock speedup. ETA without the Fisher penalty reaches 54.5% error on ImageNet-R, illustrating that retention protection need not maximize shifted-domain accuracy. [Tables 2–3 and implementation details](https://proceedings.mlr.press/v162/niu22a/niu22a.pdf).

### 4. Critical Evaluation & Limitations

**Reviewer analysis:** a confidently wrong prediction passes the entropy gate. Redundancy filtering can reject sustained but valid task behavior; in a three-class, one-window stream, the prediction simplex and temporal class runs differ from large image batches. Paper thresholds, batch size and learning rate are not EEG defaults. A source-distribution reference must be constructed without sealed labels or future windows; a training-derived importance estimate would be a declared adaptation of the method.

Its Fisher penalty cannot prevent every collapse, especially when importance is estimated under model error. Benchmark corruption changes are not a substitute for electrode mixing, cognitive strategy changes or class-prevalence shifts. In Track 2, first measure an update restricted to a small parameter subset, with a gradient budget, session reset and a frozen comparator. Test the gate and regularizer independently. A reduction in backward passes is valuable only if required predictions still arrive within the inference budget and accuracy/retention remain acceptable.

## Paper 7 — NoMAD: aligning latent dynamics across recording conditions

**Karpowicz et al., Nature Communications, 2025.** [Primary article](https://www.nature.com/articles/s41467-025-59652-y), [open manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC12089531/), especially Figs. 2–5 and Methods.

### 1. Core Hypothesis & Problem Statement

Static manifold alignment ignores how neural activity evolves. NoMAD hypothesizes that stable latent dynamics can constrain alignment when the observed neural population changes, allowing a previously trained behavioral decoder to remain useful.

### 2. Architectural & Methodological Innovations

Day-0 LFADS supplies a recurrent latent dynamics model. Later-session alignment trains an input/alignment pathway and selected readouts while holding the core dynamics fixed, using spike reconstruction and distribution matching of generator states. A fixed Wiener decoder reads the aligned representation. The alignment KL assumes multivariate Gaussian state distributions. It is not simply PCA plus rotation. [Methods, alignment model](https://www.nature.com/articles/s41467-025-59652-y).

```text
L_align = α · Poisson_NLL(observed spikes, reconstructed rates)
                           + β · D_KL(N_day0 || N_dayK).
```

**Reviewer interpretation:** temporal consistency can rule out some solutions that merely match a static cloud of points. It cannot identify every correct class correspondence. Spike counts justify a Poisson observation model more naturally than signed scalp voltages do; using the same likelihood on raw EEG would be an unjustified modality transfer.

### 3. Key Quantitative Results & Benchmarks

| Offline longitudinal task | NoMAD | Key comparator results |
|---|---|---|
| Isometric force: 20 sessions, 95 days, 380 directed session pairs | Median **R²=0.91**, no negative-R² failures | Aligned FA 0.59; ADAN 0.65; static decoder 223 failures |
| Reaching: 12 sessions, 38 days, 132 directed pairs | Median **R²=0.78**, no negative-R² failures | Aligned FA 0.17; ADAN 0.29 |
| Two human T11 sessions, offline replay | NoMAD + retrospective intention inference: median **R²=0.72/0.83** | NoMAD alone 0.66/0.79 |

The paper also fits stability half-lives; these are modeled decay summaries, not observations of successful deployment for the entire extrapolated duration. [Figs. 3–5](https://www.nature.com/articles/s41467-025-59652-y).

### 4. Critical Evaluation & Limitations

**Documented boundary:** the main longitudinal alignment uses target-session data offline. Past-only sliding-window inference is demonstrated, but this is not the same as learning the alignment causally during a previously unseen session. Human results reuse previously collected closed-loop data; they are not a prospective online NoMAD clinical trial. The authors explicitly discuss behavior consistency and stable manifold-to-behavior assumptions. [Discussion and Methods](https://www.nature.com/articles/s41467-025-59652-y).

**Reviewer analysis:** 380 pairs do not mean 380 independent subjects, and testing both directions between sessions is not identical to prospective earlier-to-later transfer. A whole-session unlabeled alignment followed by replay can use future information relative to early predictions. NoMAD is therefore an important mechanistic reference, not an eligible competition implementation without redesign.

Reference-model search can be substantial: human fitting uses AutoLFADS with 20 workers. The cited sub-10-ms LFADS inference feasibility is not a measurement of the complete NoMAD retraining/alignment latency. Total update cost and the amount of accumulated target data remain separate deployment constraints.

For Track 2, the testable principle is narrower: does a representation's task-relevant temporal structure remain stable when a simple observation adjustment is allowed? Start by comparing a static covariance descriptor to temporal features under the same corruption and split. A future dynamical alignment variant must fit only on permitted past/current windows, choose an EEG-appropriate observation loss, and report adaptation cost separately from inference latency. Preserve possible task changes rather than forcing every new cognitive state into a fixed old manifold.

## Paper 8 — Haufe transformation: predictive weights are not activation maps

**Haufe et al., NeuroImage, 2014 issue / online 2013.** [Primary record](https://pubmed.ncbi.nlm.nih.gov/24239590/), [author-hosted manuscript](https://doc.ml.tu-berlin.de/bbci/publications/HauMeiGoeDaeHayBlaBie13.pdf), especially Fig. 1 and Eq. 7.

### 1. Core Hypothesis & Problem Statement

Researchers often read classifier weights as if large weights identify the anatomical origin of class information. The paper shows why a discriminative “backward” filter can assign large weights to channels that contain no class signal, because those channels help cancel correlated noise. It provides a corresponding forward activation pattern for the linear case.

### 2. Architectural & Methodological Innovations

For centered column-vector data, let decoded components be `ŝ=Wᵀx`. The associated least-squares forward pattern is:

```text
A = Σ_x W Σ_ŝ⁻¹,       Σ_ŝ = Wᵀ Σ_x W.
```

For one output, the scalar normalization may be omitted when only pattern direction is of interest. The innovation is the formal separation of discriminative filters from generative patterns and the transformation between them, not a new high-accuracy decoder. [Eq. 7 and surrounding analysis](https://doc.ml.tu-berlin.de/bbci/publications/HauMeiGoeDaeHayBlaBie13.pdf).

### 3. Key Quantitative Results & Benchmarks

In the paper's two-feature example, the optimal LDA filter is proportional to **[1,2]**, although only the first channel carries class-related information; the transformed activation pattern is proportional to **[1,0]**. Reported class/projection correlations are **0.83** for channel 1, **0.04** for channel 2, and **0.92** for the LDA projection. This is an interpretation counterexample, not a leaderboard benchmark or a claim of an accuracy improvement. [Fig. 1 and text](https://doc.ml.tu-berlin.de/bbci/publications/HauMeiGoeDaeHayBlaBie13.pdf).

### 4. Critical Evaluation & Limitations

**Reviewer analysis:** a sensor-space forward pattern is still shaped by volume conduction and reference; it is not a localized cortical generator map. Covariance estimation changes across sessions, so identical filters can yield different activation patterns, and different filters can produce similar predictions. With nearly dependent decoded outputs, the output covariance inverse needs careful handling. For softmax logits, common offsets and redundant contrasts also complicate interpretation.

The linear transform is not a general explanation of an entire nonlinear network. A local linearization may be useful, but its validity is local and should be tested. If the goal is stable task subspaces, compare held-out decoding, patterns, perturbation sensitivity and coordinate-invariant subspace measures. Do not equate stability of one weight vector with stability of the computation.

For our linear reference, this is the first interpretation check worth implementing. Compare the trained channel-mean weights with their forward patterns using training-only covariance. Then examine changes under controlled sensor perturbations. This can help distinguish noise-cancellation features from task-associated sensor patterns before adding sparse autoencoders or mechanistic claims.

## Paper 9 — Nolte: imaginary coherency and the limits of connectivity claims

**Nolte et al., Clinical Neurophysiology, 2004.** [Institutional primary abstract](https://pure.johnshopkins.edu/en/publications/identifying-true-brain-interaction-from-eeg-data-using-the-imagin-3/), [DOI](https://doi.org/10.1016/j.clinph.2004.04.029). Full experimental methods were not independently accessible in this review.

### 1. Core Hypothesis & Problem Statement

A single neural source contributes to multiple EEG/MEG sensors through volume conduction, creating apparent connectivity. The paper proposes inspecting the imaginary component of coherency because instantaneous mixing of mutually non-interacting sources cannot by itself create such a component under the model assumptions.

### 2. Architectural & Methodological Innovations

Coherency is the complex-normalized cross-spectrum:

```text
coh_ij(f) = S_ij(f) / sqrt(S_ii(f) S_jj(f)).
Feature of interest: Im[coh_ij(f)].
```

**Reviewer derivation of the key idea:** with a real instantaneous mixing matrix M, `S_x(f)=M S_s(f) Mᵀ`. If non-interacting source spectra make S_s diagonal and real, S_x is real as well. A nonzero imaginary component therefore rejects that particular explanation. The novelty is the interpretation and use of an existing spectral quantity to address mixing artifacts; it is not a new neural decoder.

### 3. Key Quantitative Results & Benchmarks

The primary abstract describes voluntary finger-movement EEG. It reports weaker interhemispheric interaction around **20 Hz from −5 seconds to movement onset**, with stronger interaction in the opposite phase-leading relationship around **2–4 seconds after movement**. This is a physiological timing/frequency result, not command-classification accuracy. **Participant count, detailed statistical corrections and runtime were not verified from the accessible primary record.** [Primary abstract](https://pure.johnshopkins.edu/en/publications/identifying-true-brain-interaction-from-eeg-data-using-the-imagin-3/).

### 4. Critical Evaluation & Limitations

**Reviewer analysis from the model:** eliminating zero-lag contributions also removes genuinely interacting signals with zero or near-zero phase lag; absence of an imaginary component does not imply absence of a connection. Nonzero phase lag does not by itself identify a direct causal edge or exclude common drivers, delayed artifacts or mixtures of several interacting sources. A phase lead is not a general proof of biological causal direction. Short windows and limited spectral averages make estimation noisy; inspect uncertainty, not only heatmaps.

This paper constrains the language of our mechanistic study. A covariance classifier can exploit useful scalp statistics without measuring connectivity. We should call a drifting covariance direction a sensor-statistical feature until stronger evidence exists. Imaginary coherency could be a later exploratory feature/control, but it need not be more discriminative or stable. For a first study, compare its incremental value against band-power/covariance baselines and explicitly distinguish predictive value from a circuit-level interpretation.

## Paper 10 — PSID: prioritizing behaviorally relevant dynamics

**Sani, Abbaspourazad, Wong, Pesaran & Shanechi, Nature Neuroscience, 2021 issue / online 2020.** [Primary article](https://www.nature.com/articles/s41593-020-00733-0), [author-hosted full paper](https://pesaranlab.org/wp-content/uploads/2020/12/SaniPesaranShanechi2020.pdf), especially Box 1, Fig. 3 and Extended Data Figs. 2–3.

### 1. Core Hypothesis & Problem Statement

Unsupervised dynamical models can devote their latent capacity to large neural fluctuations unrelated to the measured behavior. PSID asks whether learning the behaviorally relevant subspace first produces more compact and more useful dynamics than learning a general neural model and attaching a decoder afterward.

### 2. Architectural & Methodological Innovations

A linear state-space model jointly represents neural measurements y and behavior z:

```text
x_(t+1) = A x_t + w_t;   y_t = C_y x_t + v_t;   z_t = C_z x_t + e_t.
x_t = [x_relevant; x_other], with C_z acting on the relevant block.
```

First project future behavior onto past neural data and use SVD to identify the predictive state subspace. Then optionally identify residual neural dynamics; least-squares estimates and a Kalman filter complete the model. The prioritization is the central contribution; SVD, linear system identification and Kalman filtering are established tools. Training uses behavior; test-time state estimation uses neural observations. [Box 1](https://pesaranlab.org/wp-content/uploads/2020/12/SaniPesaranShanechi2020.pdf).

### 3. Key Quantitative Results & Benchmarks

| Experiment | Reported result |
|---|---|
| 100 simulated random models | Parameter identification with **<1% normalized error** in the reported high-data test |
| Simulated 16-dimensional systems, four behavior-relevant dimensions | Relevant dynamics recovered with a four-dimensional model; **200× fewer samples** in a specified simulation comparison |
| Motor-cortex LFP power, reach/grasp task | Median relevant dimension **4 versus 12–30** implied by comparators |

Real evaluations include reach/grasp and saccade datasets, using LFP/spiking modalities. The 200× finding is not an EEG label-efficiency result. [Results and Extended Data](https://pesaranlab.org/wp-content/uploads/2020/12/SaniPesaranShanechi2020.pdf).

### 4. Critical Evaluation & Limitations

**Reviewer analysis:** “behaviorally relevant” means relevant to the recorded z under this model and training distribution. It does not mean the remaining activity is biologically meaningless, nor that the selected subspace causally generates behavior. Model order, observation quality and linearity matter. The latent coordinates are identifiable only up to appropriate state transformations, and training cannot recover neural information about unmeasured behavior without assumptions.

The block matrices used for past/future projections grow with the embedding horizon, observation dimension and number of training samples. Their storage and SVD can dominate offline fitting even when the final latent state is small. No measured PROTEUS training-time or memory figure follows from this paper.

The use of future behavior during training is not leakage if it is confined to training data and inference remains past-only. It becomes leakage if a test participant's future labels guide representation fitting. Our three command labels are trial-level categories, not rich continuous kinematics, so a direct PSID copy does not inherit the same identifiability or dimensionality claims.

The transferable experiment is to compare task-supervised low-dimensional representations with variance-maximizing ones. At matched dimension, ask which retains task information and which tracks session/sensor nuisance. Use held-out labels for evaluation; fit probes only on the allowed training partition. A low-dimensional stable readout is an attractive objective, but its existence for these three EEG tasks must be measured.

## Paper 11 — DPAD: nonlinear prioritized dynamical modeling

**Sani, Pesaran & Shanechi, Nature Neuroscience, 2024.** [Primary article](https://www.nature.com/articles/s41593-024-01731-2), [open manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC11452342/), especially Fig. 1, Fig. 2, Extended Data Fig. 9 and Methods.

### 1. Core Hypothesis & Problem Statement

PSID's linear setting cannot characterize every nonlinear neural–behavioral relationship. DPAD aims to dissociate and prioritize behavior-relevant nonlinear dynamics while testing where nonlinearities are needed, including categorical or intermittently observed behavior.

### 2. Architectural & Methodological Innovations

DPAD uses a two-section recurrent model and four ordered training steps: learn behavior-relevant states, fit their neural contribution, learn residual neural dynamics, and complete readouts. The prediction-form model is:

```text
x_(t+1) = A′(x_t) + K(y_t).
y_t = C_y(x_t) + e_t;       z_t = C_z(x_t) + ε_t.
```

Each mapping can be linear or nonlinear, with likelihoods suited to the data type. The distinctive contribution is prioritized dissociation plus controlled placement of nonlinearities; it is not simply adding an RNN to PSID's SVD. [Formulation and Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11452342/).

### 3. Key Quantitative Results & Benchmarks

Four nonhuman-primate movement datasets include reach/grasp, saccade and cursor tasks. Fig. 2 includes **35, 35, 15 and 35 session-folds** across these datasets; folds are not distinct animals. Flexible nonlinearity considers **90 architectural cases**: 61 feedforward-mapping configurations plus 29 with LSTM recursion. In a separate sparse-behavior simulation with **87 models**, DPAD outperforms the non-prioritized comparator even with one behavior sample per day (**P<0.0005**). This last result is simulated sparse mood-like sampling, not clinical mood restoration. [Fig. 2; Extended Data Fig. 9; Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11452342/).

### 4. Critical Evaluation & Limitations

**Reviewer analysis:** cross-validated prediction differences and significance do not supply a universal accuracy gain. Exact bar-height effect sizes and per-model wall-clock costs are not independently digitized here. The 90-case selection with twofold inner cross-validation creates a substantial nested-validation burden; more flexible models can win partly through larger search budgets unless comparisons are controlled.

A nonlinear readout being sufficient in a particular movement dataset does not establish that the brain's only nonlinearity lies at a literal anatomical readout. This is localization within a predictive model class, with correlated measurements and possible unobserved inputs. The same caution applies to calling a latent block “causally necessary.”

For our project, DPAD is a methodological guide to **separate task-predictive dynamics from nuisance-rich variance**, not the first model to implement. If a temporal baseline beats covariance features and sufficient training data exist, compare a small supervised recurrent bottleneck with a non-prioritized control at equal dimensions and tuning budget. Later investigate whether nonlinearities are needed in observation, dynamics or readout. Do not use sealed labels to choose the latent decomposition or adaptation behavior.

## Paper 12 — Saez: the clinical horizon of cognitive BCIs

**Ignacio Saez, The emerging field of cognitive brain–computer interfaces, Trends in Cognitive Sciences, 2026, 30(10), 881–902.** [DOI](https://doi.org/10.1016/j.tics.2026.06.012). The attached 22-page PDF was inspected directly; this is a feature review rather than a new controlled decoder experiment.

### 1. Core Hypothesis & Problem Statement

Motor and communication BCI successes do not automatically transfer to cognition. Attention, memory, decision-making and affect involve distributed, overlapping processes with less direct behavioral labels. The review argues for connecting decoding with neuromodulation and adaptive feedback to create useful cognitive interfaces.

### 2. Architectural & Methodological Innovations

This is a conceptual synthesis: high-dimensional recordings, multimodal/behavioral measurements, computational descriptions of cognition, learned neural representations and closed-loop intervention form a proposed translational pathway. It does not introduce a single executable architecture, loss or clinical protocol whose novelty can be benchmarked like EEGNet. Its distinctive contribution is the integration and problem framing. [Review](https://doi.org/10.1016/j.tics.2026.06.012).

**Reviewer formulation:** a cognitive label may be a noisy proxy for a latent state c, `Y_observed=h(c, context)+ε`. A model that predicts the instruction cue or the task condition has not necessarily estimated the clinically relevant state. An intervention then changes future observations and behavior, making the full closed-loop system a different evaluation problem from offline classification.

### 3. Key Quantitative Results & Benchmarks

**No new unified dataset, decoder benchmark, latency study or randomized intervention outcome is introduced by this review.** Numbers about existing motor, communication and stimulation systems refer to the cited original studies and should be checked there before reuse. There is therefore no defensible single “Saez accuracy,” SOTA delta or compute cost. Its quantitative contribution is not a new experimental leaderboard result. [Article type and scope](https://doi.org/10.1016/j.tics.2026.06.012).

### 4. Critical Evaluation & Limitations

**Reviewer analysis:** the clinical challenge is not solved by increasing channel count or model size. We need reliable target-state measurement, patient-relevant outcomes, feasible sensing, and evidence that changing the inferred state improves function. Correlated markers can be consequences, compensatory responses or context indicators rather than intervention targets.

For Track 2, successful command decoding can advance a measurement/adaptation method, but it cannot establish restored memory, a psychiatric biomarker or safe stimulation. The practical connection is to make reliability, recalibration burden and failure warning measurable. Later, a vendor-neutral layer could expose input units, channel identity, timestamp/state-reset semantics and uncertainty to hardware-specific adapters. BIDS/NWB can support data interoperability; they do not establish shared biology, live hardware compatibility or equivalent clinical performance.

The review supports our long-term direction while keeping the first study narrow: determine whether a permitted label-free update improves transfer and whether a separate monitor can identify when the fixed decoder is becoming unreliable. Clinical intervention claims require additional data, target definitions and prospective studies.

## Research report 13 — Neuralink: pretraining on unlabeled intracortical data

**Neuralink engineering/decoding update, 1 October 2026.** [Primary company report](https://neuralink.com/updates/pretraining-on-50000-hours/). The article body was inspected in its public page asset because the web text extractor returned an empty page. This is a company report, not a peer-reviewed benchmark paper.

### 1. Core Hypothesis & Problem Statement

Repeated supervised calibration limits practical BCI use. The report argues that participant-specific pretraining on freeform recordings can yield embeddings that support more accurate and longer-lived decoders with less labeled calibration.

### 2. Architectural & Methodological Innovations

The described system uses event/channel embeddings, a Perceiver input layer, a Mamba temporal model and **spatially masked auto-Poisson regression**: feed half the channels and predict the other half. The report cites prior dynamical systems and population-decoding work. Its contribution is their production-oriented integration and reported user experience, not the invention of all components. Model sizes and a full reproducible training configuration are not disclosed. [Report, “Towards a stable representation”](https://neuralink.com/updates/pretraining-on-50000-hours/).

**Reviewer interpretation:** the “2” after Mamba in plain text is a citation marker, not evidence that the architecture is specifically Mamba-2. Event sparsity and Poisson-style targets fit intracortical spike observations. Scalp voltages are dense, signed mixtures, so the input tokenizer and likelihood do not transfer directly. Channel identity and cross-channel prediction remain relevant principles.

### 3. Key Quantitative Results & Benchmarks

The company reports **50,000+ hours across participants**, with live results from **single-participant pretrained models**; six participants reached personal records, with a peak **11.32 bits/s**. It reports a comparison in which **30 seconds** of labeled embeddings matched about **3.5 minutes** of raw spikes, and examples of strong control for over three weeks. A 20-month-old calibration example demonstrates usable control, not a cohort-wide guarantee. [Results and longevity sections](https://neuralink.com/updates/pretraining-on-50000-hours/).

### 4. Critical Evaluation & Limitations

The report explicitly says pooled multi-participant models have **not yet improved online performance** over participant-specific models. It does not supply a controlled public benchmark with a complete cohort-level uncertainty analysis or a verified cross-subject EEG scaling law. [Future directions](https://neuralink.com/updates/pretraining-on-50000-hours/).

**Reviewer analysis:** 50,000 cohort hours does not mean each successful model trained on all 50,000, nor does it establish how many EEG hours our model needs. The ~7× calibration-time comparison is an arithmetic reading of the reported example, not a universal label-efficiency factor. Bits/s depends on the interaction task and cannot be converted into our three-command balanced accuracy.

The useful research transfer is to test whether SSL produces task-useful representations with lower sensitivity to nuisance shifts, using matched downstream label budgets. Pretraining may improve the starting representation without providing any explicit retention guarantee or future-failure warning. A representation that hides recording day in one middle layer is also not automatically invariant: low-dimensional plots can hide day information, and task information may be lost along with nuisance information. Measure both with held-out probes and downstream performance.

## Research report 14 — Goodfire / Prima Mente: interpretation-guided biomarker hypotheses

**Wang and collaborators, company research report, 28 January 2026.** [Primary report](https://www.goodfire.com/research/interpretability-for-alzheimers-detection). Evidence is a small-cohort proof of concept, not a clinical-validation trial.

### 1. Core Hypothesis & Problem Statement

A strong disease classifier may use biological features that differ from researchers' expectations. This work asks whether examining an epigenetic model can identify simpler candidate predictors that generalize to another cohort.

### 2. Architectural & Methodological Innovations

Pleiades representations feed a hierarchical fragment-to-region-to-patient classifier. Probes, BatchTopK sparse autoencoders and gradient attribution identify fragment-length information; these observations guide a simpler logistic-regression predictor. The final fragmentomic model retains **11 of 2,478 candidate features**, selected for consistency over training bootstraps. The distinctive step is interpretation-guided hypothesis triage followed by an independent-cohort test, rather than SAEs or logistic regression alone. [Methods and results](https://www.goodfire.com/research/interpretability-for-alzheimers-detection).

### 3. Key Quantitative Results & Benchmarks

| Evaluation | Reported result |
|---|---|
| Development cohort / independent OPTIMA cohort | **81 / 25 people** |
| Fragmentomic predictor, independent cohort | **AUROC 0.78**, 95% CI **[0.56,0.95]** |
| Combined feature categories | **AUROC 0.84**, 95% CI **[0.66,0.97]** |
| Methylation-only / deconvolution-only references | AUROC **0.55 / 0.54** |

The report's Pleiades scale is 7B parameters and 1.9T pretraining tokens; these are company-reported quantities, not compute requirements for our EEG study. [Primary report](https://www.goodfire.com/research/interpretability-for-alzheimers-detection).

### 4. Critical Evaluation & Limitations

The authors acknowledge small, geographically limited cohorts and the interpretation-to-biology gap. Wide intervals mean the point estimate is imprecise. Fragment-level sample counts do not replace independent patients. Gene proximity and association do **not** establish that identified genes cause Alzheimer's disease. [Limitations](https://www.goodfire.com/research/interpretability-for-alzheimers-detection).

**Reviewer analysis:** probes show accessible information, attribution suggests model dependence, and a simpler predictor tests a hypothesis; these are different levels of evidence. Model-internal causal necessity would require controlled interventions and checks for off-manifold artifacts. Biological causality requires still stronger experimental evidence. A PCA manifold and its curvature also depend on representation and projection choices.

The foundation model and the final logistic model have very different costs. As a rough reviewer storage calculation, 7 billion 16-bit parameters alone occupy about 14 GB before activations, gradients or optimizer state. That is not a measured runtime budget for this report, and it should not be confused with the much smaller cost of deploying a distilled feature classifier.

This workflow transfers more cleanly than the epigenetic architecture: inspect an EEG decoder, form an explicit candidate feature hypothesis, test a simple model, and validate on held-out participants/sessions. For example, if a latent direction tracks high-frequency power, test whether it reflects task information or muscle contamination using matched controls. Use random-subspace ablations, held-out decoding and nuisance probes before naming it a “stable neural mechanism.” Interpretation is useful when it changes a testable experiment, not merely because a visualization looks structured.

## Supporting tutorial — masked EEG prediction in the official stack

[Meta NeuroAI masked-pretraining tutorial](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_pretrain_mae.html).

This is an implementation reference, not a paper showing a new Track 2 SOTA. It tokenizes channel–time patches, masks selected tokens, reconstructs hidden patches with MSE and retains the encoder. Its configuration illustrates 3D position embeddings, absent-channel masking, whole-subject splits, a 256-dimensional model, 32-sample patches and a 0.5 mask ratio. Those are tutorial settings, not validated optimal values for our study. The page states that published pretrained weights are not supplied. [Configuration and downstream scoring](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_pretrain_mae.html).

**Reviewer analysis:** this matches our preference to preserve electrode geometry, but spatial proximity is not the only relevant relation: reference, bipolar derivations and volume conduction change what a coordinate describes. Mapping a bipolar derivation to one endpoint does not fully represent its measurement operator. Test absent channels and montage changes explicitly. At 120 Hz, 32 samples represent about **267 ms**; that conversion is arithmetic, not a physiological optimum.

A particularly useful implementation caution is the tutorial's warning that a checkpoint size mismatch can log a warning while retaining random weights. A valid transfer experiment must confirm which tensors loaded, verify a parameter fingerprint and compare against an explicitly random-initialized control. A path called “pretrained.ckpt” is not evidence of successful pretraining transfer.

## Synthesis & Comparative Analysis

### Cross-paper comparison table

The rows summarize mechanisms and evidence limits, not a cross-dataset performance ranking.

| Paper/report | Core idea | Methodology | Key strength | Primary bottleneck / trade-off |
|---|---|---|---|---|
| RPA | Remove geometric domain mismatch | SPD recentering, dispersion matching, class-wise rotation | Compact, physically interpretable statistical reference | Target labels for rotation; covariance drops event ordering |
| MINDFUL | Track instability associated with loss of control | Reference/current moment divergence on neural and decoder features | Label-free score during use | Association with current error; reference/task confounds; no universal forecast |
| EWC | Protect previously important weights | Diagonal Fisher quadratic anchor | Explicit stability–plasticity control | Local approximation; can obstruct a necessary correction |
| EEGNet | Learn compact EEG features | Factorized temporal/spatial/separable convolutions | Strong, affordable supervised reference | Montage/rate and normalization sensitivity; no drift guarantee |
| LaBraM | Reuse heterogeneous EEG through SSL | Spectrum-trained VQ codes and masked token prediction | Direct scalp-EEG pretraining evidence | Pretraining/tuning cost; target compatibility; task-dependent gains |
| EATA | Avoid unreliable/redundant updates and forgetting | Entropy/diversity selection plus Fisher penalty | Couples update efficiency and retention | Confident errors; ID Fisher reference; image-to-EEG gap |
| NoMAD | Preserve useful latent dynamics despite changed observations | LFADS core plus unsupervised alignment | Strong long-span intracortical offline results | Whole-session alignment versus causal adaptation; stable-dynamics assumption |
| Haufe | Separate extraction filters from activation patterns | Covariance-based linear forward transformation | Prevents common interpretation errors | Linear scope; a sensor pattern is not a cortical cause |
| Nolte | Reduce instantaneous-mixing explanations of connectivity | Imaginary part of coherency | Explicit physical assumption behind a spectral measure | Discards zero-lag interactions; no general direct-causality proof |
| PSID | Learn measured-behavior-relevant dynamics first | Predictive subspace projections, SVD, state-space model | Compact task-relevant latents | Linear model and behavior supervision; relevance depends on labels |
| DPAD | Extend prioritized dynamics to nonlinear mappings | Two-section RNN, staged learning, selective nonlinearities | Tests the location of predictive nonlinearities | Nested search cost; predictive localization is not anatomy |
| Saez | Connect cognitive decoding and intervention | Clinical/conceptual synthesis | Defines the translation gap and target-label problem | No new unified experimental benchmark or clinical efficacy result |
| Neuralink report | Use freeform recordings to lower calibration burden | Participant-specific spatial masking and dynamical encoder | Deployment-oriented intracortical examples | Unreleased data/configuration; no pooled online gain demonstrated |
| Goodfire report | Turn model inspection into simpler testable hypotheses | Probes/SAEs/attribution, distilled features, independent cohort | Interpretation leads to an external validation test | Small cohort, confounding, no gene-causality demonstration |

### Complementary mechanisms: combinations worth testing

**1. Statistical alignment and compact temporal decoding — RPA + EEGNet.** These target different aspects of the observation: nuisance statistics versus task features. First test recentering as its own baseline; then test a declared causal preprocessing variant with a fixed decoder. A covariance-space congruence transform and a raw-signal transform are related only under specified estimation/reference assumptions; do not apply a matrix operation to EEG and assume it reproduces every RPA pipeline. Preserve a no-alignment control and assess whether alignment removes useful class structure.

**2. Monitoring and adaptation control — MINDFUL + EATA/EWC.** A monitor can flag when a change deserves attention; sample filtering chooses update evidence; a regularizer constrains what changes. Their responsibilities should stay separately testable. Monitor the fixed teacher as well as the adaptive model. Otherwise adaptation can reduce its own novelty score while worsening true accuracy. Test “always adapt,” “never adapt” and “monitor-gated adapt” under identical budgets. A gate's value is the improvement in the final decision policy, not just a correlation plot.

**3. Reusable features and prioritized information — LaBraM + PSID/DPAD principles.** SSL may provide useful initial features, while task-supervised bottlenecks can prioritize a small subset. Compare frozen pretrained features to random features, and task-supervised projections to PCA at equal dimension. The full dynamical methods need an appropriate temporal label/observation setup; with trial-level task categories, this is a proposed adaptation of the principle, not a replication of their motor-kinematics experiments.

**4. Explanation and stability — Haufe/Nolte + the Goodfire workflow.** Use correct sensor-pattern interpretation, artifact controls and matched latent ablations to form a narrow hypothesis. Then build a simpler feature-based decoder or monitor and evaluate it on new participants/sessions. A finding becomes useful when it survives these controls and predicts a measurable failure or preservation benefit. EEG covariance can remain a powerful feature even when no connectivity claim is warranted.

No reviewed study verifies this combined stack for PROTEUS. Combining all components immediately would obscure which one helps. These are hypotheses for staged experiments with matched controls, not a proposed finished architecture.

### Drift, forgetting and a recalibration penalty: formulate three separate objectives

The following are **reviewer-proposed experimental definitions**. They are not benchmark metrics established by the papers.

Let θ₀ be the fixed trained model, θ_s its permitted adapted state, and A a predeclared higher-is-better held-out score, such as cell-averaged balanced accuracy.

```text
Adaptation gain: G_s = A_later(θ_s) − A_later(θ₀).
Signed retention change: R_s = A_reference(θ_s) − A_reference(θ₀).
Forgetting loss: F_s = max(0, −R_s).
```

Evaluate the same reference examples in both terms and keep them out of update selection. Reference evaluation is an offline research diagnostic; it does not give the sealed model labels or permission to retain state across test sessions. Also record the signed change so positive backward transfer is not hidden by truncating F at zero.

There are three different meanings of “penalize recalibration”:

| Penalty | Example | What it buys / what it cannot guarantee |
|---|---|---|
| Parameter protection | `λ Σ_i F_i(θ_i−θ_i⁰)²` | Limits change in important parameter directions; does not itself price user effort |
| Functional protection | Distillation or divergence from a frozen teacher on permitted reference data | Protects selected outputs; can preserve old mistakes and may require storing data/features |
| Operational cost | Label count, calibration minutes, gradient steps, latency or energy | Makes deployment trade-offs explicit; units and weights must be declared |

A useful **research reporting objective** is a Pareto comparison of target performance, retention loss, labeled-data cost and update compute. A scalar `J = A − αF − β·labels − γ·update_time` is possible only after deciding the relative units and preferences. Its α/β/γ are design choices, not values supplied by these papers or the competition. Do not optimize an arbitrary scalar and hide a serious drop in one component.

For the current competition, the score is cell-averaged balanced accuracy and no extra labeled sealed calibration is provided. Label efficiency can first be studied by subsampling **public training labels**, keeping validation/test and sampling protocols fixed. A separate known-user longitudinal calibration experiment can use declared public early-session labels, but its limited repeated-session cohort must be reported honestly.

### Harmful-drift prediction: a falsifiable research question

**Proposed question:** from a prefix of unlabeled EEG windows, can a monitor predict a later decline of a fixed decoder better than signal quality, prediction confidence and generic distribution distance?

Specify a feature window W ending at time k and a non-overlapping future evaluation block of horizon H. One operational harm event is:

```text
h_k = 1[A_fixed(future block k+1 ... k+H) < τ].
Alternative: h_k = 1[future performance drop from a defined reference > δ].
```

H, τ, δ and the required sample size are **unverified design choices to be selected using training/validation data**, not invented recommended hyperparameters. A short future block missing a class can make balanced accuracy noisy or undefined under some conventions. Predeclare the missing-class handling or use an appropriate trial-level loss aggregate with uncertainty; do not change the event definition after seeing test failures. If event probabilities come from a learned monitor, their calibration must be evaluated too.

| Question | Minimal useful evidence |
|---|---|
| Is it forecasting rather than detecting current harm? | Predictor uses information through k only; future outcome begins after k; report lead time |
| Does it add value beyond existing signals? | Same held-out examples for signal-quality, entropy/confidence and marginal-shift comparators |
| Does it distinguish harmful from benign shift? | Task-mix changes and benign sensor perturbations as controls; naturally occurring failures where available |
| Is the estimate reliable? | Participant/session-held-out evaluation; block-aware uncertainty, event counts and prevalence |
| Does it help a decision? | False alarms per defined exposure and recall at a fixed alarm budget; compare monitoring/adaptation policy utility |

Report AUROC and, for rare failure events, precision–recall performance relative to event prevalence; include calibration and lead time where meaningful. Overlapping rolling blocks inflate the apparent sample size, so split and resample at defensible temporal/participant units. Do not claim “hundreds of independent failures” from hundreds of overlapping windows. Synthetic shifts reveal sensitivity to specified perturbations; they cannot establish prediction of natural clinical failure.

Only two public participants have repeated sessions in this release. A credible first conclusion may therefore be an exploratory within-stream warning result with individual outcomes, plus a clearly separated cross-participant evaluation. Longitudinal forecasting needs more independent failure episodes before broad claims. There is also no guarantee that useful pre-failure information exists at the chosen horizon.

### What counts as a good result?

**Competition:** improve the official cell-averaged score on genuinely held-out participants/sessions, with valid causal access, completed inference and a reproducible artifact. A warm-up score or a local split does not establish sealed placement. There is no verified “winning accuracy threshold” in this review.

**Methodological advancement:** show a reproducible improvement over strong matched controls: fewer labeled training examples for comparable decoding, less forgetting at comparable target performance, or a warning signal with useful lead time at an acceptable false-alarm burden. A small average improvement can matter if it reliably protects poor-performing cells; conversely, a large pooled gain can hide failures in a subset of users.

**Mechanistic advancement:** identify a specific computational component whose removal/alteration has a reproducible task-dependent effect, survives matched random interventions and artifacts controls, and explains a prediction about transfer. That supports model-level dependence. Claiming a biological cause or clinical benefit requires further experiments.

The current project record contains a real linear baseline of **0.3796610** public held-out cell-averaged balanced accuracy from one seed/split and a user-confirmed **0.33** constant warm-up score. These are project status figures, not results from any reviewed paper and not direct paired competitors on an identical evaluation set. [Local verification record](C:/Projects/EEG-EMG-Foundation-Challenge-2026/eeg-emg-challenge-2026-track2-bcidecoding/docs/STATUS.md).

### Five actionable design principles

1. **Keep a fixed reference and preserve the actual measurement contract.** Match units, channel order, sample rate and state reset before interpreting gains. EEGNet and a covariance reference should remain independently reproducible controls. Every checkpoint load must prove the intended weights were used.
2. **Adapt nuisance estimates before broad parameter updates, and test the assumption.** Compare a causal recentering/normalization reference against no adaptation. Then compare restricted updates with uniform and Fisher anchoring at equal compute. Never reuse RPA's target-label rotation as if it were unsupervised.
3. **Evaluate label efficiency and retention as separate curves.** Vary only the declared public training-label budget, repeat sampling, and record target score versus reference retention. Use matched splits and tuning budgets; do not promise a calibration reduction based on an invasive or simulated factor.
4. **Make harmful-drift forecasting an independent endpoint.** Predeclare prefix, horizon, event definition and alarm budget; compare frozen-decoder quality/confidence/shift monitors. Keep naturally occurring failures distinct from injected artifacts and current-error detection distinct from future warning.
5. **Use interpretation to produce a simple falsifiable hypothesis.** Prefer a validated forward pattern, band-power/covariance feature or matched subspace ablation before a large SAE stack. Check task, participant and nuisance information separately; follow a finding with an independent decoding/monitoring test.

## Supporting documentation: what each source contributes

| Reading-list source | Role in the project | Boundary |
|---|---|---|
| [Scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md) | Model-facing tensor, chronology, adaptation and calibration restrictions | Contract, not a scientific performance study |
| [Worked Track 2 kit](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding) | Executable loading, reference solvers, metric and streaming behavior | Pin a revision; code and prose can evolve |
| [Submission guide](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md) | One-file Python inference packaging and read-only artifact access | Generic “do not train” language must be read with Track 2's explicit allowance for causal label-free updates |
| [PROTEUS/NEMAR nm000290](https://nemar.org/dataset/nm000290) | Dataset access and license | An archive is not a transfer-method paper |
| [PROTEUS dataset README](https://github.com/nemarDatasets/nm000290) | Acquisition/reference/session context | Use actual metadata to confirm a particular downloaded release |
| [NeuralBench Track 2 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html) | Baseline environment and task workflow | Older descriptions must not override the active FAQ/objective |
| [Benchopt](https://benchopt.github.io/) | Reproducible comparison runner | Successful execution establishes a working experiment, not a novel model |
| [NWB](https://nwb.org/) | Neurophysiology/behavior data standard and ecosystem | A shared archive format is not a live cross-vendor adaptation API |
| [BIDS](https://bids.neuroimaging.io/) | Organization and metadata for neuroimaging datasets, including EEG | Correct metadata does not guarantee correct units, timing or scientific validity in a downstream adapter |

The masked-pretraining tutorial is reviewed separately above. None of these documentation entries should be assigned an invented accuracy, ablation gain or clinical outcome.

## Suggested reading sequence and decisions before implementation

Read **EEGNet → RPA → Haufe/Nolte** to understand signal features and interpretation. Then read **MINDFUL → EWC → EATA** for monitoring, update control and retention. Use **LaBraM + the official MAE tutorial** to decide whether pretraining is justified. Read **PSID → DPAD → NoMAD** when designing the stable-subspace/dynamics analysis. The **Neuralink, Goodfire and Saez** readings then connect the experimental work to calibration burden, hypothesis generation and the clinical horizon.

Three decisions deserve explicit agreement before the next algorithmic phase:

1. **Temporal representation:** should the first controlled comparison be covariance/log-power versus temporal EEGNet features at matched splits, with geometry-aware encoding deferred until that comparison reveals a benefit? Which timing information must remain recoverable?
2. **Stability budget:** what target-performance loss, reference forgetting, label cost and update latency would make an adaptation unacceptable? These are distinct axes; choose constraints before tuning their penalties.
3. **Forecasting target:** is the first endpoint warning of poor performance later in the same session, or deterioration on a later visit? The second requires genuinely longitudinal outcomes and a larger independent failure sample than the current repeated-session public subset provides.

## Evidence and uncertainty record

Source documents were checked on 21 October 2026. Stable paper titles/DOIs identify the studies; “not verified” fields are deliberate. No paper result here was reproduced in a new experiment, no clinical dataset was analyzed, and no performance claim was inferred from an unread graph. Numerical differences explicitly called arithmetic are derived from the cited values. Author-documented limitations and reviewer critiques are labeled separately wherever their status matters.

Remaining source limitations are the inaccessible original RPA result tables and unverified full Nolte experimental methods. These do not justify inventing missing accuracy/runtime numbers. A future revision can add their exact table-level results when the original full texts are available. All proposed EEG combinations, retention definitions and harmful-drift endpoints remain hypotheses until the corresponding controlled experiments have been run.
