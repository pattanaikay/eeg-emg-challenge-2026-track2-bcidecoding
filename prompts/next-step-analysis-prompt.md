```markdown
You have the full context of our PROTEUS EEG decoding competition in this chat, as well as the additional methods detailed in the attached file. 

I need you to act as a Principal ML Research Engineer and design an optimal, prioritized experimental roadmap. Your goal is to evaluate all candidate methods—both from my paper notes below and from the attached file—and determine which experiments we should run first, which to hold as secondary ablations, and which to discard.

### 1. Hard Competition Constraints (Non-Negotiable)
Every method you evaluate must respect our exact deployment boundary:
* **Task:** 3-class classification (Motor Imagery, Mental Subtraction, Word Generation).
* **Input format:** Streaming single `(41, 480)` scalp EEG windows at 120 Hz (spatial mixture, low SNR).
* **Blind inference:** No participant ID, no session ID, no target labels, and no extra calibration labels.
* **Strict Causality & Reset:** Test-time adaptation (TTA) may only use the current and earlier windows within the active session, and internal state must cleanly reset at session boundaries.
* **Corpus Constraint:** Fixed training corpus; external foundation-model weights are outside our primary comparison.

---

### 2. Core Literature Notes & Minimum Viable Experiments (MVEs)

**Track A: BTTA-DG (Domain Generalization & Test-Time Adaptation)**
* *Caveat:* Original method targets binary left/right MI, which is narrower than our heterogeneous 3-command task. The paper should inspire isolated experiments, not dictate the final architecture.
* *Safe translation sequence:*
  1. Test **only** the learned Sinc frequency front-end against our existing EEGNet baseline (identical participant split, training budget, and checkpoint rule).
  2. Separately test whether confidence summaries help, using a frozen source-trained model and a simple causal memory control.
  3. Add the GMM **only** if the confidence summary helps, and strictly with explicit cold-start, missing-class, session-reset, and wrong-pseudo-label safeguards.

**Track B: Masking & Self-Supervised Representation Learning (SSL)**
* *Smallest honest test:* A matched 2×2 experiment: `(MAE vs. JEPA)` × `(Scattered vs. Contiguous spatial-temporal masks)`.
* *Strict controls:* Enforce participant-separated splits *before* SSL pre-training. Match realized masked fraction, optimizer steps, decoder capacity, and total compute.
* *Baselines & Diagnostics:* Compare all 4 SSL variants against supervised scratch + EEGNet, supervised band-power/covariance baselines, and a random encoder control. Evaluate on official `participant × session × interface` balanced accuracy, per-class recall, and a content-swapping sanity check (verifying that the embedding changes when EEG content is swapped while channel positions remain fixed).

**Track C: NoMAD, Drift & Stable Dynamics**
* *Caveat:* Scalp EEG is a spatial mixture rather than a clean motor-cortex spike population. Do not build full NoMAD.
* *Smallest transferable hypothesis:* `Frozen EEG classifier + identity-initialized small adapter + causal normalization`.
* *Strict controls:* Use only unlabeled current/past windows in the session. Compare: (1) no adaptation, (2) causal normalization alone, and (3) adapter with a weak alignment penalty. Keep the classifier fixed initially to separate genuine adaptation from catastrophic forgetting.

**Track D: Other Methods (From Attached File)**
* Extract and evaluate every candidate method present in the attached file under the exact same lens of minimalism, strict causality, and baseline controls.

---

### 3. Mandatory Falsification Protocol for Every Model Change
To prevent chasing dead ends, overfitting, or shortcut learning, every single architectural or feature change must survive the following four-point falsification filter before being approved:
1. **The Leakage & Temporal Audit (Fit-on-Train Only):**
   - Before reporting any metric improvement from a new feature extractor, scaler, spatial filter, or PCA/ICA step, verify and prove in code that zero statistics (mean, variance, covariance, channel weights) are computed across the validation/test boundary or using future timestamps within a session.
2. **The "Shortcut Learning" Kill Test (Confounder Ablation):**
   - Neural recordings contain massive session-specific drift, electrode impedance noise, and EMG/movement artifacts.
   - Whenever a new feature extraction technique improves CV, run an **ablation/permutation test**: 
     a) Shuffle the neural features within each class/session, OR 
     b) Train the exact same classifier using ONLY out-of-band noise / baseline pre-stimulus windows.
   - If the model still achieves above-chance accuracy on pre-stimulus noise or shuffled temporal structure, our feature extractor is exploiting session metadata/drift, not neural signal. Flag it as a **FAILED technique**.
3. **Cross-Session / Drift Stress Test:**
   - Never evaluate on random K-fold splits of contiguous time windows (which leaks temporal autocorrelation). 
   - Evaluate strictly on held-out sessions, held-out blocks, or chronological tail-splits. Report the **worst-session performance** and **variance across sessions**, not just the mean metric.
4. **Complexity Penalty (Baseline Justification):**
   - Every architectural addition must beat a dead-simple, heavily regularized baseline (e.g., temporal convolution / linear projection + L2 regularization) on the exact same strict split by more than 1 standard deviation of seed variance. If it falls within noise, revert the code.

---

### 4. Required Chain-of-Thought (CoT) Analysis
Do not jump straight to recommendations. Work through the following steps explicitly:

* **Step 1: Inventory & Translation of Attached File**
  Briefly list each method in the attached file and translate it into its "smallest honest test" (Minimum Viable Experiment) that complies with our `(41, 480)` streaming causal constraints. Flag any method that violates causality, requires session/participant IDs, or relies on multi-class assumptions that break under class-imbalanced streaming.

* **Step 2: Cross-Method Friction & Failure-Mode Audit**
  For every candidate across Tracks A, B, C, and D, evaluate:
  1. *Heterogeneity risk:* Does it assume purely oscillatory features (like motor imagery) that might hurt mental subtraction or word generation?
  2. *Online adaptation failure modes:* How vulnerable is it to cold-start instability (first 1–10 windows), single-class streaks (missing classes in a short window history), or pseudo-label collapse?
  3. *Compute vs. Signal ratio:* Given our fixed 41×480 single-corpus limit, is the method over-parameterized compared to EEGNet / covariance baselines?

* **Step 3: Orthogonality & Synergy Mapping**
  Group the surviving methods by where they sit in the pipeline:
  * *Front-End / Representation* (e.g., Sinc filters, Band-power/Covariance, SSL pre-training)
  * *Backbone / Classifier* (e.g., EEGNet, Riemannian/Linear heads)
  * *Causal Test-Time Adaptation* (e.g., Causal norm, Confidence summary, Identity-initialized adapter)
  Identify which experiments are orthogonal (can be combined later if both succeed) and which are mutually exclusive.

* **Step 4: Prioritized Experimental Roadmap (Expected Information Gain per GPU Hour)**
  Rank the concrete experiments into three tiers:
  * **Tier 1 (Immediate Execution):** Low engineering complexity, minimal failure modes, high diagnostic value (e.g., representation & front-end baselines, causal norm alone).
  * **Tier 2 (Conditional Follow-ups):** Medium complexity or gated behind Tier 1 success (e.g., 2×2 SSL grid, identity-initialized adapter, confidence summaries).
  * **Tier 3 (High-Risk / Late-Stage):** Complex moving parts or heavy hyperparameter sensitivity (e.g., safeguarded GMM, complex methods from the attached file).

For each recommended experiment in Tier 1 and Tier 2, explicitly state the **Hypothesis**, **Control/Baseline**, **Primary Metric + Sanity Check**, and how it must pass the **Mandatory Falsification Protocol** before moving to the next step.

```