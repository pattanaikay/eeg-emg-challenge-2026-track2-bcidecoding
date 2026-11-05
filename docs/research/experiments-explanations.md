**E1, E2 and E3 ask three different questions about why our decoder struggles with new recordings:**

| Experiment | Question |
|---|---|
| **E1: frequency power** | Can we recognize the command from the strength of different EEG rhythms? |
| **E2: covariance** | Does the pattern of channels varying together provide additional information? |
| **E3: normalization/alignment** | Can we compensate for changes in how the EEG was recorded, while preserving command information? |

They help us identify the bottleneck before building a more complicated model. None is guaranteed to outperform EEGNet.

Start with the input. Each trial is a matrix:

$$
X \in \mathbb{R}^{41 \times 480}.
$$

There are 41 electrode signals, each containing 480 voltage measurements over four seconds. Think of 41 microphones recording the same orchestra from different positions. Each electrode captures a mixture of electrical sources; it does not correspond to one isolated brain region.

**E1: summarize which rhythms are strong.**

An EEG signal fluctuates at different rates. Frequency tells us how quickly a fluctuation repeats: a 10 Hz oscillation repeats approximately ten times per second.

Two signals can average to almost zero while having very different strengths:

```text
Small oscillation:    small positive and negative fluctuations → mean ≈ 0
Large oscillation:    large positive and negative fluctuations → mean ≈ 0
```

Our existing channel-mean linear baseline largely loses that distinction. E1 measures it explicitly.

For each electrode, we separate the signal into candidate frequency bands. The roadmap proposes 4–8, 8–13, 13–30 and 30–45 Hz, with 1–4 Hz tested separately. These are starting choices, not established optimal bands for this dataset.

For each filtered signal, we calculate its power—roughly, its average squared amplitude:

$$
P_{c,b} = \frac{1}{T} \sum_{s=1}^{T} x_{c,b}(s)^2.
$$

Here, $c$ identifies the electrode and $b$ the frequency band.

We then take the logarithm:

$$
z_{c,b} = \log(P_{c,b} + \epsilon).
$$

The small $\epsilon$ prevents taking the logarithm of zero. The logarithm compresses large differences: an unusually large amplitude becomes less dominant.

With four bands and 41 channels, each trial becomes **164 numbers**, rather than 19,680 voltage samples.

```text
41 EEG signals
      ↓
Separate into frequency bands
      ↓
Measure power in each channel and band
      ↓
164 features
      ↓
Small classifier → command
```

The classifier learns combinations such as “this distribution of rhythmic power across electrodes supports motor imagery.” We should let the data establish the combinations; we cannot assume subtraction or word generation has one universal frequency signature.

“**L2 logistic regression**” means a classifier that produces class probabilities while penalizing excessively large weights. That penalty reduces its freedom to memorize peculiarities of individual training participants.

What are we attempting? **Give the classifier a useful summary directly, so it does not have to learn frequency extraction from our limited data.**

The trade-off is that average power discards timing. Two trials with the same band powers but different sequences of activity can look identical to E1.

We compare it with freshly trained EEGNet on the same splits. Removing one band at a time—an **ablation**—helps reveal which information contributes. We also inspect each command’s recall so an overall improvement does not conceal recurring harm to subtraction or word generation.

**E2: examine how channels vary together.**

E1 describes each electrode separately. E2 asks whether relationships between electrode signals add information.

Imagine two channels with identical individual power:

```text
Trial A: when channel 1 rises, channel 2 usually rises.
Trial B: when channel 1 rises, channel 2 usually falls.
```

Their individual powers can be the same. Their **covariance** differs.

After subtracting each channel’s mean, we calculate:

$$
C_{ij}
=
\frac{1}{T-1}
\sum_{s=1}^{T}
\bigl(x_i(s)-\bar x_i\bigr)
\bigl(x_j(s)-\bar x_j\bigr).
$$

The resulting $41 \times 41$ matrix contains:

- **Diagonal entries:** each channel’s variance, closely related to its power.
- **Off-diagonal entries:** how pairs of channels vary together.

Covariance includes amplitude information; it is not the same as correlation.

This might capture a spatial pattern that distinguishes commands. However, channels can vary together because scalp EEG spreads through conductive tissue, because they share a reference, or because an artifact affects several electrodes. **Covariance is not proof of communication between brain regions.**

A full covariance matrix has **861 unique entries**. Estimating all those relationships from four seconds is noisy, especially because neighboring time samples are dependent.

We therefore use **shrinkage**: pull the estimated matrix slightly toward a simpler, better-conditioned matrix:

$$
\widehat{C} = (1-\alpha) C + \alpha \tau I.
$$

Think of it as saying: “Trust the measured relationships, but moderate the most unstable estimates.”

The next phrase, “**tangent space**,” sounds more intimidating than its purpose. Valid positive-definite covariance matrices occupy a curved mathematical space. We choose a reference covariance from training data and use a matrix transformation to express each trial relative to that reference as an ordinary feature vector.

```text
EEG window
    ↓
Stable covariance estimate
    ↓
Express relative to training reference
    ↓
Feature vector
    ↓
Small classifier → command
```

A regularized LDA or logistic classifier then operates on those vectors.

What are we attempting? **Determine whether spatial relationships provide useful command information beyond individual channel powers.**

The key control is a **diagonal-only covariance model**. It retains individual variances and removes cross-channel relationships. If the full model cannot improve on that control, we have little evidence that estimating all the extra relationships earns its complexity.

E2 does not automatically solve session drift. Changes in electrode contact, reference or spatial mixing can also change covariance.

**E3: compensate for recording changes.**

Suppose the person performs the same command on two visits, but some channels have larger amplitudes on the second visit because recording conditions changed.

The classifier may interpret that measurement change as a command change.

E3 asks whether transforming the input into a more consistent form improves transfer. It is a **preprocessing/adaptation experiment**, which can be applied to a selected decoder.

We test alternatives separately.

**Source mean/std scaling** uses statistics calculated from training data:

$$
x'_c = \frac{x_c - \mu_c^{\mathrm{train}}}{\sigma_c^{\mathrm{train}} + \epsilon}.
$$

It establishes a fixed scale. It cannot automatically follow a new session’s measurement changes.

**Per-window robust scaling** calculates a median and a spread measure, such as the interquartile range, from the current four-second window. Compared with mean/std, these statistics are less sensitive to extreme values.

The danger is that amplitude itself may distinguish commands. Normalization could remove the information E1 needs.

**Running normalization or alignment** gradually estimates the current session’s recording statistics using available windows. It begins with a training-derived reference rather than trusting one noisy first trial.

Covariance alignment extends this idea beyond individual amplitudes: it applies a shared spatial transformation intended to reduce differences in the session’s overall covariance.

Crucially, we must not whiten every trial using that trial’s own covariance. Doing so can make each transformed covariance approximately identity, erasing E2’s features.

There is another trap: if recent windows mostly contain one command, their statistics may reflect that command rather than a recording problem. An adaptation method can mistakenly “correct away” useful task activity.

What are we attempting? **Remove nuisance variation without removing the differences between commands.**

That is why we measure startup performance, behavior during long command streaks, and later performance. Running methods must use only current/past windows and reset each session.

**How the experiments work together**

E1 asks whether **rhythmic strength** is sufficient. E2 asks whether **spatial relationships** add value. E3 asks whether **recording inconsistency** prevents either representation from transferring.

A useful outcome is therefore more than a higher score: we learn *which information helps, which transformation damages it, and whether the improvement holds across participants.*

These are explanations of proposed experiments; I have not run new evaluations. Their main risk is removing useful command information while trying to simplify or stabilize the EEG.


**E4–E6 improve how we extract information. E7–E9 improve how we combine models or train them. E10–E11 introduce carefully controlled adaptation. E12 asks whether learning from unlabeled EEG gives us a better starting representation.**

These are alternative hypotheses to test. They do not need to become one large architecture.

Recall our starting point: each trial contains **41 electrode signals over four seconds**, and we must predict motor imagery, subtraction or word generation.

**E4: CSP — learn which combinations of electrodes distinguish commands.**

E1 measured frequency power separately at each electrode. But useful activity may appear across several electrodes, mixed with other sources.

Instead of examining each electrode independently, we can form a new signal by combining them:

$$
z(s)=w_1x_1(s)+w_2x_2(s)+\cdots+w_{41}x_{41}(s).
$$

The weights $w$ define a **spatial filter**. “Spatial” means that we combine electrodes at the same moment, rather than combining different moments.

Imagine several microphones recording an orchestra. Adding some microphones and subtracting others can emphasize an instrument while reducing shared background sound. Spatial filtering attempts a related separation of scalp voltage patterns.

**Common Spatial Patterns, or CSP**, chooses combinations whose variance differs between classes. For a binary comparison, a simplified objective is:

$$
\max_w
\frac{w^\top C_A w}
     {w^\top (C_A + C_B) w}.
$$

Here, $C_A$ and $C_B$ describe the classes’ average covariance. The numerator measures the projected signal’s variance in class A. The denominator measures its combined variance across both classes.

CSP finds directions emphasizing one class, along with directions emphasizing the other.

Because we have three commands, **one-versus-rest CSP** makes comparisons such as motor imagery against subtraction and word generation combined. We retain a small number of useful projections, measure their log variance, and train a classifier.

```text
Electrode signals
      ↓
Learn a few weighted electrode combinations
      ↓
Measure each combination’s variance
      ↓
Classifier → command
```

**Filter-bank CSP, or FBCSP**, repeats this process within several frequency bands.

What are we attempting? **Find better places to measure power than the original electrodes alone.**

The risk is that CSP can learn participant-specific patterns, including artifacts. It therefore needs regularized covariance estimates, few components and participant-held-out evaluation.

The useful comparison is E1 versus CSP with the same frequency bands and classifier. That isolates the benefit of learning spatial combinations.

---

**E5: ShallowConvNet/FBCNet — learn power features directly from the classification objective.**

CSP uses a specific mathematical criterion to choose spatial filters. A neural network can instead learn its filters through prediction errors.

A typical ShallowConvNet-style pipeline is:

```text
EEG
 ↓
Temporal filters: detect signal patterns
 ↓
Spatial filters: combine electrodes
 ↓
Square → average → logarithm
 ↓
Classifier
```

The square-and-average stages behave like power estimation. This gives the network a strong prior: **rhythmic strength is likely to matter**.

“Shallow” means relatively few processing stages. It still learns meaningful transformations; it simply restricts how complicated those transformations can become.

A typical FBCNet-style approach begins with a fixed frequency bank, learns spatial filters within those bands, and summarizes their activity through variance-based features.

The distinction is useful:

| Approach | What is prescribed? | What is learned? |
|---|---|---|
| E1 | Bands and power calculation | Classifier |
| E4 | Bands and CSP criterion | Spatial projections and classifier |
| E5 | Power-oriented network structure | Filters and classifier through supervised training |

What are we attempting? **Let task labels refine the feature extractor while retaining a useful EEG-specific structure.**

This is attractive when a fully flexible network has too many ways to fit a small dataset. But the power bias could miss useful task timing.

We compare E5 with EEGNet and the strongest statistical baseline. A gain should survive comparable preprocessing, tuning opportunities and training budgets. We also inspect subtraction and word-generation recall rather than assuming a motor-imagery-oriented design suits all three commands.

---

**E6: Sinc filters — learn which frequency ranges matter, with fewer freedoms.**

A conventional temporal convolution learns each coefficient of a filter separately:

$$
z(s)=\sum_k h(k)x(s-k).
$$

With many freely learned coefficients, the model can discover useful filters—but it can also fit accidental details.

A **Sinc filter** constrains the filter to resemble a band-pass filter. Instead of learning every coefficient independently, it primarily learns a lower and upper frequency boundary.

Conceptually:

```text
Ordinary temporal filter:
“Learn an arbitrary waveform detector.”

Sinc temporal filter:
“Learn which frequency interval to pass.”
```

For our 120 Hz input, the cutoffs must satisfy:

$$
0 < f_{\mathrm{lower}} < f_{\mathrm{upper}} < 60 \text{ Hz}.
$$

The experiment asks whether this restriction improves generalization by reducing unnecessary freedom.

What are we attempting? **Learn useful frequency bands without asking the network to invent the entire filter shape.**

We keep the spatial processing, pooling and classifier matched, changing only the temporal filter family.

A pleasant-looking learned passband is not a successful result. The constrained model must improve held-out decoding. It could lose if broad band-pass filters cannot express the temporal patterns the task requires.

This experiment borrows one mechanism from BTTA; it does not require BTTA’s memory or GMM components.

---

**E7: an ensemble — combine models that make different mistakes.**

Suppose a power model and a covariance model assign probabilities to the same trial:

| Model | Imagery | Subtraction | Words |
|---|---:|---:|---:|
| Power model | 0.55 | 0.35 | 0.10 |
| Covariance model | 0.20 | 0.70 | 0.10 |
| Equal average | 0.375 | 0.525 | 0.10 |

These are illustrative numbers. The average predicts subtraction.

The opportunity is that the models may use different evidence. Power features could work well on some trials, while cross-channel relationships help on others.

What are we attempting? **Reduce individual model errors through complementary evidence.**

Two similar models that make the same mistakes may provide little benefit. Conversely, a weaker standalone model can sometimes improve an ensemble if its errors differ usefully.

We start with equal probability averaging because tuning many weights on a small validation set is another opportunity to overfit.

The ensemble must beat the best individual model under matched evaluation. Identical subtraction errors would show that it has not fixed that particular weakness; they would not automatically invalidate improvements elsewhere.

This combines predictions for the **same window**. It does not assume neighboring windows have the same command.

---

**E8: augmentation — teach the model which small recording changes to tolerate.**

During training, we create slightly altered versions of genuine EEG examples while retaining their labels.

Examples include modest amplitude changes, added noise or dropping some channels.

The intended lesson is:

> “A small recording disturbance should not change which command you predict.”

The critical assumption is that the disturbance actually preserves the command information.

For scalp EEG, dropping a channel can approximate a missing or unusable electrode. However, dropping all electrodes that capture an informative pattern may make the trial ambiguous. Large noise or aggressive cropping can create the same problem.

What are we attempting? **Encourage robustness to plausible disturbances without teaching the model to ignore its useful evidence.**

We test one augmentation family at a time against ordinary training with the same number of updates.

The geometry comparison asks whether dropping neighboring electrodes teaches something different from dropping the same number of scattered electrodes. A local failure removes one area’s coverage; scattered failures leave a different spatial pattern.

We must compare equal severity. Otherwise, an apparent benefit from geometry might simply reflect an easier perturbation.

---

**E9: multiple temporal scales — inspect both brief patterns and longer developments.**

E1 summarizes an entire window. A neural model might benefit from examining different durations separately.

A shorter temporal filter examines relatively local patterns. A longer filter sees how activity develops over a broader interval.

```text
                   ┌→ shorter temporal context ─┐
EEG window ────────┤                            ├→ combine → command
                   └→ longer temporal context ──┘
```

These are different **time scales**, not necessarily separate frequency bands. Filter duration affects what can be represented, but it does not uniquely determine a frequency response.

What are we attempting? **Preserve useful information that one temporal scale misses.**

For example, the progression of activity during a cognitive task might help distinguish commands beyond average power. That remains a hypothesis for this dataset.

The extra branch adds parameters and computation. We therefore compare it with a similarly sized single-scale model. If giving the single-scale model comparable capacity produces the same gain, we have not established that multiple scales were responsible.

---

**E10: confidence summaries — examine consistency within a window before adding memory.**

First, the decoder needs to output class probabilities at multiple temporal positions within the four-second window. Our current window-level output alone does not provide that sequence.

Consider two illustrative sequences of probabilities for one class:

```text
A: 0.60, 0.60, 0.60, 0.60
B: 0.95, 0.95, 0.25, 0.25
```

Both average to 0.60. But one provides consistent support, while the other is divided.

E10 asks whether a summary of that difference adds predictive information beyond the ordinary mean probability. Mean-log or Dirichlet-based summaries are candidate ways to describe the distribution of predictions.

A **Dirichlet distribution** is a distribution over possible class-probability vectors. Its concentration describes how tightly those vectors cluster. That statistical consistency is not automatically calibrated confidence about the true command.

What are we attempting? **Use disagreement within a trial as additional evidence.**

These summaries generally discard temporal order. They can distinguish consistency from fluctuation without distinguishing “support increased” from “support decreased.”

Only if the summaries help do we test a bounded memory of earlier current-session windows. That introduces a new risk: confidently incorrect examples can contaminate the memory and reinforce future mistakes.

The method must work from the first window, cope with missing classes and reset every session.

---

**E11: a small adapter — correct the input mapping while protecting the decoder.**

We freeze a trained decoder and place a small adjustable transformation before it:

$$
\hat{y}=f_\theta(A_\phi(X)),
\qquad \theta \text{ fixed}.
$$

The adapter initially implements the identity:

$$
A_{\phi_0}(X)=X.
$$

That means the starting prediction matches the existing decoder.

As current-session EEG arrives, the adapter makes limited adjustments intended to compensate for recording changes. An alignment loss can encourage selected statistics to resemble the training reference. An identity penalty discourages large transformations.

What are we attempting? **Improve the match between current recordings and the decoder’s expected input without retraining the entire decoder.**

This extends E3. Normalization applies relatively prescribed corrections; a learned adapter has more flexibility.

More flexibility also creates more ways to fail. An adapter can reduce covariance mismatch while pushing subtraction and imagery closer together. Lower alignment loss can therefore accompany worse decoding.

Freezing the classifier does not eliminate forgetting: changing its input changes its behavior.

This is where our sleep-paper discussion fits. A behavior-preservation penalty could discourage destructive updates, but it could also preserve incorrect predictions. We must measure both adaptation gain and retention separately.

---

**E12: masked self-supervision — learn EEG structure before learning command labels.**

Here, we first train an encoder using EEG from the training partition without requiring command labels for that objective.

We hide parts of a recording and ask the model to predict something about the missing portion.

There are two main objective choices:

| Objective | What does the model predict? |
|---|---|
| **MAE: masked autoencoding** | Missing signal values or patches |
| **JEPA-style latent prediction** | A teacher encoder’s representation of the hidden content |

For MAE, the model might infer a missing waveform from surrounding signals. For JEPA, it predicts an encoded description rather than every voltage value. The teacher is typically updated slowly from the student’s parameters.

We also compare scattered masks with contiguous missing regions:

```text
Scattered:    visible / hidden / visible / hidden / visible

Contiguous:   visible / visible / hidden / hidden / hidden
```

Scattered gaps may be solved through local interpolation. Larger contiguous gaps can demand broader structure, but may also remove too much information.

What are we attempting? **Learn a representation that makes subsequent supervised command decoding easier.**

The main trap is that reconstructing EEG is not equivalent to understanding the command. The model might become excellent at reconstructing shared artifacts or predictable background rhythms.

JEPA adds another risk: representations might become uninformative or rely excessively on position rather than EEG content. We therefore test whether changing the signal while keeping positions fixed changes the embeddings meaningfully.

After pretraining, we evaluate command decoding against the same encoder trained directly with labels, including a compute-matched control. Held-out participants and reserved later sessions must be excluded from pretraining.

**The overall progression is deliberate:** first establish which features work, then improve their extraction and training, and finally test whether adaptation earns its additional risks.

These remain proposed mechanisms and comparisons; I have not run new experiments. None of their apparent sophistication establishes a gain on our dataset.