Riemannian features are a way of representing EEG by its **channel-to-channel covariance structure** rather than by raw voltage samples.

### Intuition

Suppose two EEG channels increase and decrease together during a mental task. Their covariance is high. If one increases while the other decreases, covariance is negative.

For one EEG window:

$$
X \in \mathbb{R}^{41 \times 480}
$$

we compute a covariance matrix:

$$
\Sigma = \frac{1}{T-1}XX^\top
$$

This produces a $41 \times 41$ matrix describing relationships between all electrode pairs.

A covariance matrix is not an ordinary Euclidean vector. It is a symmetric positive-definite (SPD) matrix, so Riemannian methods use geometry designed for these matrices.

### Why “Riemannian”?

If we compare covariance matrices using ordinary subtraction, we may get misleading distances. Riemannian geometry measures distance along the curved space of valid covariance matrices.

A common distance is:

$$
d_R(A,B)
=
\left\|
\log\left(A^{-1/2}BA^{-1/2}\right)
\right\|_F
$$

In plain language: it asks how much one spatial covariance pattern must be transformed to become another.

### Typical pipeline

A practical Riemannian EEG decoder often does this:

1. Band-pass EEG into ranges such as mu and beta.
2. Compute a covariance matrix for each band.
3. Apply shrinkage to make the covariance estimate stable.
4. Compute a reference covariance, often a geometric mean.
5. Map each covariance matrix into a tangent-space vector.
6. Classify using LDA, logistic regression, or a small ensemble.

The tangent-space step converts curved covariance objects into ordinary vectors that standard classifiers can use.

### Analogy

EEGNet looks at the evolving waveform:

> “What happens in this signal over time?”

A Riemannian decoder looks at the spatial coordination pattern:

> “Which electrodes fluctuate together, and how does that coordination differ between commands?”

This can be useful because EEG task information is often distributed across coordinated electrode activity rather than one isolated channel.

### Why it may help cross-session decoding

Riemannian features can be less sensitive to:

- overall amplitude changes;
- electrode gain differences;
- some reference changes;
- moderate spatial mixing and volume conduction;
- variations in absolute waveform shape.

However, this is not guaranteed drift invariance. A changed electrode montage, strong artifact pattern, or altered task strategy can still change the covariance structure.

### Important limitations

Covariance features discard a lot of information:

- precise temporal order;
- phase evolution over time;
- waveform shape;
- event timing;
- the mean signal.

With 41 channels, the covariance matrix contains:

$$
\frac{41 \times 42}{2}=861
$$

unique entries. Estimating 861 relationships from only 480 samples can be noisy, so shrinkage is important.

A covariance representation can also preserve nuisance structure. For example, eye movement or muscle activity may create a stable covariance pattern that the classifier mistakes for a command.

### In our competition

A fair experiment would compare:

```text
EEGNet
versus
shrinkage covariance + tangent-space features + LDA
versus
EEGNet combined with covariance features
```

All preprocessing must be fitted using training participants only. We should report:

- participant-held-out balanced accuracy;
- per-class recall;
- performance on the repeated-session participant;
- whether subtraction or word-generation recall collapses;
- sensitivity to session-specific normalization.

The key question is not simply:

> “Does the Riemannian model score higher?”

It is:

> “Does it preserve class separation when the participant or session changes?”

Riemannian features are therefore a strong, lightweight baseline for our project—not automatically a superior replacement for EEGNet.