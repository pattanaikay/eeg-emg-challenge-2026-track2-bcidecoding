# Agent C in plain English: masking and EEG representation learning

## The one-sentence version

Agent C asks a simple question: when we hide parts of an EEG recording during self-supervised learning, what kind of hiding teaches an encoder useful brain structure rather than easy interpolation, electrode-position memorization, or artifact reconstruction?

The paper is valuable mainly as a warning and an experimental template. It does **not** prove that its winning mask is optimal for our four-second PROTEUS windows.

## The Kramer explanation

Jerry, imagine Kramer gives a student a jigsaw puzzle with some pieces covered and says, “Guess what goes under the paper.”

If Kramer covers one isolated piece but leaves all its neighbours visible, the student can fill it in by copying local texture. Great score, but perhaps no understanding of the picture.

If Kramer covers an entire corner, the student may learn that “corner-shaped things usually belong here.” That can be useful, but it can also become a position trick.

EEG masking has the same problem. A model can reconstruct a hidden voltage because nearby time samples and neighbouring electrodes already reveal it. Or it can learn that a particular electrode position usually has a certain variance. Neither result proves that it learned the task distinction between motor imagery, subtraction, and word generation.

The practical question is not “Did the model reconstruct the hidden signal?” It is “Did the model retain information that helps classify a new participant or session?”

## What is being masked?

Let an EEG window be a matrix `X ∈ R^(channels × time)`. A mask `M` hides some channel-time patches.

### MAE: reconstruct the missing voltage

The masked autoencoder receives the visible part and predicts the raw EEG values at the hidden positions:

`loss_MAE = || X_hidden − decoder(encoder(X_visible)) ||²`

This rewards preservation of predictable voltage structure. That structure may include real task information, but it may also include mains noise, eye artifacts, muscle bursts, reference effects, and redundant scalp mixtures.

### JEPA: predict a teacher representation

JEPA does not necessarily reconstruct the exact voltage. A slowly updated teacher network encodes the hidden region, and the student predicts that hidden representation:

`loss_JEPA = || teacher(X_hidden) − predictor(encoder(X_visible)) ||²`

This lets the teacher discard some low-level detail. That may help avoid chasing noise. But teacher agreement is not automatically task relevance: the teacher can discard a useful class cue or preserve a nuisance cue.

## Why mask geometry matters

There are two independent geometries.

**Spatial geometry:** hide scattered channels, neighbouring electrodes, or contiguous scalp patches. Scalp EEG has volume conduction, so nearby electrodes share mixtures of sources. Hiding one electrode while leaving its neighbours may make reconstruction almost trivial.

**Temporal geometry:** hide isolated time samples, contiguous time blocks, or whole temporal regions. EEG is autocorrelated, so a short missing point can be interpolated from its neighbours. A long block forces the model to use slower rhythms and cross-channel context—but may remove the very evidence needed to distinguish a command.

The paper's headline nominal winner was a small spatial patch (reported as two patches of roughly 9 cm), but many configurations were close. That is a comparison result inside their large pretraining recipe, not a universal PROTEUS hyperparameter.

## What the study actually showed

The authors trained 58 MAE/JEPA variants on a very large multi-dataset corpus and tested frozen representations on twelve downstream EEG datasets. Their all-channel temporal-masking JEPA condition showed a particularly dangerous shortcut: the representation could be dominated by electrode-position information while decoding of low-frequency task structure was poor.

That matters because “the model knows where the electrode is” is not the same as “the model knows what the person is doing.” The study also reports close results among many mask settings and uses probe seeds rather than repeated full pretraining seeds. The evidence supports a useful controlled comparison, not a single proven best mask.

## What to underline while skimming

**The mask-definition figure and methods.** Ask exactly which channels and time patches are hidden, what fraction is actually hidden after overlap, and whether the decoder receives the same amount of visible context across conditions.

**The MAE versus JEPA objective.** MAE asks for voltage reconstruction. JEPA asks for agreement with a teacher representation. Neither objective directly says “preserve class information.”

**The position-bias result.** Treat this as a shortcut diagnostic. If the embedding changes when electrode labels change but the EEG content is held fixed, it is using geometry. If it barely changes when EEG content is exchanged between trials while positions stay fixed, it may be ignoring the signal.

**The transfer tables.** Check whether a downstream dataset or participant appeared in pretraining. A later held-out probe cannot undo upstream exposure. Also check whether the downstream split is by participant, session, run, or trial.

**The limitations and compute comparison.** Equal token counts do not guarantee equal decoder work, teacher work, or total FLOPs. A masking result selected on a large corpus may not survive our small in-domain dataset.

## The smaller-data comparison

The report also reviewed EEG2Rep, which uses subsequence-preserving context and latent prediction. Its results support one recipe on several datasets, including an average improvement over random masking and raw-input targets. That is encouraging, but it is not a universal latent-space advantage, and its training code creates a selection-risk pathway by printing probe results during pretraining.

The lesson for us is to reproduce the comparison under a clean participant split rather than import the winning label.

## The competition translation

Our data are only 41 EEG channels × 480 samples at 120 Hz, with three commands and a fixed corpus. External foundation-model checkpoints are outside our primary single-corpus comparison.

The smallest honest test is a matched 2×2 experiment:

`MAE vs JEPA` × `scattered vs contiguous spatial-temporal masks`

Use participant-separated splits **before** self-supervised training. Match the realized masked fraction, optimizer steps, decoder capacity, and total compute. Compare every SSL variant with:

`supervised scratch + EEGNet`, `supervised band-power/covariance baselines`, and a `random encoder control`.

Then measure official participant × session × interface balanced accuracy, per-class recall, and whether the embedding changes when EEG content is swapped while channel positions remain fixed.

## The falsifier

Reject the objective or mask if lower SSL loss does not improve held-out task balanced accuracy over matched scratch training, if the gain disappears under participant rotation, or if the representation preserves electrode position while losing spectral or command information.

In plain English: a model that gets very good at guessing hidden EEG but does not decode a new person is solving the puzzle, not the competition.

## Source

Pierre Guetschel, Bruno Aristimunha, Yassine El Ouahidi, Arnaud Delorme, Thomas Moreau, and Michael Tangermann, “What masking geometry works best for EEG foundation models?”, arXiv:2609.33487 (preprint). [Paper](https://arxiv.org/html/2609.33487) · [MAE implementation](https://raw.githubusercontent.com/PierreGtch/eeg-fm-masking/main/eeg_fm_masking/pl_mae.py) · [JEPA implementation](https://raw.githubusercontent.com/PierreGtch/eeg-fm-masking/main/eeg_fm_masking/pl_ssl.py).
