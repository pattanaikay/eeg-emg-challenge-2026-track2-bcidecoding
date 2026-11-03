# Agent D in plain English: NoMAD, drift, and stable dynamics

## The one-sentence version

NoMAD tries to keep a decoder's useful internal dynamics fixed while learning a small input adapter that makes a changed recording look like the old recording.

It is a strong inspiration for drift handling, but it was demonstrated on invasive recordings from individual animals—not on cross-person scalp EEG or our PROTEUS competition.

## The Kramer explanation

Jerry, imagine a band has the same song every night, but the club keeps changing the microphones. The guitar signal becomes louder, the drums become quieter, and one microphone is moved to the other side of the stage.

NoMAD says: do not rewrite the song or retrain the entire band. Keep the old music model and add a small sound engineer at the entrance who adjusts the changed microphone signals until they resemble the old recording conditions.

If the microphones still capture enough of the song, the old decoder can work again. If a microphone has stopped hearing the guitar entirely, no amount of alignment can reconstruct the missing guitar. And if the engineer matches only the average loudness, the song can still be wrong.

That is the central NoMAD lesson: alignment can correct a measurement change, but it cannot manufacture task information that was never recorded.

## What “latent dynamics” means

The decoder first turns a sequence of neural observations into a hidden state:

`z_t = fθ(x_1, x_2, ..., x_t)`

The hidden state is meant to capture the evolving motor condition—roughly, where the movement is in time and what it is doing. A fixed readout then maps that state to behavior:

`ŷ_t = g(z_t)`

On a later recording day, the observations change. NoMAD inserts an adapter `Aφ` before the fixed decoder:

`z'_t = fθ(Aφ(x'_1), ..., Aφ(x'_t))`

The adapter is trained so that the target recording's latent states have similar distributions to the reference recording's states, while still reconstructing the observed neural activity. The recurrent core and the behavior readout are intended to remain stable.

## What was actually demonstrated

The paper used two macaques with 96-channel motor-cortex recordings during force and reaching tasks across many sessions. It also included a retrospective analysis of one human intracortical participant. The headline result was improved within-individual recording transfer compared with other alignment methods.

The target-day alignment used neural data without behavior targets in the main monkey protocol, but it was fitted from the target session before evaluation. That is transductive adaptation. It is not the same as receiving one four-second PROTEUS window at a time and having to reset at a session boundary.

The paper did not demonstrate:

- cross-person scalp-EEG transfer;
- three-command classification;
- robustness to 41-channel electrode mixtures;
- a minimum-calibration curve after one or a few windows;
- that the aligned latent coordinates are unique biological causes.

## What is fixed and what learns

The intended division is:

`fixed recurrent dynamics + trainable input/alignment adapter + fixed behavioral readout`

The adapter can learn how the new recording's scale, channel mapping, or activity pattern should be transformed. This is attractive for catastrophic-forgetting control: because the core decoder is frozen, adaptation cannot directly overwrite all of the learned task mapping.

But freezing is not the same as being safe. A flexible adapter can still map different commands into the wrong latent regions. A low alignment loss can fall while classification balanced accuracy falls.

## The mathematical trap: matching moments is not matching meaning

NoMAD aligns pooled latent statistics, including means and covariances. Suppose the target latent distribution is a mixture of commands:

`p(z) = Σ_y p(z | y) p(y)`

Its mean and covariance can change because the command frequencies changed, even if every command-specific representation stayed stable. Conversely, the pooled mean and covariance can stay similar while one minority command collapses into another.

A rotation, class permutation, or loss of minority-class separation can preserve pooled moments. Therefore:

`low alignment loss ≠ preserved task information`

The only convincing safety test is behavioral: does the adapter preserve or improve per-class, per-participant, per-session balanced accuracy under a causal protocol?

## What to underline while skimming

**The architecture diagram.** Identify what is frozen and what is trainable. This is the paper's most reusable design principle.

**The target-session calibration description.** Mark every place where target data, session statistics, behavior labels, or future samples enter. This determines whether the method is causal or transductive.

**The alignment loss.** Understand the two jobs: neural reconstruction keeps the adapter tied to observed activity; latent-distribution matching keeps it near the reference dynamics. Neither term directly measures command accuracy.

**The ablations.** Notice that some comparisons remove several components together. They support the complete procedure but do not isolate the unique causal contribution of the recurrent core, KL term, normalization, or adapter.

**The identifiability issue.** A good-looking latent plot is not proof that a coordinate is “the motor variable.” Different latent coordinate systems can produce the same behavior.

## The competition translation

PROTEUS is a much harder and different deployment boundary:

- scalp EEG is a spatial mixture, not a clean motor-cortex spike population;
- there are 41 fixed channels and 480 samples per window;
- the model receives no participant or session ID;
- target labels and extra calibration labels are unavailable;
- state must reset at session boundaries;
- only current and earlier windows may influence adaptation.

The smallest transferable hypothesis is not “build NoMAD.” It is:

`frozen EEG classifier + identity-initialized small adapter + causal normalization`

Use only unlabeled current/past windows in the session. Compare no adaptation, normalization alone, and the adapter with a weak alignment penalty. Keep the classifier fixed at first so we can distinguish adaptation from catastrophic forgetting.

## The falsifier

Reject alignment as a useful safety signal if its loss decreases while balanced accuracy or a class recall decreases. Reject the adapter if it provides no gain over normalization alone, requires future-session statistics, or only helps one participant/interface while harming others.

Before trusting any borrowed implementation, check prefix invariance: predictions already emitted from a stream must not change when future windows are appended. A demo utility that fills initial outputs from the first complete window would fail this test even if its later predictions look good.

## Source

Brianna M. Karpowicz et al., “Stabilizing brain-computer interfaces through alignment of latent dynamics,” *Nature Communications* 16, 4662 (2025). [Full article record](https://www.nature.com/articles/s41467-025-59652-y) · [Verified full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12089531/fullTextXML) · [Pinned code](https://github.com/snel-repo/nomad/tree/aa3c8faadf053d0b22160458854c076dcb13eb7e).
