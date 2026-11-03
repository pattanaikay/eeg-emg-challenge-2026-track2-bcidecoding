# Agent A in plain English: what BTTA-DG is really doing

## The one-sentence version

BTTA-DG does not magically relearn EEG. It takes a trained motor-imagery classifier, watches its confidence pattern on the new stream, and uses a small memory of confident recent predictions to gently re-weight future predictions.

That is the useful idea. The paper's reported gains are on binary motor-imagery datasets, not on our three-command PROTEUS competition, and the released implementation has deployment problems that we must not copy blindly.

## The Kramer explanation

Jerry, imagine the decoder is a bouncer at a club. It looks at a new EEG trial and says, “Probably left hand: 80%, right hand: 20%.” That is the bouncer's first opinion.

Now imagine the bouncer has been working in a new city. The faces look different, the lighting is different, and the old rules are slightly off. BTTA-DG gives the bouncer a notebook. Every time the bouncer is very confident, it writes down the *shape of its confidence*, not just the final label. After enough notes, it learns what “a normal left-hand-looking trial in this new city” tends to look like.

For the next trial, it combines two opinions:

`final opinion = old classifier opinion × how much this trial resembles the recent left/right notebooks`

Kramer would say: “You don't repaint the whole club. You just notice that the bouncer's left-hand guesses have a particular vibe tonight.” That is BTTA-DG: a lightweight calibration layer around a frozen or mostly frozen decoder, not a proof that the brain signal itself has become stationary.

## The actual pipeline

The paper has three main pieces.

### 1. A frequency-aware classifier

The front end, SincAdaptNet, learns band-pass filters. Instead of fixing every frequency band by hand, it learns lower and upper cut-offs for several filters and then combines them with spatial convolutions. This is a sensible EEG prior because motor imagery often changes rhythmic activity in bands such as mu and beta. It is still a learned filter bank, not a guarantee that every learned filter is physiologically meaningful.

For our data, the sampling rate is 120 Hz. The Nyquist limit is 60 Hz, so any new implementation must keep learned upper cut-offs below 60 Hz. The released code's 100-Hz upper cap cannot be copied unchanged.

### 2. A compact summary of the classifier's time-varying confidence

The classifier produces a probability vector at several time steps. For a binary example, one time step might be `[0.8, 0.2]`, another `[0.6, 0.4]`, and another `[0.9, 0.1]`.

BTTA-DG compresses the whole sequence of probability vectors into Dirichlet parameters, written conceptually as `alpha`. The components of `alpha` say which class the window tends to favour; the total concentration `alpha_0 = sum(alpha)` says how concentrated or diffuse the evidence is.

The important limitation is easy to miss: this summary forgets order. If the same probability vectors are shuffled in time, the summary can remain the same. It retains “which class is winning and how decisive the evidence looks,” but not “when the evidence appeared,” transitions, or the detailed temporal waveform. It cannot recover information that SincAdaptNet already discarded.

### 3. A class-conditional GMM memory

The method stores high-confidence summaries in a memory bank labelled by the decoder's own predicted class. A Gaussian mixture model (GMM) estimates where recent summaries for each class live. For a new trial, the network supplies its normal probability, while the GMM supplies a compatibility score. The product is normalized into the final probability.

This is gradient-free: no neural-network weights need to be updated at test time. That is attractive for a streaming competition because a bad update cannot directly destroy the network. But the memory is still pseudo-label based. If the decoder is confidently wrong, it can place a wrong summary in the wrong class bank and make later mistakes more likely. “No gradient” does not mean “no error accumulation.”

## What the paper actually demonstrated

The authors studied binary motor imagery, mostly left/right hand or right-hand/feet, with 3–32 EEG channels and different sampling rates. They reported strong accuracies on their datasets and a within-subject cross-session result on BNCI2014001. Those numbers are useful evidence that the mechanism can be worth testing; they are not a PROTEUS result, not three-class balanced accuracy, and not evidence for our exact 41-channel 120-Hz stream.

The paper's “pretraining” is supervised source-participant training. It is not a demonstration that a huge external EEG foundation model transfers to our competition.

## What to underline while skimming

Read these anchors first:

**Abstract and introduction.** Look for the precise claim: gradient-free test-time calibration using a learned frequency front end, a Dirichlet summary, and GMM-based inference. Do not read “Bayesian” as proof that the posterior is biologically correct; it is a probabilistic calibration construction.

**The SincAdaptNet architecture section.** Note the learnable frequency bands and the spatial mixing. This is the most portable part for a first controlled experiment because it can be compared with EEGNet without requiring target labels or a memory bank.

**The Dirichlet feature-projection equations.** Ask: what is preserved, and what is destroyed? Preserved: class preference and concentration/dispersion. Destroyed: temporal order and any information absent from the network output.

**The GMM/Bayesian inference equations and Algorithm 1.** Trace the information flow: historical pseudo-labelled summaries build the class memories; the current trial is scored against them; the current summary is then eligible for insertion. This ordering matters for a causal stream.

**The cross-session experiment.** Check exactly what is called “cross-session,” what labels or target information are used, and whether it resembles our session-reset protocol. Similar words do not imply the same evaluation contract.

**Appendices and released code.** The audit found that the public path has an unused sample-insertion function, uses target labels during checkpoint selection, estimates alignment from the whole target array, and has an upper frequency cap invalid for a 120-Hz port. These are reasons to reproduce only isolated pieces, not reasons to claim the paper's published experiments are all invalid.

## The competition translation

Our competition has three commands: motor imagery, mental subtraction, and word generation. A binary left/right motor-imagery method therefore has a narrower task than ours. We receive one `(41, 480)` EEG window at a time, with no participant ID, session ID, or label. Adaptation must reset at session boundaries and may use only the current and earlier windows in that session.

The safe translation is therefore:

1. First test only the learned Sinc frequency front end against the existing EEGNet, with the same participant split, training budget, and checkpoint rule.
2. Separately test whether confidence summaries help, using a frozen source-trained model and a simple causal memory control.
3. Add the GMM only if the summary itself helps and only with explicit cold-start, missing-class, reset, and wrong-pseudo-label safeguards.

The paper should inspire experiments, not dictate the final architecture.

## The question that decides whether it is useful

Does the method preserve task information while tolerating recording variation?

For our first test, the falsifier is straightforward: if the Sinc front end or the confidence summary does not improve participant-held-out cell-balanced accuracy across several seeds—or improves one participant while harming another—we should not add the more complicated GMM. A method that looks elegant but amplifies confident mistakes is harmful drift adaptation, not successful adaptation.

## Source

Huan Luo, Na Lu, Xiaopeng Wang, and Xu Niu, “Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding,” ICLR 2026. [Official proceedings entry](https://proceedings.iclr.cc/paper_files/paper/2026/hash/36eeec256f89eba9ed60cbca73ea92ed-Abstract-Conference.html) · [Full paper PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/36eeec256f89eba9ed60cbca73ea92ed-Paper-Conference.pdf) · [Pinned implementation audit](https://github.com/luo-huan-123/BTTA-DG/tree/5932d026bbd8a7de106d31a6d264f4f4924537e4).
