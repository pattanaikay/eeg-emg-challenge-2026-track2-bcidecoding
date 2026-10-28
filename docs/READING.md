# Reading route: one question per source

**Detailed companion review (21 October 2026):** [Browsable HTML](LITERATURE_REVIEW.html) · [Markdown](LITERATURE_REVIEW.md). It reviews every scientific paper and both company reports below, with verified results, evidence limits, comparison tables, and proposed experiments for retention and harmful-drift prediction.

Start with the contract, then EEG signal/representation basics, then drift and forgetting. Read the abstract, main figure, assumptions and limitations first. Write what transfers to our setting and what still needs a new experiment.

## Competition and implementation

| Source | Read for |
|---|---|
| [Official scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md) | Actual tensor, metadata restrictions, chronology, reset and permitted adaptation |
| [Track 2 worked kit](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/bci_decoding) | Official loader, solvers, objective and export contract |
| [Submission guide](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md) | Flat ZIP, `CompetSolver`, read-only inference and common errors |
| [PROTEUS/NEMAR nm000290](https://nemar.org/dataset/nm000290) | Public data access and license |
| [PROTEUS data README](https://github.com/nemarDatasets/nm000290) | Reference electrode, public sessions, tasks and known acquisition issues |
| [NeuralBench streamed Track 2 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track2_eeg_to_bci.html) | Model/task environment; some prose is older than the FAQ, so prefer the active contract for conflicts |
| [Benchopt documentation](https://benchopt.github.io/) | Optional upstream benchmark runner, not a new neural method |

## First three scientific papers

1. **Rodrigues, Jutten & Congedo (2019), Riemannian Procrustes Analysis: Transfer Learning for Brain–Computer Interfaces.** [Primary record](https://pubmed.ncbi.nlm.nih.gov/30596565/), DOI `10.1109/TBME.2018.2889705`. Read for covariance geometry and alignment. Ask which transformations retain class information and what calibration data are required.
2. **Pun et al. (2024), Measuring instability in chronic human intracortical neural recordings towards stable, long-term brain-computer interfaces (MINDFUL).** [Paper](https://www.nature.com/articles/s42003-024-06784-4). Read for label-free instability monitoring evaluated against fixed-decoder behavior. Its invasive cursor results are motivation, not a validated EEG monitor.
3. **Kirkpatrick et al. (2017), Overcoming catastrophic forgetting in neural networks (EWC).** [Primary record](https://pubmed.ncbi.nlm.nih.gov/28292907/), DOI `10.1073/pnas.1611835114`. Read for protecting important parameters and the stability/plasticity tradeoff. Ask when parameter preservation conflicts with necessary adaptation.

## Model and adaptation references

| Paper | Why read it / question |
|---|---|
| [Lawhern et al., EEGNet (2018)](https://pubmed.ncbi.nlm.nih.gov/29932424/) · DOI `10.1088/1741-2552/aace8c` | Compact temporal/spatial EEG learning. What does a small supervised model already capture? |
| [Jiang, Zhao & Lu, LaBraM (ICLR 2024)](https://arxiv.org/abs/2405.18765) | Channel patches and learned neural codes for EEG pretraining. What does its target preserve? |
| [Niu et al., EATA (ICML 2022)](https://proceedings.mlr.press/v162/niu22a.html) | Adaptation filtering and anti-forgetting regularization. What happens to confidently wrong samples? |
| [Karpowicz et al., NoMAD (2025)](https://www.nature.com/articles/s41467-025-59652-y) | Learned dynamics for unsupervised invasive-signal stabilization. Which invariants could survive scalp mixing? |
| [Neuralink: Pretraining on 50,000 Hours of Unlabeled Brain Data](https://neuralink.com/updates/pretraining-on-50000-hours/) | A company report about invasive pretraining. Useful ideas do not establish an EEG scaling law or an EEG data requirement. |
| [Official masked EEG pretraining tutorial](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_pretrain_mae.html) | How the existing stack implements masked learning; not a mandate to download terabytes before a baseline works. |

## Interpretation and clinical horizon

| Source | Why read it / limitation |
|---|---|
| [Haufe et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/24239590/) · DOI `10.1016/j.neuroimage.2013.10.067` | Predictive weights are not straightforward neural-source maps. |
| [Nolte et al. (2004)](https://pubmed.ncbi.nlm.nih.gov/15351371/) · DOI `10.1016/j.clinph.2004.04.029` | Volume conduction and interaction estimates; covariance alone is not connectivity. |
| [Sani et al., PSID (2021 issue; online 2020)](https://www.nature.com/articles/s41593-020-00733-0) | Prioritize behaviorally relevant linear dynamics. Which behavior labels are available? |
| [Sani, Pesaran & Shanechi, DPAD (2024)](https://www.nature.com/articles/s41593-024-01731-2) | Nonlinear behaviorally relevant dynamics. “Causal decoding” refers to present/past inputs, not a proved biological cause. |
| [Goodfire / Prima Mente, Alzheimer's biomarker report (2026)](https://www.goodfire.com/research/interpretability-for-alzheimers-detection) | Model inspection → hypothesis → simpler explanatory features → independent validation. Candidate biomarkers do not establish gene causality. |
| [Saez, The emerging field of cognitive brain–computer interfaces (2026)](https://doi.org/10.1016/j.tics.2026.06.012) | Your supplied clinical review: distributed cognition, imperfect labels, and proposed closed-loop systems. Competition decoding does not establish a clinical biomarker. |
| [NWB](https://nwb.org/) and [BIDS](https://bids.neuroimaging.io/) | Existing data/metadata standards for later portability; a common schema does not make all modalities interchangeable. |

## How to turn a reading into an experiment

For each paper, record: its data/modality, labels required, split, inference information, baseline, main limitation, and one falsifiable EEG experiment. Do not copy a metric, hyperparameter or architectural result into this competition without matching its setting and checking the actual source.
