# What we trained and how to submit it

Prepared 21 October 2026. This is the plain-language explanation of the first real-data GPU batch. Exact commands, hashes, environment details and operational issues are in [JARVISLABS_RUN_LOG.md](JARVISLABS_RUN_LOG.md).

## The seed 35 file

Use this existing package:

**[eegnet-seed35-20261021T164141Z.zip](../artifacts/jarvis/20261021T162400Z/eegnet-seed35-20261021T164141Z.zip)**

It is already downloaded to the local project. Its size is **14,135 bytes** and its SHA-256 is:

```text
c91ec7811a08471d962962cd02d421ca304e2f82f20cec22ccfbca0022c90518
```

Its three files are directly at the ZIP root:

| File | Purpose |
|---|---|
| `submission.py` | Builds the model, loads its weights and predicts a class for each EEG window. |
| `weights.pt` | The numerical parameters learned during training. |
| `model.json` | Identifies EEGNet, initialization seed 35 and the expected input dimensions/channel order. |

The ZIP is the submission artifact. The larger result archive contains logs and experiment evidence and should be kept as a backup.

## What EEGNet is

EEGNet is a compact convolutional neural network designed for EEG classification. It learns patterns over time and combinations of signals across electrodes. Its architecture uses depthwise and separable convolutions to learn these features with relatively few parameters. Our project uses Braindecode's implementation of the established [EEGNet architecture by Lawhern and colleagues](https://arxiv.org/abs/1611.08024).

The architecture is the model's blueprint. Training adjusts its weights using labeled examples. We trained this supervised baseline from a random initialization on public PROTEUS data. Foundation-model pretraining and new drift/adaptation methods were outside this batch.

Our model receives one four-second EEG window: **41 channels × 480 samples at 120 Hz**, spanning 1–5 seconds after the cue. It returns one class:

- `0`: motor imagery;
- `1`: mental subtraction;
- `2`: word generation.

The challenge input for this track contains EEG channels only. Windows arrive chronologically, one at a time, within a participant-session stream. Each session starts with a fresh model copy. These baseline models keep their learned weights fixed during inference.

We also ran a simple linear reference model. It averages each channel over time and applies a learned linear classifier.

## What seeds 33, 34 and 35 mean

A seed sets the random-number sequence used by training. It affects choices such as initial weights, training sampling and dropout. Running EEGNet with three seeds lets us observe variation caused by training randomness.

All three EEGNet runs used the same architecture, hyperparameters, preprocessing and participant split. The numbers identify random sequences; they are not participant IDs, epochs or model version numbers. A seed helps reproduction within the recorded environment, but does not by itself guarantee identical results across different hardware/software stacks.

There are two separate uses of a seed here:

- **Participant split seed: always 33.** This fixes which people belong to training, validation and public test.
- **Training initialization seed: 33, 34 or 35.** This changes training randomness while keeping those people in the same partitions.

The three runs therefore use one held-out-participant split. They are not three cross-validation folds.

## The data and what each partition does

The public release has **10 participants, 14 participant-sessions and 112 runs**. We used the pinned official loader and preprocessing.

| Partition | Participants | Windows | Use |
|---|---|---:|---|
| Training | P01, P02, P03, P06, P07, P10 | 1,521 | Fit the model's weights. |
| Validation | P05, P08 | 379 | Choose the saved checkpoint. |
| Public test | P04, P09 | 755 | Measure the selected checkpoint's held-out performance. |

All sessions of a participant stay in one partition. No sealed evaluation data or labels were used for fitting.

An epoch is one training cycle through the training loader. EEGNet ran for 20 epochs. After each epoch, the code measured validation performance and retained the best checkpoint. For seed 35, **epoch 17** was best, so its weights were packaged even though training continued through epoch 20.

## What we executed

1. Inspected the current source, competition contract, account and CLI. Preserved the local uncommitted source state in a clean snapshot.
2. Uploaded a source-only ZIP, checked its archive hash remotely and verified all 58 extracted source files. All 30 vendored official files remained unchanged.
3. Used the user-created, on-demand L4 instance **542193**, named `eeg-foundation-challenge-2026`. Its existing persistent storage was 100 GB.
4. Prepared a separate persistent Python **3.12.13** environment because the template's Python 3.10 was too old. The final project environment used CUDA-enabled **torch 2.12.1** and **Braindecode 1.8.1**.
5. Ran the existing setup script, including doctor and the full synthetic EEGNet smoke check. Then passed all **10 contract tests**. Synthetic checks establish workflow behavior, not EEG accuracy.
6. Downloaded the public dataset, ran official preprocessing, audited the data and explicitly verified participant membership and input dimensions.
7. Ran the existing training script sequentially for linear/33, EEGNet/33, EEGNet/34 and EEGNet/35. Each next run waited for the preceding run's checks to succeed.
8. Reloaded every saved checkpoint and confirmed that both validation and test replay metrics exactly matched their original metrics. Validated every exported ZIP in an isolated process.
9. Downloaded the completed model directories, ZIPs, logs, package listing, source evidence and GPU observations. All **169 downloaded evidence-file checksums** matched their remote counterparts.
10. Paused the instance and verified its actual **Paused** state through the live account API at **2026-10-21 16:45:25 UTC**.

The conservative active-time upper bound was **25.43 minutes**, within the authorized two hours. The account's L4 compute rate was **INR 41.31/hour**. The corresponding conservative compute-cost upper estimate was **INR 17.51**, excluding storage and taxes; this is an estimate, not an invoice. Persistent storage remains billable while paused. Remote wall-clock timestamps were about 64 seconds behind local UTC; the detailed log explains the timing records.

## Results

The score is **participant × session × interface cell-averaged balanced accuracy**: compute mean class recall within each cell, then average the cells equally. It differs from ordinary percentage-correct accuracy.

| Model | Training seed | Epochs run | Selected epoch | Validation | Public test |
|---|---:|---:|---:|---:|---:|
| Linear | 33 | 30 | 2 | 44.59% | 37.97% |
| EEGNet | 33 | 20 | 20 | 45.39% | 35.24% |
| EEGNet | 34 | 20 | 7 | 45.42% | 38.28% |
| EEGNet | 35 | 20 | 17 | 46.71% | 36.39% |

EEGNet settings were Adam, learning rate **0.001**, batch size **32**, zero loader workers and CUDA. Linear used Adam, learning rate **0.01**, batch size **32**, zero loader workers and CUDA. These were the existing script defaults.

Seed 35 had the highest validation score in this batch. Its public-test score is a local development measurement; a Codabench score for this package is still pending. Public-test and leaderboard scores should not become a repeated seed-selection or tuning target.

## What changed in the code

The project already contained the model definitions and the training/data/export workflow before this GPU batch:

| Existing code | Role |
|---|---|
| [submission.py](../submission.py) | Linear model, Braindecode EEGNet wrapper and inference-only Solver. |
| [track2/workflow.py](../track2/workflow.py) | Training, validation checkpoint selection and official streaming scoring. |
| [track2/data.py](../track2/data.py) | Official preparation/loading and participant-split audit. |
| [track2/packaging.py](../track2/packaging.py) | Flat ZIP export and isolated inference validation. |
| [scripts/train_jarvis.sh](../scripts/train_jarvis.sh) | Prepare → train → saved-checkpoint test replay → package. |

For this batch, we preserved those core model/training files, their hyperparameters and the pinned official helpers. Training generated new learned weights and evidence.

The additions were operational helpers for clean snapshots, checksums, environment setup, the time-limit watchdog, GPU sampling, validation replay and result retrieval. We repaired a Windows CRLF issue in the managed-job launcher and used an in-memory patch to the installed CLI's text-transfer helper for later launches. The installed SDK files were not edited. Shell scripts in staging were normalized to Linux LF endings.

We updated [README.md](../README.md) and [STATUS.md](STATUS.md), created the detailed [run log](JARVISLABS_RUN_LOG.md), and saved machine-readable handoff/results. This guide is another documentation addition. No repository commit or push was performed.

## How to submit seed 35 to Codabench

Instructions checked against the organizers' guide and rules on 21 October 2026. The live account UI could not be reliably inspected, so confirm its active phase and remaining quota when uploading.

1. Sign in to [Track 2 on Codabench](https://www.codabench.org/competitions/17982/) and complete any registration prompts.
2. Open **My Submissions** and select the active phase. The published schedule places 21 October in **warm-up**.
3. Upload the **seed 35 ZIP linked above**, preserving its root layout.
4. Wait for **Finished**. Save the submission ID, phase, UTC time, status and score alongside the ZIP checksum.
5. If it fails, inspect the first ingestion error before trying again.

Published limits are **five submissions per participant per day during warm-up**, and **one per day during the sealed final** (28 October–21 November 2026, AoE). Failed uploads also consume the daily quota. Confirm the current schedule/quota shown in your account.

If the form offers a description, this is a suitable factual summary:

> EEGNet supervised PROTEUS baseline. Fixed official participant split seed 33; training initialization seed 35. Adam, learning rate 0.001, batch size 32, 20 epochs; validation selected epoch 17. Saved-checkpoint validation/test replay matched exactly. No inference adaptation.

The guide requires self-contained Python inference code and its weights/config at the ZIP root. Our package has that layout. Worker requirements list torch and Braindecode; deployed-version compatibility and the platform score remain to be established by evaluation.

## Checks still incomplete and retained evidence

- The trained ZIPs have **not been submitted to Codabench**. The earlier constant warm-up result of 0.33 was user-reported and is a separate artifact.
- Remote inference ZIP validation passed. Local structure, member-byte and checksum checks passed. **Local inference revalidation remains incomplete** because the installed local interpreters lack the required dependencies.
- The recorded environment has torch 2.12.1 and inherited torchaudio 2.11.0+cu130. Requested EEG training/replay checks passed; a general dependency check and unused audio functionality were not tested.
- One early process diagnostic exposed the notebook token in tool output. Its value was excluded from the source snapshot, project logs, handoff and downloaded artifacts. The detailed log records the incident without the credential.

Evidence links:

- [Detailed execution log](JARVISLABS_RUN_LOG.md)
- [Machine-readable handoff](../runs/jarvis/20261021T162400Z/handoff.json)
- [Verified model results](../runs/jarvis/20261021T162400Z/verified-results.json)
- [Seed 35 run verification](../runs/jarvis/20261021T162400Z/remote/runs/eegnet-seed35-20261021T164141Z/batch_verification.json)
- [Exact environment package listing](../runs/jarvis/20261021T162400Z/remote/evidence/pip-freeze.txt)
- [Official source revision](../UPSTREAM.json)

Official references:

- [Submission guide](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md)
- [Competition rules and schedule](https://neural-interfaces26.github.io/rules.html)
- [Track 2 scientific FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md)
- [Worker requirements](https://github.com/neural-interfaces26/2026-competition/blob/main/requirements.txt)
- [Warm-up configuration](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/bci_decoding/config.yaml)
