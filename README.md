# Track 2: EEG cross-session and cross-participant decoding

This repository starts the EEG/EMG Foundation Challenge 2026 Track 2 project with a working submission contract, public PROTEUS data tools, supervised baselines, and reproducible evaluation. The immediate deliverable is a validated ZIP for the warm-up. The scientific goal is reliable transfer with less labeled data and, later, useful prediction of harmful drift.

**Current status:** you reported a successful warm-up submission with score **0.33** on 20 October 2026. The constant reference checks the platform; it learns no EEG decoding. A real-data linear baseline has also completed training, identical inference replay and export: `artifacts/linear-seed33.zip`, with **0.3797** held-out public cell-averaged balanced accuracy. Its local development score is separate from the Codabench result.

**Next:** follow section 3 to run EEGNet on JarvisLabs. Section 1 preserves the steps for reproducing the constant warm-up check. A failed upload consumes a submission, so validate locally before each upload.

## 1. First warm-up submission

Use Python **3.12 or newer** for the complete project. The four official neural packages require it. A dependency-light constant/linear smoke test also works on Python 3.11.

```bash
python -m venv --system-site-packages .venv
# Linux / JarvisLabs:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py warmup
```

If the ZIP has already been generated, use `python run.py validate artifacts/warmup_constant.zip`. To create another copy, pass a new `--output` filename; existing artifacts are not overwritten.

Upload the **ZIP itself**, not the repository, to [Track 2 → My Submissions](https://www.codabench.org/competitions/17982/). Select the active warm-up phase. Wait for `Finished`, then record the status and score. For a failure, the first ingestion error is usually the useful one. Files are at the ZIP root: `submission.py` and `model.json`; trained runs also contain `weights.pt`.

The official kit explicitly provides constant baselines for this first platform check. No weights or training are needed for that reference. Do not present it as a useful mental-command decoder.

## 2. Understand the current contract

Read [the contract](docs/COMPETITION_CONTRACT.md) before changing preprocessing. The earlier 47-signal/calibration description is obsolete for model-facing evaluation.

- Input: one **41-channel EEG** window, `(1, 41, 480)`, at 120 Hz in microvolts, spanning 1–5 seconds after the cue.
- Output: one integer label: `0` motor imagery, `1` mental subtraction, `2` word generation.
- `predict(X)` sees no participant/session/run/interface identifiers or labels.
- Windows arrive chronologically within a participant-session stream. Every session starts from a fresh model copy.
- Current-session label-free adaptation is allowed by the scientific FAQ; these initial baselines remain fixed at inference.
- Rank with **participant × session × interface cell-averaged balanced accuracy**. Pooled accuracy and a session-only average are different diagnostics.

The public release has 10 participants, 14 sessions and 112 runs. Only two participants have three sessions; the other eight have one. Sealed evaluation combines later sessions and unseen participants. There is no extra labeled calibration supplied for sealed participants. See [official FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md).

## 3. Upload the project to JarvisLabs

Follow [the JarvisLabs guide](docs/JARVISLABS.md). Use a Python 3.12+ PyTorch GPU environment, put the project under persistent `/home/`, and run:

```bash
bash scripts/setup_jarvis.sh
source .venv/bin/activate
python run.py prepare --data-dir data
python run.py audit --data-dir data
python run.py train --model linear --device cuda --output runs/linear-seed33
python run.py evaluate --run runs/linear-seed33 --device cuda
python run.py package --run runs/linear-seed33 --output artifacts/linear-seed33.zip
```

The first preparation downloads the public release (about 1.5 GB) and caches official preprocessing. No private evaluation data are downloaded. Keep disk space for the extracted data and caches too. Once the linear path works:

The workflow generates project-local NeuralBench configuration under `.cache/`, preventing a first-run path prompt in detached jobs. An explicitly supplied, existing `NEURALBENCH_CONFIG` is respected. Your home configuration is not edited.

On Windows, this long repository location exceeded the normal path limit inside an upstream cache. Preparation passed using the extended-length data path below. Pass the same `--data-dir` to `prepare`, `train` and `evaluate` when working at this location; JarvisLabs uses the ordinary `data` path.

```powershell
$eegData = '\\?\C:\Projects\EEG-EMG-Foundation-Challenge-2026\eeg-emg-challenge-2026-track2-bcidecoding\data'
python run.py prepare --data-dir $eegData
```

```bash
python run.py train --model eegnet --device cuda --seed 33 --output runs/eegnet-seed33
python run.py evaluate --run runs/eegnet-seed33 --device cuda
python run.py package --run runs/eegnet-seed33 --output artifacts/eegnet-seed33.zip
```

Run seeds 34 and 35 in separate output folders after a successful first run. `scripts/train_jarvis.sh linear 33` or `scripts/train_jarvis.sh eegnet 33` performs prepare → train → replay → package in one fail-fast shell script. It creates a timestamped run folder. It assumes `.venv` was prepared by the setup script.

Training defaults are transparent implementation choices: Torch-Linear uses 30 epochs/Adam learning rate 0.01; EEGNet uses 20 epochs/Adam learning rate 0.001. These mirror the worked solver defaults, while this project selects the checkpoint using validation cell-averaged balanced accuracy. They are not claimed to be optimal challenge settings. Override `--epochs`, `--lr`, `--batch-size`, `--seed` or `--num-workers` deliberately; every run records them.

## 4. Run the checks before paying for training

```bash
python run.py doctor
python -m pytest tests -q
python scripts/smoke_test.py
python scripts/smoke_test.py --include-eegnet
```

The full smoke script uses synthetic examples at the real input dimensions, optimizes each selected architecture, selects a validation checkpoint, reloads it, compares replay scores, and validates the exported flat ZIP in another process. It also tests official metric semantics, shared session state, fresh-stream reset and out-of-order rejection. **It does not establish EEG performance.** The `--include-eegnet` flag is required to claim the EEGNet path was checked.

`python run.py --help` lists the portable commands. No command provisions a GPU instance, uploads to Codabench, contacts JarvisLabs, or modifies your account. The upload remains a separate action.

## 5. What is saved?

Each training run has a new folder; an existing folder is rejected rather than silently warm-started or overwritten.

| Artifact | Meaning |
|---|---|
| `training.json`, `environment.json`, `UPSTREAM.json` | Hyperparameters, installed versions, hardware and exact official-kit revision |
| `model.json`, `submission.py`, `weights.pt` | Matching inference manifest, inference source and best validation weights |
| `history.csv`, `selection.json` | Optimization trace and reason for the selected checkpoint |
| `split_windows.csv` on real data | Public participant/session/run membership for checking disjointness |
| `val_*`, `test_*` | Metrics, predictions, cell scores and confusion matrix |
| `completed.json` | Written only after training and both evaluations finish |
| `replay/` | Inference-only replay of the saved checkpoint |

`runs/data-audit/` records the BIDS inventory. Dataset folders, caches, runs and submission ZIPs are ignored by Git. Back up useful runs separately; do not expect `git push` to preserve them. Uploading a source checkout to a new cluster does not include ignored local datasets or checkpoints unless you explicitly transfer them.

## 6. Code map

| Module | Responsibility |
|---|---|
| `run.py` | One command-line entry point for local and cluster use |
| `submission.py` | Every Python inference component that can enter a ZIP |
| `track2/data.py` | Pinned official preparation/loaders, split audit and BIDS inventory |
| `track2/workflow.py` | Training, validation selection, official streaming evaluation and saved evidence |
| `track2/packaging.py` | Flat ZIP construction and exported-module validation |
| `track2/smoke.py`, `tests/` | Behavioral checks rather than only import/shape checks |
| `vendor/competition/` | Unmodified selected upstream sources; revision in `UPSTREAM.json` |

The official loader/scorer are reused. We do not randomly split EEG windows or substitute scikit-learn's subtly different missing-class balanced accuracy. The BIDS metadata are available for **development analysis**, not supplied to a submitted model's `predict` method.

## 7. Learn and extend the research

- [Methodology and interpretation](docs/METHODOLOGY.md): what the baselines establish, what longitudinal transfer means, and how calibration/forgetting/monitoring fit the current rules.
- [Paper and documentation reading route](docs/READING.md): primary sources, what to read in each, and the question it should answer.
- [Current verification status](docs/STATUS.md): completed checks, measured results and remaining limits.
- [Third-party source notes](NOTICE.md): origin and scope of the vendored kit.

After a real baseline, inspect errors by participant/session/interface and command. Then compare a causal normalization change or a covariance-based reference at the same splits and label budget. Establish whether the change helps before adding a foundation model or a combined adaptation stack. This repository implements the first baseline milestone; it does not claim a novel drift algorithm or clinical validation.
