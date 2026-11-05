# Run the six Tier 1 experiments

This guide runs the implemented CLI. [Experiment index](../experiments/README.md) links exact models, ablations, controls, fitting/state boundaries and failure criteria. Training happens outside Codabench; `evaluate` reloads a saved checkpoint and performs inference through the official chronological single-window evaluator.

## 1. Use the prepared Python environment

The project requires **Python 3.15**. This Windows checkout uses your installed **Python 3.15.2** in `.venv`; the shell's default `python` still resolves to 3.10. Use the explicit project interpreter below. Neither the Windows `.venv` nor the ignored runtime cache should be copied to Linux.

```powershell
Set-Location 'C:\Projects\EEG-EMG-Foundation-Challenge-2026\eeg-emg-challenge-2026-track2-bcidecoding'
$py = Join-Path $PWD '.venv\Scripts\python.exe'
$data = '\\?\C:\Projects\EEG-EMG-Foundation-Challenge-2026\eeg-emg-challenge-2026-track2-bcidecoding\data'
& $py --version
& $py run.py doctor
& $py -m pytest tests experiments -q --basetemp=.cache/pytest-manual
& $py run.py smoke --include-eegnet --tier1
```

The local Torch installation is CPU-only. Use `--device cpu` locally; `--device cuda` requires a separately verified CUDA-enabled environment. `doctor` reports this rather than silently changing the requested device. The smoke command fits all 21 named experiment presets and two controls on synthetic windows, reloads validation/test predictions, and validates each isolated ZIP. It leaves its summary in `.cache/tier1-smoke-results.json`; temporary synthetic runs are removed. These are execution checks, not EEG performance measurements.

To recreate the environment, use the installed interpreter explicitly (the old Windows launcher does not currently discover it):

```powershell
& 'C:\Users\Pratik\AppData\Local\Programs\Python\Python315\python.exe' -m venv .venv
$env:PYYAML_FORCE_LIBYAML = '0'
.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

For the existing uv-created environment, use `$env:PYYAML_FORCE_LIBYAML = '0'` followed by `uv pip install --python .venv\Scripts\python.exe -r requirements-training.txt --cache-dir .cache/uv`. PyYAML's optional C extension is disabled; its supported Python implementation is sufficient. NumPy, SciPy, scikit-learn and pandas pins were updated to releases with Python 3.15 wheels. Stable h5py/numcodecs releases currently lack those wheels, so `requirements-python315.txt` pins their upstream scientific-nightly builds and scopes wheel discovery to those two packages. This is a documented compatibility limit; replace the development pins once stable 3.15 wheels pass the same checks.

## 2. Prepare once, then choose one preset

```powershell
& $py run.py prepare --data-dir $data
```

Preparation downloads only public data if needed and reuses the official preprocessing cache. It may report the original official 1,521-window training allocation; no decoder is fitted by preparation. The config-based training command repartitions before fitting and records the reserved allocation in `split.json` and `split_windows.csv`.

Choose a config from the index. For the first spectral run:

```powershell
$config = 'experiments/e1_spectral/configs/logistic.json'
$run = 'runs/tier1/e1-logistic-screen-s33'
& $py run.py train --config $config --protocol screen --data-dir $data --seed 33 --device cpu --output $run
```

Other primary configs and explicit variants:

| Path | Primary config | Comparison config in the same directory |
|---|---|---|
| E1 | `experiments/e1_spectral/configs/logistic.json` | `lda.json`, `low_logistic.json` |
| E2 | `experiments/e2_covariance/configs/tangent_logistic.json` | `diagonal_logistic.json`, `tangent_lda.json`, `diagonal_lda.json` |
| E3 | `experiments/e3_normalization/configs/eegnet_source.json` | `eegnet_none.json`, `eegnet_robust.json`, `eegnet_causal_zscore.json`, `eegnet_causal_ea.json` |
| E4 | `experiments/e4_csp/configs/csp.json` | `csp_control.json`, `fbcsp.json`, `fbcsp_control.json` |
| E5 | `experiments/e5_spatial_power/configs/shallowconvnet.json` | `fbcnet.json` |
| E6 | `experiments/e6_sinc/configs/sinc.json` | `free.json` |

Replace `$config` and use a new `$run` for each deliberate comparison. E1/E2/E4 are CPU statistical fits. E3/E5/E6 are small neural paths; use CPU for local execution or the verified JarvisLabs CUDA environment below. `--epochs`, `--lr`, `--batch-size` and `--seed` are actual training flags. Statistical heads use their fixed JSON configuration and do not optimize epochs. Neural defaults are 20 epochs/lr 0.001; channel-mean linear retains 30/lr 0.01. These defaults do not by themselves establish equal compute budgets; predeclare matching budgets for the comparison being claimed.

Training fits only the training partition, selects neural checkpoints by validation cell-balanced accuracy, then reports validation/test/P10 diagnostics. It does not use those test/P10 results to select settings. Freeze comparison choices before inspecting outer results. Predeclare a useful improvement margin and class/session harm criteria before selecting finalists; the code does not turn a favorable diagnostic score into a validated improvement claim.

Existing output folders are rejected. Training cannot resume by pointing at an old run. Use another output name for a retry and inspect the original failure rather than deleting evidence.

## 3. Refit matching controls

```powershell
& $py run.py train --config experiments/controls/linear.json --protocol screen --data-dir $data --seed 33 --device cpu --output runs/tier1/control-linear-screen-s33
& $py run.py train --config experiments/controls/eegnet.json --protocol screen --data-dir $data --seed 33 --device cpu --output runs/tier1/control-eegnet-screen-s33
```

Use the same protocol/fold and relevant preprocessing/checkpoint budget as the candidate. E3's `eegnet_none.json` is the fixed reference for its normalization comparisons; E4's controls match the same filtering, bands and head exactly. E6 sinc/free presets have identical source scaling and downstream topology but different parameter counts. Old scores from runs trained on 1,521 windows are not matching controls.

## 4. Saved-checkpoint inference and export

With `$run` still pointing to the selected real-data run:

```powershell
& $py run.py evaluate --run $run --data-dir $data --device cpu --split val
& $py run.py evaluate --run $run --data-dir $data --device cpu --split test
& $py run.py evaluate --run $run --data-dir $data --device cpu --split longitudinal
& $py run.py package --run $run --output artifacts/tier1-e1-logistic-screen-s33.zip
& $py run.py validate artifacts/tier1-e1-logistic-screen-s33.zip
```

Change the ZIP filename to match your actual run. `evaluate` reconstructs the saved protocol, source/config and tensors without fitting. It writes `replay/<split>_predictions.csv`, metrics, cells, confusion, participant/session and startup diagnostics. Local evaluation labels are used only by scoring; the model receives only EEG. For synthetic runs add `--data simulated` to `evaluate`; otherwise the command correctly rejects a data-kind mismatch. Synthetic data have no P10 longitudinal split.

Primary score: `balanced_accuracy_cell_mean`, equally weighted over participant x session x interface cells. Inspect `command_recalls`, worst-session score, session variation, `<split>_participants.csv`, `<split>_sessions.csv` and `<split>_startup.csv`. Startup rows report per-class errors/recall for first window, first ten windows and later windows; a missing class has a blank recall, not a forced three-class startup score. Use saved participant/cell rows for paired comparisons and average initialization seeds within participants; windows and seeds are not independent people.

`model.json`, `weights.pt` and the run's assembled `submission.py` reconstruct inference. Normalization/reference/projection/scaler state is included as fitted tensors; transient running-session state is excluded. Each participant-session starts from an independent copy with reset state. Packaging requires `completed.json`, snapshots the run's source rather than today's working source, and validates the flat ZIP in a fresh process that cannot import project experiment modules.

## 5. One fixed outer participant fold

```powershell
$foldRun = 'runs/tier1/e1-logistic-outer0-s33'
& $py run.py train --config experiments/e1_spectral/configs/logistic.json --protocol outer --fold 0 --data-dir $data --seed 33 --device cpu --output $foldRun
& $py run.py evaluate --run $foldRun --data-dir $data --device cpu --split test
& $py run.py evaluate --run $foldRun --data-dir $data --device cpu --split diagnostic
& $py run.py evaluate --run $foldRun --data-dir $data --device cpu --split longitudinal
```

| Fold | Outer test pair | Inner validation pair | P10 later sessions test a known training participant? |
|---|---|---|---|
| 0 | P01, P02 | P03, P05 | Yes |
| 1 | P03, P05 | P06, P07 | Yes |
| 2 | P06, P07 | P08, P10 | No |
| 3 | P08, P10 | P01, P02 | No |

The remaining four development participants are the fitting partition. P10 sessions 2-3 are always excluded, regardless of fold. `test` means the chosen outer pair; `diagnostic` means the already viewed P04/P09. In screen runs P04/P09 are `test`, so there is no separate `diagnostic` loader. `split.json` records actual membership/counts and `longitudinal_known_person`. Refit both controls for each fold being compared. Four folds reuse this cohort and do not create four independent new datasets.

## 6. JarvisLabs when local neural training is impractical

Read the [upload/environment guide](JARVISLABS.md). Use an existing user-authorized running instance with Python 3.15 and matching CUDA Torch. Upload current source, including the uncommitted `experiments/` and shared changes and `vendor/competition/`; exclude Windows `.venv`, `.cache`, `.env` and unrelated old artifacts. Public data can be prepared remotely instead of transferred.

In its terminal, from the persistent project directory:

```bash
python3 -c 'import sys; assert sys.version_info[:2] == (3, 15), sys.version'
nvidia-smi
bash scripts/setup_jarvis.sh
source .venv/bin/activate
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'
python -m pytest tests experiments -q --basetemp=.cache/pytest-jarvis
python run.py smoke --tier1
python run.py prepare --data-dir data
python run.py train --config experiments/e5_spatial_power/configs/shallowconvnet.json --data simulated --epochs 1 --device cuda --output runs/e5-cuda-synthetic
python run.py evaluate --run runs/e5-cuda-synthetic --data simulated --device cuda --split test
python run.py train --config experiments/e5_spatial_power/configs/shallowconvnet.json --protocol screen --data-dir data --seed 33 --device cuda --output runs/tier1/e5-shallow-screen-s33
python run.py evaluate --run runs/tier1/e5-shallow-screen-s33 --data-dir data --device cuda --split test
python run.py evaluate --run runs/tier1/e5-shallow-screen-s33 --data-dir data --device cuda --split longitudinal
python run.py package --run runs/tier1/e5-shallow-screen-s33 --output artifacts/tier1-e5-shallow-screen-s33.zip
```

Wait for each dependent command to succeed. For managed jobs, authenticate with `jl status --json`, then use the existing directory-upload route (requires local rsync):

```bash
jl run . --script run.py --on MACHINE_ID --requirements requirements-training.txt --json --yes -- train --config experiments/e5_spatial_power/configs/shallowconvnet.json --protocol screen --data-dir data --seed 33 --device cuda --output runs/tier1/e5-shallow-screen-s33
jl run status RUN_ID --json
jl run logs RUN_ID --tail 50
```

`MACHINE_ID` and `RUN_ID` are placeholders for actual returned identifiers. The managed directory environment must already pass the same Python/CUDA checks; do not launch a duplicate of the terminal job. Managed `--on` jobs do not pause the machine afterward. After successful replay/packaging, retrieve the entire run, ZIP and hashes, then verify the agreed paused state:

```bash
jl download MACHINE_ID /home/eeg-emg-challenge-2026-track2-bcidecoding/runs/tier1/e5-shallow-screen-s33 runs/jarvis/e5-shallow-screen-s33 -r
jl download MACHINE_ID /home/eeg-emg-challenge-2026-track2-bcidecoding/artifacts/tier1-e5-shallow-screen-s33.zip artifacts/tier1-e5-shallow-screen-s33.zip
jl pause MACHINE_ID --yes --json
jl get MACHINE_ID --json
```

Adjust the persistent project path if the uploaded directory has another name. Obtain/check SHA-256 on both sides before considering retrieval complete. Commands here do not imply a cloud job was launched, a current instance exists, or a submission is authorized. No Codabench submission command is included.

## Verification and limits

Final local evidence is recorded in [STATUS.md](STATUS.md) and `runs/tier1-local-verification-20261105-py315/`: environment, pytest report, real protocol/input checks and synthetic preset results. The shared OAS shrinkage helper now matches scikit-learn's independent estimator with the documented unbiased scaling and positive floor. E2's diagonal control discards off-diagonal entries before estimating shrinkage. Older saved sources/checkpoints remain their historical implementations.

Real-data fitting, matched decoding scores, GPU execution and worker-image replay have not been run for these new implementations. Local export validation used the recorded Python/Braindecode environment; verify the actual worker stack/image before any later submission. Two import-time MNE montage deprecation warnings do not fail the current checks. Nothing in the synthetic results establishes improved EEG decoding.
