# Running on JarvisLabs

The code is portable Python plus two Linux shell scripts. It requires **Python 3.15** for the real EEG stack. Keep the project and data under the instance's persistent home directory (normally `/home/` for containers), not `/tmp` or another ephemeral location.

## Manual upload route

1. Choose a Python 3.15 PyTorch GPU template and verify its Python/CUDA torch combination.
2. Upload the complete source folder, including `vendor/competition/`. No submodule download is needed.
3. In the cluster terminal, change into that folder and run:

```bash
python3 --version
nvidia-smi
bash scripts/setup_jarvis.sh
source .venv/bin/activate
python -m pytest tests -q
```

The setup script inherits the template's packages through `--system-site-packages`; it does not intentionally replace a satisfactory CUDA torch installation. Requirements are not a command to switch your CUDA build. Select/install Python 3.15 first; a 3.15 environment cannot inherit Torch wheels built for an older Python. The full data stack currently uses the two pinned development wheels in requirements-python315.txt; verify installation and smoke tests before real training. Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) for a matching GPU build rather than a CPU-only wheel.

Then run:

```bash
bash scripts/train_jarvis.sh linear 33
bash scripts/train_jarvis.sh eegnet 33
```

Each command prepares public data, trains, replays and packages the trained model. Failures stop the sequence; a failed or partial run is not silently packaged. Start one seed/model first. Run seeds 34 and 35 only after a successful first run. The script requires CUDA; use the portable `run.py ... --device cpu` commands for CPU debugging.

## Managed `jl` route on an existing instance

Authenticate locally with `jl status --json` before using account operations. Keep API keys out of this repository. Directory uploads need local `rsync`; manual upload is an alternative on Windows.

```bash
jl run . --script run.py --on MACHINE_ID --requirements requirements-training.txt --json --yes -- smoke --include-eegnet
```

Use a Python 3.15 instance for that environment. Directory targets get a persistent project venv. After smoke checks succeed, data preparation is explicit:

```bash
jl run . --script run.py --on MACHINE_ID --requirements requirements-training.txt --json --yes -- prepare
jl run . --script run.py --on MACHINE_ID --requirements requirements-training.txt --json --yes -- train --model linear --device cuda --output runs/linear-seed33
```

Wait for each run to succeed before starting the next dependent run. The example output folder must not already exist. The same directory target name is reused so its data/cache/environment persist.

```bash
jl run status RUN_ID --json
jl run logs RUN_ID --tail 50
```

Run records are local to the computer that launched the job. Diagnose a failed run with its logs/status rather than launching duplicates. Use the returned machine ID after a resume; it can change. Do not use `--follow` for agent-managed monitoring.

## Persistence and handoff

- Preserve the entire selected `runs/...` directory, its `completed.json`, and its packaged ZIP.
- Download results before destroying an instance. Pausing preserves home storage but still incurs storage charges; destroying deletes it.
- Existing-instance `--on` jobs do not automatically pause the machine. Monitor them and manage lifecycle explicitly.
- No script here creates, destroys, pauses or uploads to a cloud account. You control those operations.
- Avoid uploading `.venv`, `.cache`, large old runs or `.env` when copying source. Git ignores them, but a manual directory upload can still include them.

## Useful references

- [JarvisLabs documentation](https://docs.jarvislabs.ai/)
- [IN1 migration information](https://docs.jarvislabs.ai/in1-migration)
- `jl run --help`, `jl run start --help`, `jl run logs --help`, `jl upload --help`

The workflow was prepared with the local JarvisLabs skill/CLI guide. Local account authentication was not present during initial implementation; no remote run is implied by a successful local smoke test.
