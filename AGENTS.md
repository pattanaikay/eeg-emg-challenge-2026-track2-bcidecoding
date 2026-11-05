# Project instructions

## Current checkpoint

- The user approved proceeding with parallel implementation on 5 November 2026 ("Let us spawn agents in parallel"). Use the discussed architecture; defer quizzes/study until the user asks to resume them.
- Read-only inspection, instruction/journal documentation, and the requested Python setup verification and existing smoke tests are authorized separately. Do not treat discussion, quiz answers, or Python installation as architecture approval.
- Build all six Tier 1 experiment paths once approved; the recommended execution order does not remove E4-E6 or their requested variants from scope.

## Help the user keep understanding

- The user is learning EEG methods and this codebase while reading the papers. Explain one concept at a time in plain language; define new terms before using abbreviations.
- At meaningful milestones, give a short explanation: the question being tested, what changed, where it lives, what the check establishes, and what remains unknown.
- The user deferred study/quizzes on 5 November; resume them only when requested. Then offer two or three brief diagnostic questions at conceptual checkpoints, invite "unsure" answers and revisit blind spots without treating the quiz as a gate or a grade.
- Do not use a quiz answer as permission to launch work. Do not require approval for every routine change once the architecture is approved; honor any later request to slow down or pause.
- Keep progress updates concise. Distinguish proposed, implemented, smoke-tested, real-data-tested, and benchmarked work. Synthetic success establishes execution contracts, not EEG performance.

## Read before coding

- Read `README.md`, `docs/research/NEXT_EXPERIMENTS.md`, `docs/STATUS.md`, `docs/COMPETITION_CONTRACT.md`, `docs/METHODOLOGY.md`, `docs/RESEARCH_JOURNAL.md`, and `runs/jarvis/20261021T162400Z/verified-results.json`.
- Inspect `git status` and the existing diff. Preserve uncommitted user work, existing checkpoints, and run artifacts. Do not reset, overwrite, or clean them.
- Apply the Ponytail skill: smallest complete change, reuse existing helpers, no speculative abstractions or unnecessary dependencies. Search callers before changing shared behavior and run focused checks.

## Code map and ownership

- `run.py`: CLI. `track2/data.py`: official preparation/loaders and split audit. `track2/workflow.py`: fitting, selection, official scoring and replay. `submission.py`: shared inference, models and Solver. `track2/packaging.py`: isolated flat-ZIP checks. `track2/smoke.py`, `tests/`: behavioral checks.
- Reuse the unchanged vendored official loader/scorer; do not fork the harness or edit `vendor/competition` for experiments.
- The approved layout is `experiments/e1_spectral`, `e2_covariance`, `e3_normalization`, `e4_csp`, `e5_spatial_power`, and `e6_sinc`, each with its own configuration, README, focused tests and experiment-specific code where needed. Shared contracts are in `experiments/INTERFACE.md`; execution instructions are in `docs/TIER1_EXECUTION.md`.
- The coordinator owns shared splits, runner, scoring, inference/export interfaces, dependencies, shared tests, experiment index and final integration.
- After approval, parallel agents own only assigned experiment folders, read the agreed interface first, and request shared-file changes through the coordinator. Check the actual concurrency limit; four total slots means one coordinator plus at most three workers, using two waves for six experiments.
- Agents report assumptions, changed files, checks actually run and limitations briefly; durable details belong in experiment READMEs, not separate lengthy reports.

## Scientific boundaries

- Verify current official sources before relying on rules. Recorded contract: one `(1, 41, 480)` EEG window at 120 Hz; classes `0` imagery, `1` subtraction, `2` word generation. Inference receives no labels or participant/run/interface identifiers.
- Reserve P10 sessions 2-3 before every new fitting or selection step, including scaling, spatial filters, covariance references and pretraining. P10 session 1 may remain in training. Evaluate sessions 2 and 3 independently from the starting model.
- Screening after reservation has 1,136 fitting windows, 379 validation windows, 755 previously viewed P04/P09 diagnostic windows, and 385 P10 later-session diagnostic windows. These counts are for the fixed screen, not every participant fold.
- Fit learned quantities only on the relevant training partition. Inner validation chooses hyperparameters/checkpoints. Outer evaluation measures the selected pipeline and must not feed model selection. Preserve participant/session/run boundaries and log fit membership.
- P10 later-session results measure known-person transfer only when that run trained on P10 session 1. P04/P09 results have already been viewed; neither they nor repeatedly inspected outer folds are pristine confirmation sets.
- Refit EEGNet and channel-mean linear controls on each matching allocation. Do not compare old 1,521-training-window scores as matched controls. External pretrained weights stay outside the primary comparison.
- Reuse official equal participant x session x interface cell-balanced accuracy; report class recalls and participant/session diagnostics. Windows and initialization seeds are not independent participants.
- Evaluation-time adaptation uses only current/earlier windows, without labels, and resets all state/updates at every participant-session. Document startup, current-window inclusion and reset policy before testing.
- E1: log band power, L2 logistic head, shrinkage-LDA ablation. E2: centered shrunk covariance, fixed training-reference tangent features, diagonal control. E3: separately selectable source, robust-window and causal normalization/alignment; no factorial sweep. E4: regularized three-class one-vs-rest CSP and selectable FBCSP, matched E1 bands/head. E5: small ShallowConvNet and selectable FBCNet. E6: matched compact decoder with Sinc/free temporal kernels; strict `0 < lower < upper < 60 Hz`, no GMM or pseudo-label memory.

## Runtime, export and verification

- The user requires Python 3.15 for this project. `.python-version` records that choice; `.venv` uses the installed Python 3.15.2. Use its explicit Python path for installs/tests/commands, not the shell's older global `python`. Do not silently fall back to 3.12. Check current 3.15 dependency/wheel compatibility and report actual failures.
- Start with existing relevant tests, then focused checks for input/channel/class contracts, finite outputs, split/fit separation, reload prediction parity, single-window inference, independent copies/reset, causal prefix independence, covariance/scaling/Sinc stability, official scoring and isolated export.
- All shipped Python inference code must be in one standalone `submission.py`; saved source, manifest and tensor checkpoint must agree. Do not let exported inference import experiment folders or training-only modules.
- Preserve legacy reload compatibility and existing no-overwrite protections. Separate immutable fitted quantities from transient session state; package the starting model, not the final state of an evaluated stream.
- Record actual instantiated parameter counts where applicable. Report runtime only when measured; label estimates. Never claim better decoding without matched real-data evaluations.
- READMEs must describe fitted quantities/state, controls, metrics/failure criteria, dependencies, execution status, and actual preparation/train/evaluation/reload/export commands. Do not document unimplemented flags or unrun checks as passed.

## Local and JarvisLabs execution

- New decoder implementation and its verification follow architecture approval. Existing environment/smoke-test verification may proceed after the user's Python installation without approving the new experiments. JarvisLabs is the intended training fallback when local execution is impractical.
- For later cloud execution, honor the user's authorized run scope; a fallback preference is not permission for arbitrary paid resources, expensive sweeps, or Codabench submissions. Implementation approval does not require learning quizzes to resume.
- Read the JarvisLabs skill before preparing its commands or using `jl`. Keep code/data/checkpoints in persistent storage, monitor authorized jobs, retrieve and verify results, and verify the agreed paused state when finished. Historical instance IDs/status are not current evidence.
- Keep durable updates concise in `docs/RESEARCH_JOURNAL.md`; update `docs/STATUS.md` when implementation or verification status changes. End handoffs with checks actually run, remaining limitations and exact next commands when applicable.
