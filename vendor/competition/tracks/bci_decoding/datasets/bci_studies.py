"""BCI-decoding studies through neuralbench's streamed Track 2 task.

Wraps the neuralbench ``eeg/_motor_imagery_stream`` task config (study,
subject-level split, 4-s cued windows, one-hot labels; the signal
notch-filtered at 50 and 60 Hz, band-pass filtered at 0.1-75 Hz, resampled
to 120 Hz and kept in microvolts, with no scaling or clamping) — see
``benchmark_utils.nb_task``. The ``study`` parameter picks the dataset
variant:

- ``dreyer2026proteus`` : Dreyer2026Proteus — the PROTEUS training release
                          (NEMAR nm000290 v1.0.0: 41 EEG channels, three
                          cued commands, one window from 1 s after each
                          cue), the task default and the warm-up evaluation
                          study; the default.
- ``tangermann2012``    : BNCI2014_001 — small, handy for real-data smoke
                          tests.

The test split is streamed one session of one participant at a time
(``data.stream_by: [subject, session]``), each window's ``info`` giving its
session (``stream_id``) and, on PROTEUS, its context, the interface of its
run (Graz or BrainHero), which the objective scores separately within each
session. On PROTEUS, a session's runs come in acquisition order, their BIDS
run numbers.

Requires a one-time download (``benchopt prepare``); the zero-dependency
``Simulated`` dataset covers no-network smoke testing.
"""

from benchopt import BaseDataset
from benchopt.config import get_data_path

# Hard requirements of the real-data path, imported at module level so
# benchopt reports the dataset as not-installed when they are missing.
import moabb  # noqa: F401
import neuralbench  # noqa: F401

from benchmark_utils.data import get_device
from benchmark_utils.nb_task import download_study, load_task, require_prepared

TASK = "_motor_imagery_stream"

# Overlay yaml in the task's ``datasets/`` folder (None = the task default).
_OVERLAYS = {
    "dreyer2026proteus": None,
    "tangermann2012": "tangermann2012",
}

# PROTEUS runs by BIDS task -> the interface they used, the context of the
# subject x session x context cells. Baseline runs hold no cue, so no window.
_PROTEUS_INTERFACES = {
    "AcquisitionBH": "BrainHero",
    "OnlineRawBH": "BrainHero",
    "AcquisitionGraz": "Graz",
    "OnlineRawGraz": "Graz",
}


def _proteus_interface(trigger):
    """Interface (Graz or BrainHero) of the run a PROTEUS window comes from."""
    task = trigger.get("task")
    if task not in _PROTEUS_INTERFACES:
        raise ValueError(f"PROTEUS run with unknown task {task!r}: its "
                         "interface, the window's context, is unknown.")
    return _PROTEUS_INTERFACES[task]


# Context of each window, for the studies that define one.
_CONTEXTS = {"dreyer2026proteus": _proteus_interface}


def _proteus_run(trigger):
    """Acquisition rank of a PROTEUS window's run within its session.

    PROTEUS numbers the runs of a session in recording order across its
    tasks (``run-1`` ... ``run-6``), so the BIDS run number is that rank.
    """
    return int(trigger["run"])


# Acquisition order of the recordings within a stream, for the studies
# that define one.
_ORDERS = {"dreyer2026proteus": _proteus_run}


class Dataset(BaseDataset):

    name = "BCI"

    requirements = [
        # neuroai release, pinned like requirements.txt.
        "pip::neuralset==0.4.0",
        "pip::neuralfetch==0.4.0",
        "pip::neuraltrain==0.4.0",
        "pip::neuralbench==0.4.0",
        # The PROTEUS release downloads from NEMAR through NeuralFetch;
        # BNCI2014_001 through MOABB.
        "pip::nemar-py>=0.3.1", "pip::moabb", "pip::mne", "scikit-learn",
    ]

    parameters = {
        "study": ["dreyer2026proteus"],
        "batch_size": [64],
        # Dataloader workers; 0 extracts windows in-process, which is what
        # shared CI/platform runners want. Raise it from the phase config
        # (``BCI[num_workers=4]``) on a worker with spare cores.
        "num_workers": [0],
        # "test" would restrict the study to its test split, which only a
        # PredefinedSplit supports (see benchmark_utils.nb_task): the stream
        # task draws its split at runtime, so the warm-up stages "full".
        "subset": ["full"],
    }

    test_parameters = {
        "study": ["tangermann2012"],
        "batch_size": [32],
    }

    # Ignore loader config for prepare cache key.
    prepare_cache_ignore = ("batch_size", "num_workers")

    def prepare(self):
        # Download the study, then run the pipeline once: the extraction
        # (filtering, segmenting, targets) caches next to the data, so runs
        # only touch warm caches. Both steps are idempotent.
        download_study(
            "eeg", TASK, self._data_dir(),
            dataset=_OVERLAYS[self.study],
        )
        self._load()

    def _data_dir(self):
        path = get_data_path("neural_compet")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self, device="cpu"):
        return load_task(
            "eeg", TASK,
            data_dir=self._data_dir(),
            dataset=_OVERLAYS[self.study],
            device=device,
            batch_size=self.batch_size,
            seed=self.get_seed(),
            num_workers=self.num_workers,
            subset=self.subset,
            # One-hot ``(K,)`` -> integer class label.
            target_transform=lambda y: y.argmax(-1),
            context_of=_CONTEXTS.get(self.study),
            order_of=_ORDERS.get(self.study),
        )

    def get_data(self):
        # Load already-prepared data only; downloading + extracting is the
        # explicit ``prepare`` step (``benchopt prepare`` / ``--prepare``).
        require_prepared("eeg", TASK, self._data_dir(),
                         dataset=_OVERLAYS[self.study])
        loaders, meta = self._load(device=get_device())
        return dict(
            # subset="test" stages the evaluation split only: no train
            # split to hand over, which is how a run learns it cannot train.
            train_loader=None if self.subset == "test" else loaders["train"],
            test_loader=loaders["test"],
            n_classes=int(meta.pop("raw_target_shape")[-1]),
            **{k: v for k, v in meta.items() if k != "target_shape"},
        )
