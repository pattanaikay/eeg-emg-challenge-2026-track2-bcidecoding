"""Data loading through neuralbench task configs (the official pipelines).

Built on :func:`neuralbench.experiment_config.merge_task_config` for config
composition (defaults <- task <- dataset overlay) and
:class:`neuralbench.data.Data` for the loaders::

    loaders, meta = load_task(
        "eeg", "motor_imagery", data_dir=..., dataset="tangermann2012",
    )

Differences with running neuralbench itself:

- everything runs locally with an explicit ``data_dir`` (the study path and
  the exca infra are overridden — the ``~/.neuralbench`` config resolves in
  the background but none of its paths are used);
- the returned loaders follow the competition contract (``(X, y, info)``
  torch batches moved onto ``device``, see ``benchmark_utils.data``), so
  nothing downstream is tied to neuralset/neuralbench types;
- ``subset="test"`` restricts the pipeline to the test split (see
  :func:`build_test_only_filter`) so workers can stage evaluation data only;
- the test loader keeps neuralbench's order and batch size (recordings in
  order, each in time order; one window per batch for the stream tasks),
  and each window's ``info`` tells its recording, its start in it, its
  stream (``data.stream_by``) and, when the caller defines one, its context,
  which streamed evaluation and its scores need (see
  ``benchmark_utils.streaming``). With ``order_of``, the validation and test
  loaders put each stream's recordings in acquisition order.

Like the rest of the neuro stack, this module is import-heavy; import it
only from ``datasets/`` modules (never from solvers).
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from benchmark_utils.data import to_numpy


class _LogOnce(logging.Filter):
    """Keep the first record of each message, drop the repeats."""

    def __init__(self):
        super().__init__()
        self._seen = set()

    def filter(self, record):
        key = (record.name, record.levelno, record.getMessage())
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


# Shared instance: ``addFilter`` is then idempotent across calls.
_LOG_ONCE = _LogOnce()


def _quiet_neuro_logs():
    """Hide neuro stack's logs out of the participant output.

    Call after importing the stack: both libs set their logger level on
    import, which would undo this.
    """
    for name in ("neuralset", "exca"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(sys.stdout)

    # The neuro extractor logs the same WARNING for every recording (the
    # task config's lowpass sits above the Nyquist of its resampling, so it
    # is dropped): worth reading once, not once per file.
    logging.getLogger("neuralset.extractors.neuro").addFilter(_LOG_ONCE)

    # Reading a recording raises a RuntimeWarning per EDF filter quirk.
    # They are matched as coming from "mne" even though they are reported
    # at the neuralfetch study that opened the file: mne.utils.warn calls
    # warn_explicit with its own module and the caller's filename.
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


def _task_data_config(modality, task, dataset, data_dir, overrides):
    """The task's merged ``data`` config, localized: everything runs from
    ``data_dir`` with no exca cluster, caller overrides applied last."""
    from neuralbench.experiment_config import merge_task_config

    data_dir = Path(data_dir)
    cfg = merge_task_config(modality, task, dataset)["data"]
    cfg.update({
        "study.source.path": str(data_dir),
        # keep every cache next to the data (the studies' timeline loaders
        # need a folder for their exca Cached backend).
        "study.source.infra.folder": str(data_dir / "cache"),
        "neuro.infra.cluster": None,
        "neuro.infra.folder": str(data_dir / "cache"),
        # exca chmods every cache entry it reads to 0o777 for cluster
        # sharing, which warns on each one when the data is mounted
        # read-only; None disables the chmod. Only exca infras take it —
        # the study source is a neuralset model that rejects the key.
        "neuro.infra.permissions": None,
    })
    # Force serial extraction when set (NEURALBENCH_EXTRACT_CPUS=1): the neuro
    # and target exca extractors (cpus_per_task=10, cluster=auto) hammer the
    # study's TimelineLoader Cached backend concurrently and corrupt its jsonl
    # on big studies (the p4d has 96 cores). Cap those two to serialize; the
    # study.source infra is a Cached backend and rejects these keys.
    extract_cpus = os.environ.get("NEURALBENCH_EXTRACT_CPUS")
    if extract_cpus:
        n = int(extract_cpus)
        cfg.update({
            "neuro.infra.cluster": None,
            "neuro.infra.cpus_per_task": n,
            "target.infra.cluster": None,
            "target.infra.cpus_per_task": n,
        })
    cfg.update(overrides or {})
    return cfg


def build_test_only_filter(data_cfg):
    """``filter_stimuli`` override restricting a study to its test split.

    Composes a :class:`neuralset.events.transforms.QueryEvents` query that
    keeps non-trigger events untouched and selects trigger events with the
    split's ``test_split_query`` when the config defines one, or on the
    ``split`` column otherwise (predefined splits shipped with the source
    data). Any ``filter_stimuli`` already present in the config is preserved
    (queries are and-composed).

    Only expressible for ``PredefinedSplit`` — runtime splits (e.g.
    ``SklearnSplit`` by subject hash) are not a stimuli query; asking for
    ``subset="test"`` on such a study raises.
    """
    split = dict(data_cfg.get("study", {}).get("split") or {})
    if split.get("name") != "PredefinedSplit":
        raise ValueError(
            "subset='test' requires a PredefinedSplit (an explicit "
            f"test_split_query or split column); got {split.get('name')!r}."
        )
    selector = split.get("test_split_query")
    if not selector:
        selector = f"{split.get('col_name', 'split')} == 'test'"

    trigger = data_cfg["trigger_event_type"]
    triggers = [trigger] if isinstance(trigger, str) else list(trigger)
    query = f"type not in {triggers!r} or ({selector})"

    existing = dict(data_cfg.get("study", {}).get("filter_stimuli") or {})
    if existing.get("query"):
        query = f"({existing['query']}) and ({query})"
    return {"name": "QueryEvents", "query": query}


def _channel_names(seg_ds):
    """Ordered channel names of a prepared ``SegmentDataset``'s extractor.

    The extracted signal is a bare tensor, but the neuro extractor keeps a
    name -> column-index map assigned during ``prepare()``. There is no
    public accessor (neuralbench's ``Data.prepare`` reads the same private
    attribute); returns ``None`` when unavailable so datasets degrade
    gracefully (``chs_info`` stays ``None``).
    """
    chans = getattr(seg_ds.extractors["neuro"], "_channels", None)
    if not chans:
        return None
    return [name for name, _ in sorted(chans.items(), key=lambda kv: kv[1])]


def download_study(modality, task, data_dir, dataset=None):
    """One-time download of a task's study data (``Dataset.prepare``).

    Skipped when ``data_dir`` is not writable: on the competition workers
    the staged data is mounted read-only, and even a fully-cached
    ``Study.download()`` ends with a ``chmod`` that would crash there.
    """
    if not os.access(data_dir, os.W_OK):
        return

    import neuralset as ns
    _quiet_neuro_logs()

    cfg = _task_data_config(modality, task, dataset, data_dir, None)
    study = dict(cfg["study"]["source"])
    study["path"] = Path(study["path"]) / study["name"]
    ns.Study(**study).download()


def require_prepared(modality, task, data_dir, dataset=None):
    """Fail with a clear message when the study data is not prepared.

    ``get_data`` only loads already-prepared data — downloading + extracting
    is the explicit ``prepare`` step (``benchopt prepare`` / ``--prepare``).
    On a read-only staged mount the data is present by construction, so this
    is a no-op; on a writable checkout it fails fast pointing at ``prepare``
    rather than silently downloading a large study mid-run.
    """
    if not os.access(data_dir, os.W_OK):
        return  # read-only staged bundle: prepared by construction

    cfg = _task_data_config(modality, task, dataset, data_dir, None)
    study = dict(cfg["study"]["source"])
    study_path = Path(study["path"]) / study["name"]
    if not study_path.exists():
        raise RuntimeError(
            f"{modality}/{task} data is not prepared under {data_dir}. "
            "Run `benchopt prepare` first."
        )


class _NBWindows(torch.utils.data.Dataset):
    """Adapt a prepared neuralset ``SegmentDataset`` split to ``(X, y, info)``.

    ``seg_ds[i]`` yields a neuralset ``Batch`` whose ``data`` holds the
    ``neuro`` signal, the ``target``, the ``subject_id`` and, for a task
    that sets ``data.stream_by``, the ``stream_id``; this wrapper converts
    them to the competition contract (plain torch tensors). ``info`` gives
    each window's ``subject_id``, ``record_id`` (its recording),
    ``onset`` (its start in that recording, in seconds) and ``stream_id``
    (the task's stream, else its recording), so the scores can group the
    windows by stream. With ``context_of``, it also gives each window's
    ``context_id``: ``context_of`` maps the fields of a window's trigger
    event (``subject``, ``session``, ``task``, ``run``, ...) to its context
    label. With ``order_of``, the windows come with each stream's recordings
    in acquisition order (see :func:`_acquisition_order`); ``stream_by``
    names the trigger fields that make up a stream (``data.stream_by``).
    """

    def __init__(self, seg_ds, target_transform=None, context_of=None,
                 order_of=None, stream_by=None):
        self.seg_ds = seg_ds
        self.target_transform = target_transform
        # Recording and start of every window, from the segments alone (no
        # signal is read). Recordings and contexts are numbered in order of
        # appearance.
        records = {}
        self.record_id = np.array(
            [records.setdefault(seg.timeline, len(records))
             for seg in seg_ds.segments], dtype=np.int64)
        self.onset = np.array([seg.start for seg in seg_ds.segments],
                              dtype=np.float64)
        self.context_id = None
        if context_of is not None:
            contexts = {}
            self.context_id = np.array(
                [contexts.setdefault(context_of(seg.trigger.to_dict()),
                                     len(contexts))
                 for seg in seg_ds.segments], dtype=np.int64)
        # Position i of this dataset reads the segment self.index[i].
        self.index = np.arange(len(seg_ds.segments), dtype=np.int64)
        if order_of is not None:
            self.index = _acquisition_order(
                [seg.trigger.to_dict() for seg in seg_ds.segments],
                self.record_id, self.onset, stream_by, order_of)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        i = int(self.index[i])
        data = self.seg_ds[i].data
        X = torch.as_tensor(to_numpy(data["neuro"]), dtype=torch.float32)
        if X.ndim == 3:                 # (1, C, T) -> (C, T)
            X = X[0]
        y = torch.as_tensor(to_numpy(data["target"]))
        if y.ndim and y.shape[0] == 1:  # (1, ...) -> (...)
            y = y[0]
        if self.target_transform is not None:
            y = self.target_transform(y)
        y = y.to(torch.float32 if y.is_floating_point() else torch.long)
        subject = data.get("subject_id")
        stream = data.get("stream_id")
        info = {
            "subject_id": _first(subject) if subject is not None else -1,
            "record_id": int(self.record_id[i]),
            "onset": float(self.onset[i]),
            "stream_id": (_first(stream) if stream is not None
                          else int(self.record_id[i])),
        }
        if self.context_id is not None:
            info["context_id"] = int(self.context_id[i])
        return X, y, info


def _first(value):
    """The single integer of an extracted ``(1,)``-shaped label."""
    return int(np.asarray(to_numpy(value)).reshape(-1)[0])


def _acquisition_order(triggers, record_id, onset, stream_by, order_of):
    """Window order putting each stream's recordings in acquisition order.

    Parameters
    ----------
    triggers : list of dict
        The fields of each window's trigger event.
    record_id, onset : arrays, shape ``(N,)``
        Each window's recording and its start in that recording.
    stream_by : list of str or None
        Trigger fields whose values make up a window's stream
        (``data.stream_by``); None makes each recording its own stream.
    order_of : callable
        Maps a window's trigger fields to the acquisition rank of its
        recording within its stream (e.g. the BIDS run number).

    Returns
    -------
    index : array, shape ``(N,)``
        The windows' indices in the new order: streams in order of first
        appearance, each stream's recordings by ``order_of`` (ties in order
        of first appearance), each recording's windows by start.
    """
    streams, first_window = {}, {}
    keys = []
    for i, trigger in enumerate(triggers):
        stream = (tuple(str(trigger[field]) for field in stream_by)
                  if stream_by else (int(record_id[i]),))
        keys.append((streams.setdefault(stream, len(streams)),
                     order_of(trigger),
                     first_window.setdefault(int(record_id[i]), i),
                     float(onset[i]), i))
    return np.array([key[-1] for key in sorted(keys)], dtype=np.int64)


def _window_sfreq(frequency, n_times, duration):
    """Sampling rate of a task's windows, in Hz.

    The neuro extractor's ``frequency``, or, when the task keeps each
    recording's own rate (``frequency: native``, e.g. Track 3's Muse data),
    the rate read off a window: its ``n_times`` samples over its
    ``duration`` in seconds.
    """
    if frequency != "native":
        return float(frequency)
    if not duration:
        raise ValueError(
            "a task at its native rate needs a window duration to give its "
            "sampling rate"
        )
    return n_times / float(duration)


class _DeviceLoader:
    """Move batches to ``device`` in the main process.

    PyTorch executes ``collate_fn`` inside worker processes when
    ``num_workers > 0``. Moving tensors to CUDA there is unsupported, so the
    underlying loader always collates CPU tensors and this thin wrapper moves
    them after they return to the main process. Attribute access delegates to
    the wrapped loader so Benchopt solvers still see ``dataset``,
    ``batch_size`` and the other standard DataLoader attributes.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        for X, y, info in self.loader:
            yield X.to(self.device, non_blocking=True), y.to(
                self.device, non_blocking=True), info

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def _make_loaders(loaders, device, target_transform, context_of=None,
                  order_of=None, stream_by=None, num_workers=0):
    """Rebuild the competition loaders from the neuralbench ones.

    Keeps neuralbench's per-split datasets, batch sizes and train sampler
    (e.g. the ``RegressionBinSampler``) but swaps the collate for the
    competition one (plain tensors moved onto ``device``). Validation and
    test windows stay in neuralbench's order, which streamed evaluation
    relies on: never shuffle them. With ``order_of``, their streams'
    recordings are put in acquisition order (the train split keeps its
    order, which its sampler's indices refer to).
    """
    from torch.utils.data import DataLoader

    out = {}
    for split, loader in loaders.items():
        ds = _NBWindows(loader.dataset, target_transform, context_of,
                        order_of=None if split == "train" else order_of,
                        stream_by=stream_by)
        cpu_loader = DataLoader(
            ds, batch_size=loader.batch_size,
            sampler=getattr(loader, "sampler", None)
            if split == "train" else None,
            shuffle=(split == "train"
                     and getattr(loader, "sampler", None) is None),
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
            persistent_workers=num_workers > 0,
        )
        out[split] = _DeviceLoader(cpu_loader, device)
    return out


def load_task(modality, task, *, data_dir, dataset=None, device="cpu",
              batch_size=64, seed=0, num_workers=0, overrides=None,
              target_transform=None, context_of=None, order_of=None,
              subset="full"):
    """Build the competition loaders + meta from a neuralbench task config.

    Parameters
    ----------
    modality : str
        neuralbench task family (``"eeg"`` or ``"emg"``).
    task : str
        Task name (e.g. ``"motor_imagery"``, ``"sleep_onset"``, ``"image"``).
    data_dir : path-like
        Root directory holding the downloaded studies (benchopt data path).
    dataset : str or None
        Optional dataset overlay from the task's ``datasets/`` folder (e.g.
        ``"tangermann2012"``); ``None`` uses the task's default study.
    device : str
        Device the batches are moved onto (see ``benchmark_utils.data``).
    batch_size, seed, num_workers : int
        Dataloader settings.
    overrides : dict or None
        Extra ``data:``-section overrides, as dotted keys (e.g.
        ``{"study.source.query": ...}``). Applied last, so they win over
        the loader settings above.
    target_transform : callable or None
        Applied to each window's target (e.g. one-hot -> class index).
    context_of : callable or None
        Maps the fields of a window's trigger event (a dict) to its context
        label, given as ``info["context_id"]`` (see :class:`_NBWindows`).
    order_of : callable or None
        Maps the fields of a window's trigger event to the acquisition rank
        of its recording within its stream: the validation and test loaders
        then give each stream's recordings in that order (see
        :func:`_acquisition_order`).
    subset : {"full", "test"}
        ``"test"`` restricts the study to its test split via a
        ``filter_stimuli`` override (:func:`build_test_only_filter`) —
        e.g. to stage only the evaluation data on a worker. The
        train/val loaders are then (near-)empty; only use the test one.

    Returns
    -------
    loaders : dict
        ``{"train", "val", "test"}`` -> competition-contract DataLoaders.
    meta : dict
        ``sfreq, ch_names, chs_info, n_chans, n_times, device`` (+ target
        shape info under ``target_shape`` / ``raw_target_shape``).
    """
    from neuralbench.data import Data
    _quiet_neuro_logs()

    from benchmark_utils.data import chs_info_from_names

    # Loader settings first, caller overrides last. ``num_workers=0`` keeps
    # extraction in-process: the task defaults inject N_CPUS, which
    # over-subscribes platform/CI runners (our rewrapped loaders extract
    # windows lazily anyway).
    cfg = _task_data_config(
        modality, task, dataset, data_dir,
        {"batch_size": batch_size, "seed": seed, "num_workers": num_workers,
         "pin_memory": False, "persistent_workers": False,
         **(overrides or {})},
    )
    if subset == "test":
        cfg["study.filter_stimuli"] = build_test_only_filter(cfg)
    elif subset != "full":
        raise ValueError(f"subset must be 'full' or 'test', got {subset!r}")

    nb_loaders = Data(**cfg).prepare()
    loaders = _make_loaders(
        nb_loaders, device, target_transform, context_of, order_of,
        cfg.get("stream_by"), num_workers=num_workers,
    )

    # Peek one window (lazy) for shapes; channel names come from the prepared
    # neuro extractor's channel map.
    X0, y0, _ = loaders["test"].dataset[0]
    raw_y0 = to_numpy(nb_loaders["test"].dataset[0].data["target"])
    if raw_y0.ndim and raw_y0.shape[0] == 1:
        raw_y0 = raw_y0[0]
    ch_names = _channel_names(nb_loaders["test"].dataset)
    neuro = nb_loaders["test"].dataset.extractors["neuro"]
    meta = dict(
        sfreq=_window_sfreq(neuro.frequency, X0.shape[-1],
                            cfg.get("duration")),
        ch_names=ch_names,
        chs_info=chs_info_from_names(ch_names),
        n_chans=int(X0.shape[-2]),
        n_times=int(X0.shape[-1]),
        target_shape=tuple(y0.shape),
        # Shape before ``target_transform`` (e.g. one-hot width -> n_classes).
        raw_target_shape=tuple(raw_y0.shape),
        device=device,
    )
    return loaders, meta
