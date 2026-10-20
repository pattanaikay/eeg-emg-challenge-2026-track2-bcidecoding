"""Unit tests for the pure parts of nb_task.

Run with ``pytest benchmark_utils``.

The loading pipeline itself needs the neuralbench stack and real data — it is
validated on the cluster; these tests cover the subset-filter composition,
which is plain dict/string logic, the windows' sampling rate, and the
per-window info and order of the loaders on stand-in segments.
"""

import pytest
import torch

from benchmark_utils.nb_task import (
    _make_loaders, _NBWindows, _window_sfreq, build_test_only_filter,
)


def test_predefined_split_with_query_and_existing_filter():
    # xu2025alljoined shape: explicit test query + a shipped filter_stimuli.
    cfg = {
        "study": {
            "split": {"name": "PredefinedSplit",
                      "test_split_query": "label == 'stim_test'"},
            "filter_stimuli": {
                "name": "QueryEvents",
                "query": "type != 'Image' or label in "
                         "['stim_train', 'stim_test']",
            },
        },
        "trigger_event_type": "Image",
    }
    out = build_test_only_filter(cfg)
    assert out["name"] == "QueryEvents"
    assert out["query"] == (
        "(type != 'Image' or label in ['stim_train', 'stim_test']) and "
        "(type not in ['Image'] or (label == 'stim_test'))"
    )


def test_predefined_split_on_data_column():
    # gifford shape: the split column ships with the source data.
    cfg = {
        "study": {"split": {"name": "PredefinedSplit",
                            "test_split_query": None, "col_name": "split"}},
        "trigger_event_type": "Image",
    }
    assert build_test_only_filter(cfg)["query"] == (
        "type not in ['Image'] or (split == 'test')"
    )


def test_runtime_split_is_rejected():
    cfg = {
        "study": {"split": {"name": "SklearnSplit"}},
        "trigger_event_type": "Stimulus",
    }
    with pytest.raises(ValueError, match="PredefinedSplit"):
        build_test_only_filter(cfg)


def test_missing_split_is_rejected():
    with pytest.raises(ValueError, match="PredefinedSplit"):
        build_test_only_filter({"study": {}, "trigger_event_type": "Stimulus"})


def test_window_sfreq_is_the_extractor_rate():
    assert _window_sfreq(120.0, 480, 4.0) == 120.0


def test_window_sfreq_at_native_rate_is_read_off_the_window():
    # Muse: 5-s windows of 640 samples at the recorded 128 Hz.
    assert _window_sfreq("native", 640, 5.0) == 128.0


def test_window_sfreq_at_native_rate_needs_a_duration():
    with pytest.raises(ValueError, match="duration"):
        _window_sfreq("native", 640, None)


class _Trigger:
    def __init__(self, **fields):
        self.fields = fields

    def to_dict(self):
        return dict(self.fields)


class _Segment:
    def __init__(self, timeline, start, **fields):
        self.timeline, self.start = timeline, start
        self.trigger = _Trigger(timeline=timeline, start=start, **fields)


class _Item:
    def __init__(self, data):
        self.data = data


class _SegmentDataset:
    """What ``_NBWindows`` reads of a neuralset ``SegmentDataset``."""

    def __init__(self, segments, streams=None):
        self.segments, self.streams = segments, streams

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        data = {"neuro": torch.zeros(1, 2, 3),
                "target": torch.tensor([[0.0, 1.0]]),
                "subject_id": torch.tensor([7])}
        if self.streams is not None:
            data["stream_id"] = torch.tensor([self.streams[i]])
        return _Item(data)


def test_windows_info_traces_recording_stream_and_context():
    segments = [_Segment("a", 0.0, task="AcquisitionGraz"),
                _Segment("a", 4.0, task="AcquisitionGraz"),
                _Segment("b", 0.0, task="OnlineRawBH"),
                _Segment("c", 2.5, task="OnlineRawGraz")]
    interface = {"AcquisitionGraz": "Graz", "OnlineRawGraz": "Graz",
                 "OnlineRawBH": "BrainHero"}
    windows = _NBWindows(_SegmentDataset(segments, streams=[3, 3, 3, 5]),
                         context_of=lambda trigger: interface[trigger["task"]])
    infos = [windows[i][2] for i in range(len(windows))]
    assert [i["record_id"] for i in infos] == [0, 0, 1, 2]
    assert [i["onset"] for i in infos] == [0.0, 4.0, 0.0, 2.5]
    assert [i["stream_id"] for i in infos] == [3, 3, 3, 5]
    assert [i["context_id"] for i in infos] == [0, 0, 1, 0]
    assert {i["subject_id"] for i in infos} == {7}
    X, y, _ = windows[0]
    assert X.shape == (2, 3) and y.tolist() == [0.0, 1.0]

    # Without stream_by the recording is the stream, and without context_of
    # there is no context.
    windows = _NBWindows(_SegmentDataset(segments))
    info = windows[2][2]
    assert info["stream_id"] == info["record_id"] == 1
    assert "context_id" not in info


class _Loader:
    """What ``_make_loaders`` reads of a neuralbench DataLoader."""

    def __init__(self, dataset, batch_size=1, sampler=None):
        self.dataset, self.batch_size = dataset, batch_size
        self.sampler = sampler


def test_evaluation_loaders_follow_acquisition_order():
    # One session's runs as BIDS file names sort them, by task (runs 1, 2,
    # 3, 5, 4, 6), two windows each; then a second session listed backwards.
    runs = [("AcquisitionBH", 1), ("AcquisitionGraz", 2), ("OnlineRawBH", 3),
            ("OnlineRawBH", 5), ("OnlineRawGraz", 4), ("OnlineRawGraz", 6)]
    segments = [_Segment(f"ses-1_run-{run}", start, subject="P04",
                         session="1", task=task, run=str(run))
                for task, run in runs for start in (0.0, 4.0)]
    segments += [
        _Segment("ses-2_run-2", 4.0, subject="P04", session="2",
                 task="AcquisitionGraz", run="2"),
        _Segment("ses-2_run-2", 0.0, subject="P04", session="2",
                 task="AcquisitionGraz", run="2"),
        _Segment("ses-2_run-1", 0.0, subject="P04", session="2",
                 task="AcquisitionBH", run="1"),
    ]
    loaders = _make_loaders(
        {"train": _Loader(_SegmentDataset(segments)),
         "test": _Loader(_SegmentDataset(segments))},
        "cpu", None, order_of=lambda trigger: int(trigger["run"]),
        stream_by=["subject", "session"])

    order = [segments[j] for j in loaders["test"].dataset.index]
    assert [(s.timeline, s.start) for s in order] == (
        [(f"ses-1_run-{run}", start)
         for run in range(1, 7) for start in (0.0, 4.0)]
        + [("ses-2_run-1", 0.0), ("ses-2_run-2", 0.0), ("ses-2_run-2", 4.0)])
    # Each window the test loader yields carries its own recording's info.
    onsets = [float(info["onset"]) for _, _, info in loaders["test"]]
    assert onsets == [s.start for s in order]
    # The train split keeps neuralbench's order, which its sampler's
    # indices refer to.
    train_index = loaders["train"].dataset.index.tolist()
    assert train_index == list(range(len(segments)))


def test_windows_keep_neuralbench_order_without_order_of():
    segments = [_Segment("b", 0.0, run="2"), _Segment("a", 0.0, run="1")]
    assert _NBWindows(_SegmentDataset(segments)).index.tolist() == [0, 1]


def test_rebuilt_loader_keeps_requested_worker_count():
    segments = [_Segment("a", 0.0)]
    loaders = _make_loaders(
        {"test": _Loader(_SegmentDataset(segments))}, "cpu", None,
        num_workers=2,
    )
    assert loaders["test"].num_workers == 2
    X, y, info = next(iter(loaders["test"]))
    assert X.device.type == y.device.type == "cpu"
    assert info["record_id"].tolist() == [0]
