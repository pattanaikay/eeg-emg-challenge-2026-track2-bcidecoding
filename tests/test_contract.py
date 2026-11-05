"""Fast checks separate from the longer train/export smoke script."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import track2
from submission import ConstantModel, Solver, TorchLinear
from track2.smoke import metric_and_stream_checks
from track2.workflow import score
from track2.data import simulated


def test_official_metric_and_stream_boundaries():
    metric_and_stream_checks()


def test_floor_real_input_contract():
    prediction = ConstantModel().predict(torch.zeros(1, 41, 480))
    assert prediction.tolist() == [0]
    assert prediction.dtype == torch.long
    with pytest.raises(ValueError, match="shape"):
        ConstantModel().predict(torch.zeros(41, 480))


def test_missing_trained_weights_never_fall_back_to_random(tmp_path):
    meta = {"n_chans": 41, "n_times": 480, "n_classes": 3, "sfreq": 120.0,
            "ch_names": None, "submission_dir": tmp_path, "device": "cpu"}
    (tmp_path / "model.json").write_text(json.dumps({"model": "linear", "input": {k: meta[k] for k in ("n_chans", "n_times", "n_classes", "sfreq")}}))
    with pytest.raises(FileNotFoundError):
        Solver().load_model(meta)


def test_incomplete_training_run_cannot_be_packaged(tmp_path):
    from track2.packaging import package

    run = tmp_path / "partial"
    run.mkdir()
    for name in ("submission.py", "model.json", "weights.pt"):
        (run / name).write_text("incomplete")
    with pytest.raises(ValueError, match="did not complete"):
        package(tmp_path / "partial.zip", run)
    assert not (tmp_path / "partial.zip").exists()


def test_channel_order_is_a_contract_not_just_a_count(tmp_path):
    meta = {"n_chans": 2, "n_times": 480, "n_classes": 3, "sfreq": 120.0,
            "ch_names": ["C4", "C3"], "submission_dir": tmp_path, "device": "cpu"}
    expected = {**meta, "ch_names": ["C3", "C4"]}
    expected.pop("submission_dir")
    (tmp_path / "model.json").write_text(json.dumps({"model": "linear", "input": expected}))
    with pytest.raises(ValueError, match="channel order"):
        Solver().load_model(meta)


def test_score_matches_upstream_objective(tmp_path):
    from vendor.competition.tracks.bci_decoding.objective import Objective

    loaders, _ = simulated(batch_size=8)
    model = ConstantModel()
    objective = Objective()
    objective.set_data(train_loader=None, test_loader=loaders["test"], n_classes=3)
    reference = objective.evaluate_result(model)
    ours = score(model, loaders["test"], tmp_path, "test")
    assert ours["balanced_accuracy_cell_mean"] == reference["balanced_accuracy_cell_mean"]
    assert ours["balanced_accuracy_session_mean"] == reference["balanced_accuracy_session_mean"]
    assert ours["balanced_accuracy_pooled"] == reference["balanced_accuracy"]
    assert ours["accuracy"] == reference["accuracy"]
    assert (tmp_path / "test_predictions.csv").exists()


def test_linear_inference_preserves_parameters():
    model = TorchLinear(41, 3).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    output = model.predict(torch.randn(1, 41, 480))
    assert output.shape == (1,)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_bids_audit_counts_commands_and_channel_order(tmp_path, monkeypatch):
    import track2.data as data

    monkeypatch.setattr(data, "ROOT", tmp_path / "project")
    source = tmp_path / "bids"
    source.mkdir()
    stem = "sub-P01_ses-S001_task-OnlineRawGraz_run-1"
    (source / f"{stem}_events.tsv").write_text(
        "onset\tduration\ttrial_type\n1\t4\tmi\n6\t4\tsub\n11\t4\tword\n16\t0\tBAD_ACQ_SKIP\n")
    (source / f"{stem}_channels.tsv").write_text("name\ttype\nC3\tEEG\nC4\tEEG\nEOG1\tEOG\n")
    result = data.audit(source)
    assert result["command_events"] == {"mi": 1, "sub": 1, "word": 1}
    assert result["channel_orders"] == [["C3", "C4"]]
    assert result["participant_sessions"] == 1
    assert (tmp_path / "project" / "runs" / "data-audit" / "runs.csv").exists()


def test_export_does_not_import_unshipped_project_code(tmp_path):
    import subprocess
    import zipfile
    from track2.packaging import validate

    archive_path = tmp_path / "private_dependency.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("submission.py", "import track2\n")
        archive.writestr("model.json", '{"model":"constant"}')
    with pytest.raises(subprocess.CalledProcessError):
        validate(archive_path)


def test_detached_configuration_is_local_and_respects_explicit_config(tmp_path, monkeypatch):
    import track2.data as data

    monkeypatch.setattr(data, "ROOT", tmp_path / "project")
    monkeypatch.delenv("NEURALBENCH_CONFIG", raising=False)
    data.configure_neuralbench(tmp_path / "data")
    import os
    generated = Path(os.environ["NEURALBENCH_CONFIG"])
    assert generated.is_relative_to(tmp_path / "project")
    assert json.loads(generated.read_text())["CLUSTER"] is None
    explicit = tmp_path / "explicit.json"
    explicit.write_text('{"do_not_change": true}')
    monkeypatch.setenv("NEURALBENCH_CONFIG", str(explicit))
    data.configure_neuralbench(tmp_path / "different-data")
    assert json.loads(explicit.read_text()) == {"do_not_change": True}



def test_p10_later_sessions_are_reserved_in_chronological_order():
    from types import SimpleNamespace
    from track2.data import reserve_p10_late_indices

    segments, onsets = [], []
    def add(subject, session, run, onset):
        event = {"subject": f"dataset/sub-{subject}", "session": f"ses-{session}",
                 "run": f"run-{run}"}
        segments.append(SimpleNamespace(trigger=SimpleNamespace(to_dict=lambda e=event: e)))
        onsets.append(onset)

    add("P10", 1, 1, 0)
    add("P01", 1, 1, 0)
    for session, count in ((3, 199), (2, 186)):
        for index in range(count):
            add("P10", session, 1 + index // 50, index % 50)

    dataset = SimpleNamespace(seg_ds=SimpleNamespace(segments=segments),
                              index=np.arange(len(segments)), onset=np.asarray(onsets))
    keep, later = reserve_p10_late_indices(dataset)
    assert len(keep) == 2
    assert len(later) == 385
    events = [segments[int(i)].trigger.to_dict() for i in later]
    assert [event["session"] for event in events[:186]] == ["ses-2"] * 186
    assert [event["session"] for event in events[186:]] == ["ses-3"] * 199
    assert [(event["session"], event["run"], onsets[int(i)]) for event, i in zip(events, later)] == sorted(
        (event["session"], event["run"], onsets[int(i)]) for event, i in zip(events, later))


def test_tier_one_feature_models_return_finite_scores():
    from submission import build_model, shrinkage_covariances

    X = torch.randn(2, 4, 120)
    meta = {"n_chans": 4, "n_times": 120, "n_classes": 3, "sfreq": 120.0}
    for kind in ("bandpower", "bandpower-low", "covariance-diagonal", "covariance"):
        model = build_model(kind, meta).eval()
        if kind == "covariance":
            model.reference_cov.copy_(shrinkage_covariances(X).mean(dim=0))
        scores = model(X)
        assert scores.shape == (2, 3)
        assert torch.isfinite(scores).all()
    covariance = shrinkage_covariances(X)
    assert torch.linalg.eigvalsh(covariance).min() > 0


def test_robust_normalization_uses_only_current_window_samples():
    from submission import robust_window_normalize

    X = torch.randn(2, 3, 80)
    first = robust_window_normalize(X[:1])
    both = robust_window_normalize(X)
    torch.testing.assert_close(first, both[:1])
    torch.testing.assert_close(torch.quantile(first, 0.5, dim=-1), torch.zeros(1, 3), atol=1e-5, rtol=0)



def test_source_normalizer_fits_only_the_supplied_training_loader():
    from types import SimpleNamespace
    from track2.workflow import fit_source_normalizer

    loaders, _ = simulated(batch_size=12, seed=51)
    source = loaders["train"].dataset.X
    model = SimpleNamespace(input_mean=torch.zeros(1, source.shape[1], 1),
                            input_scale=torch.ones(1, source.shape[1], 1))
    fit_source_normalizer(model, loaders["train"])
    torch.testing.assert_close(model.input_mean.flatten(), source.mean(dim=(0, 2)))
    torch.testing.assert_close(model.input_scale.flatten(), source.std(dim=(0, 2), unbiased=False))



def test_plain_eegnet_keeps_legacy_state_dict_keys(monkeypatch):
    import sys
    import types
    from torch import nn
    from submission import build_model

    class FakeEEGNet(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, X):
            return X.new_zeros((len(X), 3))

    package = types.ModuleType("braindecode")
    package.__path__ = []
    models = types.ModuleType("braindecode.models")
    models.EEGNet = FakeEEGNet
    monkeypatch.setitem(sys.modules, "braindecode", package)
    monkeypatch.setitem(sys.modules, "braindecode.models", models)

    meta = {"n_chans": 4, "n_times": 120, "n_classes": 3, "sfreq": 120.0}
    assert set(build_model("eegnet", meta).state_dict()) == {"net.weight"}
    assert {"input_mean", "input_scale"} <= set(
        build_model("eegnet-source-zscore", meta).state_dict())



def test_p10_reservation_rebuilds_sampler_and_chronological_loader():
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from benchmark_utils.nb_task import _DeviceLoader
    from track2.data import reserve_p10_later_sessions

    segments, onsets = [], []
    def add(subject, session, run, onset):
        event = {"subject": f"dataset/sub-{subject}", "session": f"ses-{session}",
                 "run": f"run-{run}"}
        segments.append(SimpleNamespace(trigger=SimpleNamespace(to_dict=lambda e=event: e)))
        onsets.append(onset)

    for index in range(944):
        add("P01", 1, 1 + index // 50, index % 50)
    for index in range(192):
        add("P10", 1, 1 + index // 50, index % 50)
    for session, count in ((3, 199), (2, 186)):
        for index in range(count):
            add("P10", session, 1 + index // 50, index % 50)

    class Dataset:
        def __init__(self):
            self.seg_ds = SimpleNamespace(segments=segments)
            self.index = np.arange(len(segments))
            self.onset = np.asarray(onsets)

        def __len__(self):
            return len(self.index)

        def __getitem__(self, index):
            position = int(self.index[index])
            session = int(segments[position].trigger.to_dict()["session"].split("-")[-1])
            return (torch.zeros(2, 8), torch.tensor(0),
                    {"record_id": position, "onset": onsets[position],
                     "stream_id": session, "context_id": 0})

    dataset = Dataset()
    original = _DeviceLoader(DataLoader(dataset, batch_size=32, shuffle=True), "cpu")
    validation = DataLoader(torch.utils.data.TensorDataset(torch.zeros(379, 1), torch.zeros(379)))
    test = DataLoader(torch.utils.data.TensorDataset(torch.zeros(755, 1), torch.zeros(755)))
    loaders = {"train": original, "val": validation, "test": test}
    longitudinal = reserve_p10_later_sessions(loaders)
    assert len(loaders["train"].dataset) == 1136
    assert len(longitudinal.dataset) == 385
    seen = [int(i) for _X, _y, info in longitudinal for i in info["record_id"].tolist()]
    assert seen == [int(i) for i in longitudinal.dataset.index]
    assert len(set(seen)) == 385
    sessions = [segments[i].trigger.to_dict()["session"] for i in seen]
    assert sessions == ["ses-2"] * 186 + ["ses-3"] * 199
