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
