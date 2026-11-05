"""Cross-experiment safeguards at split, fitting and exported inference boundaries."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from track2 import ROOT
from track2.experiments import inference_module, load_config
from track2.splits import membership, participant


def test_shrinkage_covariance_matches_independent_sklearn_oas():
    from sklearn.covariance import oas
    from submission import shrinkage_covariances

    generator = torch.Generator().manual_seed(14)
    X = torch.randn(3, 41, 480, generator=generator, dtype=torch.float64)
    actual = shrinkage_covariances(X)
    for index, window in enumerate(X):
        expected, _ = oas(window.T.numpy())
        expected *= window.shape[-1] / (window.shape[-1] - 1)
        centered = window - window.mean(dim=-1, keepdim=True)
        mu = (centered @ centered.T / (window.shape[-1] - 1)).diagonal().mean()
        expected += np.eye(41) * float(mu.abs() * 1e-8 + 1e-8)
        np.testing.assert_allclose(actual[index].numpy(), expected, rtol=1e-10, atol=1e-12)


def test_legacy_model_cannot_silently_ignore_an_outer_protocol(tmp_path):
    from track2.workflow import train

    with pytest.raises(ValueError, match="requires --config"):
        train(SimpleNamespace(config=None, model="linear", protocol="outer"))


def test_all_protocols_reserve_later_visits_before_fitting():
    # Use synthetic metadata here so tests do not require ignored run artifacts.
    rows = [{"subject": f"dataset/P{person:02}", "session": str(session),
             "task": "AcquisitionGraz", "run": "1", "onset": float(person)}
            for person in range(1, 11) for session in ([1, 2, 3] if person == 10 else [1])]
    all_test_subjects = []
    for protocol, fold in [("screen", None), *(("outer", index) for index in range(4))]:
        assigned = membership(rows, protocol, fold)
        groups = {split: {participant(row["subject"]) for row in assigned if row["split"] == split}
                  for split in ("train", "val", "test", "longitudinal", "diagnostic")}
        assert not groups["train"] & groups["val"]
        assert not groups["train"] & groups["test"]
        assert not groups["val"] & groups["test"]
        assert [row["session"] for row in assigned if row["split"] == "longitudinal"] == ["2", "3"]
        assert all(row["split"] == "longitudinal" for row in assigned
                   if participant(row["subject"]) == "P10" and row["session"] != "1")
        if protocol == "outer":
            assert groups["diagnostic"] == {"P04", "P09"}
            all_test_subjects.extend(groups["test"])
    assert sorted(all_test_subjects) == ["P01", "P02", "P03", "P05", "P06", "P07", "P08", "P10"]
    with pytest.raises(ValueError, match="more than once"):
        membership(rows + rows[:1])


def test_partition_assignment_is_independent_of_labels_and_initialization():
    rows = [{"subject": f"P{person:02}", "session": "1", "task": "AcquisitionGraz", "run": "1", "onset": 1.0}
            for person in range(1, 11)]
    original = membership(rows, "outer", 2)
    changed = membership([{**row, "outer_label": 2 - index % 3} for index, row in enumerate(rows)], "outer", 2)
    assert [row["split"] for row in original] == [row["split"] for row in changed]
    for args in (("outer", None), ("outer", 4), ("screen", 0)):
        with pytest.raises(ValueError):
            membership(rows, *args)


def test_outer_label_changes_leave_the_fitted_statistical_pipeline_identical(tmp_path, monkeypatch):
    import track2.data
    from track2.workflow import train

    config_path = ROOT / "experiments/e1_spectral/configs/logistic.json"
    original_load = track2.data.load
    def load_with_outer_labels(*args, **kwargs):
        loaders, meta = original_load(*args, **kwargs)
        loaders["test"].dataset.y = (loaders["test"].dataset.y + 1) % 3
        return loaders, meta
    states = []
    for index in range(2):
        if index:
            monkeypatch.setattr(track2.data, "load", load_with_outer_labels)
        run = tmp_path / str(index)
        train(SimpleNamespace(config=config_path, model=None, data="simulated", data_dir=ROOT / "data",
                              output=run, seed=33, epochs=None, lr=None, batch_size=16,
                              device="cpu", num_workers=0, protocol="screen", fold=None))
        states.append(torch.load(run / "weights.pt", weights_only=True))
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("config_name", ["logistic", "lda"])
def test_exported_pipeline_reloads_probability_order_and_rejects_bad_input(tmp_path, config_name):
    from experiments.e1_spectral.training import fit

    config = load_config(ROOT / f"experiments/e1_spectral/configs/{config_name}.json")
    module = inference_module(config)
    meta = {"n_chans": 41, "n_times": 480, "n_classes": 3, "sfreq": 120.0, "ch_names": [str(i) for i in range(41)]}
    model = module.build_model("experiment", meta, config)
    X = torch.randn(18, 41, 480)
    y = torch.arange(18) % 3
    fit(model.decoder, X, y, config)
    probabilities = model.predict_proba(X)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    (tmp_path / "model.json").write_text(json.dumps({"model": "experiment", "input": meta, "config": config}))
    loaded = module.Solver().load_model({**meta, "device": "cpu", "submission_dir": tmp_path})
    torch.testing.assert_close(loaded.predict_proba(X), probabilities, rtol=0, atol=0)
    assert torch.equal(loaded.predict(X), probabilities.argmax(dim=1))
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(len(X)))
    for invalid in (X[:, :, :-1], X[0], torch.full_like(X[:1], float("nan"))):
        with pytest.raises(ValueError):
            loaded.predict(invalid)
    with pytest.raises(ValueError, match="channel order"):
        module.Solver().load_model({**meta, "ch_names": meta["ch_names"][::-1], "device": "cpu", "submission_dir": tmp_path})


def test_causal_solver_resets_to_reloaded_source_before_direct_inference(tmp_path):
    from track2.experiments import fit_inputs

    config = {"experiment": "e3_normalization", "model": "linear", "normalization": "causal-zscore"}
    module = inference_module(config)
    meta = {"n_chans": 3, "n_times": 40, "n_classes": 3, "sfreq": 120.0, "ch_names": None}
    model = module.build_model("experiment", meta, config)
    X = 7 + 2 * torch.randn(9, 3, 40)
    fit_inputs(model, X, {"stream_id": np.zeros(9)}, config)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    (tmp_path / "model.json").write_text(json.dumps({"model": "experiment", "input": meta, "config": config}))
    loaded = module.Solver().load_model({**meta, "device": "cpu", "submission_dir": tmp_path})
    torch.testing.assert_close(model.predict_proba(X[:1]), loaded.predict_proba(X[:1]), rtol=0, atol=0)
