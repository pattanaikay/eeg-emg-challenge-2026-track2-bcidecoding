"""E2 checks spatial information, fitting boundaries and checkpoint behavior."""

import copy
import json
from pathlib import Path

import pytest
import torch

from track2.experiments import inference_module
from experiments.e2_covariance.training import fit


def e2_config(kind="covariance", head="logistic"):
    return {"experiment": "e2_covariance", "model": kind,
            "normalization": "none", "head": head}


def e2_meta(channels=4, samples=120):
    return {"n_chans": channels, "n_times": samples, "n_classes": 3,
            "sfreq": 120.0, "ch_names": [f"C{i}" for i in range(channels)]}


def e2_training_windows():
    generator = torch.Generator().manual_seed(27)
    labels = torch.tensor([2, 0, 1] * 12)
    windows = torch.randn(len(labels), 4, 120, generator=generator)
    windows[torch.arange(len(labels)), labels] *= 4
    return windows, labels


def test_full_covariance_keeps_a_relationship_discarded_by_diagonal_control():
    config = e2_config()
    module = inference_module(config)
    time = torch.arange(120, dtype=torch.float64) / 120
    sine = torch.sin(2 * torch.pi * 7 * time)
    cosine = torch.cos(2 * torch.pi * 7 * time)
    positive = torch.stack([sine, 0.8 * sine + 0.6 * cosine])
    negative = positive.clone()
    negative[1] *= -1
    windows = torch.stack([positive, negative])
    full = module.build_model("experiment", e2_meta(2), config).decoder
    diagonal = module.build_model(
        "experiment", e2_meta(2), e2_config("covariance-diagonal")).decoder
    full_features = full.extract_features(windows)
    diagonal_features = diagonal.extract_features(windows)
    torch.testing.assert_close(diagonal_features[0], diagonal_features[1])
    assert not torch.allclose(full_features[0], full_features[1])
    assert full_features[0, 1] * full_features[1, 1] < 0


def test_covariance_is_centered_positive_definite_on_degenerate_windows():
    module = inference_module(e2_config())
    generator = torch.Generator().manual_seed(8)
    source = torch.randn(2, 4, 120, dtype=torch.float64, generator=generator)
    torch.testing.assert_close(module.shrinkage_covariances(source),
                               module.shrinkage_covariances(source + 100),
                               atol=1e-11, rtol=1e-11)
    source[:, 1:] = source[:, :1]
    for windows in (source, torch.zeros_like(source), source * 1e8):
        covariance = module.shrinkage_covariances(windows)
        assert torch.isfinite(covariance).all()
        assert torch.linalg.eigvalsh(covariance).min() > 0
        model = module.build_model("experiment", e2_meta(), e2_config())
        assert torch.isfinite(model(windows.float())).all()


def test_diagonal_features_ignore_independent_channel_sample_permutations():
    config = e2_config()
    module = inference_module(config)
    generator = torch.Generator().manual_seed(91)
    common = torch.randn(1, 480, generator=generator)
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0])[:, None]
    source = scales * common + 0.02 * torch.randn(4, 480, generator=generator)
    permuted = torch.stack([channel[torch.randperm(480, generator=generator)] for channel in source])
    windows = torch.stack([source, permuted])
    full = module.build_model("experiment", e2_meta(samples=480), config).decoder
    diagonal = module.build_model("experiment", e2_meta(samples=480),
                                  e2_config("covariance-diagonal")).decoder
    diagonal_features = diagonal.extract_features(windows)
    torch.testing.assert_close(diagonal_features[0], diagonal_features[1], atol=1e-7, rtol=1e-7)
    full_features = full.extract_features(windows)
    assert not torch.allclose(full_features[0], full_features[1], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("kind", ["covariance", "covariance-diagonal"])
@pytest.mark.parametrize("head", ["logistic", "lda"])
def test_fitted_class_order_reference_immutability_and_saved_reload(kind, head, tmp_path):
    config = e2_config(kind, head)
    module = inference_module(config)
    meta = e2_meta()
    model = module.build_model("experiment", meta, config).eval()
    windows, labels = e2_training_windows()
    fit(model.decoder, windows, labels, config)
    if kind == "covariance":
        torch.testing.assert_close(model.decoder.reference_cov,
                                   module.shrinkage_covariances(windows).mean(dim=0))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    probabilities = model.predict_proba(windows)
    assert probabilities.shape == (len(labels), 3)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(len(labels)))
    assert torch.equal(probabilities.argmax(dim=1), labels)
    future = windows[-1:] * 100 + 20
    model.predict(future)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    (tmp_path / "model.json").write_text(json.dumps(
        {"model": "experiment", "config": config, "input": meta}), encoding="utf-8")
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    restored = module.Solver().load_model(
        {**meta, "submission_dir": tmp_path, "device": "cpu"})
    torch.testing.assert_close(restored.predict_proba(windows), probabilities, rtol=0, atol=0)
    assert torch.equal(restored.predict(windows[:1]), labels[:1])
    clone = copy.deepcopy(model)
    clone.predict(future)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_presets_have_matching_heads_and_reject_invalid_fits():
    directory = Path(__file__).parent / "configs"
    for head in ("logistic", "lda"):
        full = json.loads((directory / f"tangent_{head}.json").read_text())
        diagonal = json.loads((directory / f"diagonal_{head}.json").read_text())
        assert {**full, "model": diagonal["model"]} == diagonal
    config = e2_config()
    module = inference_module(config)
    model = module.build_model("experiment", e2_meta(), config)
    with pytest.raises(ValueError, match="finite EEG"):
        fit(model.decoder, torch.full((3, 4, 120), float("nan")), torch.arange(3), config)
    with pytest.raises(ValueError, match="Unknown E2"):
        module.build_model("experiment", e2_meta(), e2_config("unknown"))
