"""CSP must detect spatial structure while preserving source/evaluation boundaries."""

import copy
import io
import json
from pathlib import Path

import pytest
import torch

from experiments.e4_csp.training import fit
from track2.experiments import inference_module


META = {"n_chans": 4, "n_times": 480, "n_classes": 3,
        "sfreq": 120.0, "ch_names": ["C3", "C4", "F3", "F4"]}


def e4_config(name="csp"):
    return json.loads((Path(__file__).parent / "configs" / f"{name}.json").read_text())


def e4_model(config):
    return inference_module(config).build_model("experiment", META, config).eval()


def e4_spatial_examples(count, seed):
    generator = torch.Generator().manual_seed(seed)
    y = torch.arange(count) % 3
    patterns = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.], [1., 1., -1., -1.]])
    times = torch.arange(480) / META["sfreq"]
    phase = 2 * torch.pi * torch.rand(count, 1, 1, generator=generator)
    carrier = torch.sin(2 * torch.pi * 10 * times + phase)
    X = patterns[y, :, None] * carrier
    return X + 0.15 * torch.randn(count, 4, 480, generator=generator), y


def test_unprojected_filter_bank_exactly_matches_e1_power_features():
    config = e4_config("fbcsp_control")
    module = inference_module(config)
    model = module.build_model("experiment", META, config).eval()
    X = torch.randn(5, 4, 480, generator=torch.Generator().manual_seed(11))
    torch.testing.assert_close(model.decoder.extract_features(X),
                               module.band_power_features(X, META["sfreq"], config["bands"]),
                               atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", ["csp", "fbcsp"])
def test_csp_learns_spatial_structure_and_reloads_without_refitting(variant):
    config = e4_config(variant)
    model = e4_model(config)
    X, y = e4_spatial_examples(36, 3)
    fit(model.decoder, X, y, config)
    heldout, expected = e4_spatial_examples(12, 8)
    before = copy.deepcopy(model.state_dict())
    probabilities = model.predict_proba(heldout)
    assert probabilities.argmax(dim=1).tolist() == expected.tolist()
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(12))
    model.predict(heldout * 100)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = e4_model(config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored.predict_proba(heldout[:1]), model.predict_proba(heldout[:1]))
    assert restored.predict(heldout[:1]).shape == (1,)


def test_spatial_fit_requires_three_classes_and_precedes_inference():
    config = e4_config()
    model = e4_model(config)
    X, y = e4_spatial_examples(12, 2)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model(X[:1])
    with pytest.raises(ValueError, match="classes"):
        fit(model.decoder, X[y != 2], y[y != 2], config)


def test_zero_and_rank_deficient_training_windows_remain_numerically_stable():
    config = e4_config()
    model = e4_model(config)
    X = torch.zeros(9, 4, 480)
    fit(model.decoder, X, torch.arange(9) % 3, config)
    assert torch.isfinite(model.decoder.spatial_filters).all()
    assert torch.isfinite(model.predict_proba(X[:1])).all()


@pytest.mark.parametrize("change", [
    {"bands": [[4, 61]]}, {"bands": [[0, 4]]}, {"bands": [[8, 4]]},
    {"bands": [[4, 8], [8, 13]]}, {"components_per_class": 3},
    {"components_per_class": 6}, {"model": "fbcsp", "bands": [[4, 8]]},
])
def test_invalid_csp_config_is_rejected(change):
    with pytest.raises(ValueError):
        e4_model({**e4_config(), **change})
