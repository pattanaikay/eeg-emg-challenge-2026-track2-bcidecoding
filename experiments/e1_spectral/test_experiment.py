"""Behavioral spectral, train-only fit, and serialized inference checks."""

import copy
import io
import json
from pathlib import Path

import pytest
import torch

from track2.experiments import inference_module
from experiments.e1_spectral.training import fit


META = {"n_chans": 4, "n_times": 480, "n_classes": 3,
        "sfreq": 120.0, "ch_names": ["C3", "C4", "F3", "F4"]}


def e1_config(name="logistic"):
    return json.loads((Path(__file__).parent / "configs" / f"{name}.json").read_text())


def e1_model(config):
    return inference_module(config).build_model("experiment", META, config).eval()


def e1_signal_examples(count, seed):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 3
    times = torch.arange(480) / META["sfreq"]
    frequencies = torch.tensor([6.0, 10.0, 20.0])[labels]
    phases = 2 * torch.pi * torch.rand(count, 4, 1, generator=generator)
    amplitudes = 1 + torch.rand(count, 4, 1, generator=generator)
    values = amplitudes * torch.sin(2 * torch.pi * frequencies[:, None, None] * times + phases)
    values += 0.1 * torch.randn(count, 4, 480, generator=generator)
    return values, labels


def test_spectral_features_identify_frequency_and_band_order():
    model = e1_model(e1_config())
    times = torch.arange(480) / META["sfreq"]
    frequencies = torch.tensor([6.0, 10.0, 20.0, 36.0])
    X = torch.sin(2 * torch.pi * frequencies[:, None] * times)[None]
    features = model.decoder.extract_features(X).reshape(4, 4)
    assert features.argmax(dim=1).tolist() == [0, 1, 2, 3]
    assert torch.isfinite(features).all()


def test_low_band_is_a_separate_feature_extension():
    base = e1_model(e1_config())
    extended = e1_model(e1_config("low_logistic"))
    times = torch.arange(480) / META["sfreq"]
    X = torch.sin(2 * torch.pi * 2 * times).expand(1, 4, -1)
    features = extended.decoder.extract_features(X).reshape(4, 5)
    assert features.argmax(dim=1).tolist() == [4] * 4
    assert base.decoder.linear.in_features == 16
    assert extended.decoder.linear.in_features == 20


@pytest.mark.parametrize("head", ["logistic", "lda"])
def test_training_fit_class_order_and_checkpoint_parity(head):
    config = e1_config(head)
    model = e1_model(config)
    X, y = e1_signal_examples(36, 4)
    fit(model.decoder, X, y, config)
    features = model.decoder.extract_features(X)
    torch.testing.assert_close(model.decoder.feature_mean, features.mean(dim=0), atol=1e-6, rtol=1e-5)
    heldout, expected = e1_signal_examples(12, 9)
    state = copy.deepcopy(model.state_dict())
    probabilities = model.predict_proba(heldout)
    assert probabilities.shape == (12, 3)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(12))
    assert probabilities.argmax(dim=1).tolist() == expected.tolist()
    # Transforming an extreme held-out recording must not refit the source scaler.
    model.predict(heldout * 100)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = e1_model(config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(model.predict_proba(heldout[:1]), restored.predict_proba(heldout[:1]))
    assert model.predict(heldout[:1]).shape == (1,)


@pytest.mark.parametrize("bands", [[], [[8, 4]], [[1, 61]], [[float("nan"), 8]], [[4]]])
def test_invalid_bands_are_rejected(bands):
    with pytest.raises(ValueError, match="band"):
        e1_model({**e1_config(), "bands": bands})
