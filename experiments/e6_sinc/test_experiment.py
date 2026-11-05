"""Frequency constraints, matched downstream stages and real single-window contracts."""

import copy

import pytest
import torch

from track2.experiments import inference_module


META = {"n_chans": 41, "n_times": 480, "n_classes": 3, "sfreq": 120.0,
        "ch_names": [f"EEG{index}" for index in range(41)]}


def build(kind, **kwargs):
    config = {"experiment": "e6_sinc", "model": kind, "normalization": "none", **kwargs}
    return inference_module(config).build_model("experiment", META, config)


@pytest.mark.parametrize("kind", ["sinc", "free"])
def test_forward_backward_and_single_window_probability_contract(kind):
    model = build(kind)
    X = torch.randn(3, 41, 480)
    loss = torch.nn.functional.cross_entropy(model(X), torch.tensor([0, 1, 2]))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert model.decoder.temporal.kernels().shape == (8, 1, 31)
    probabilities = model.predict_proba(X[:1])
    assert probabilities.shape == (1, 3) and torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))
    assert model.predict(X[:1]).tolist() == probabilities.argmax(dim=1).tolist()


def test_sinc_cutoffs_survive_optimization_and_saturated_parameters():
    model = build("sinc")
    optimizer = torch.optim.SGD(model.parameters(), lr=50)
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(torch.randn(3, 41, 480)), torch.tensor([0, 1, 2]))
        loss.backward()
        optimizer.step()
        low, high = model.decoder.temporal.cutoffs()
        assert torch.all((0 < low) & (low < high) & (high < 60))
        assert torch.isfinite(model.decoder.temporal.kernels()).all()
    temporal = model.decoder.temporal
    for low_value, width_value in ((-1e6, -1e6), (-1e6, 1e6), (1e6, -1e6), (1e6, 1e6)):
        with torch.no_grad():
            temporal.raw_low.fill_(low_value)
            temporal.raw_width.fill_(width_value)
        temporal.zero_grad()
        low, high = temporal.cutoffs()
        assert torch.all((0 < low) & (low < high) & (high < 60))
        temporal.kernels().square().sum().backward()
        assert torch.isfinite(temporal.raw_low.grad).all()
        assert torch.isfinite(temporal.raw_width.grad).all()


def test_only_temporal_parameterization_differs_and_initial_filters_match():
    torch.manual_seed(7)
    sinc = build("sinc").decoder
    torch.manual_seed(7)
    free = build("free").decoder
    torch.testing.assert_close(sinc.temporal.kernels(), free.temporal.kernels(), atol=1e-6, rtol=1e-5)
    sinc_shared = {key: value for key, value in sinc.state_dict().items() if not key.startswith("temporal.")}
    free_shared = {key: value for key, value in free.state_dict().items() if not key.startswith("temporal.")}
    assert sinc_shared.keys() == free_shared.keys()
    for key in sinc_shared:
        torch.testing.assert_close(sinc_shared[key], free_shared[key])
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in (sinc, free)]
    assert counts[1] - counts[0] == 8 * (31 - 2)


@pytest.mark.parametrize("kind", ["sinc", "free"])
def test_zero_signal_and_zero_free_kernel_are_numerically_stable(kind):
    model = build(kind)
    if kind == "free":
        with torch.no_grad():
            model.decoder.temporal.taps.zero_()
    result = model(torch.zeros(2, 41, 480))
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["sinc", "free"])
def test_strict_checkpoint_reload_and_class_order(tmp_path, kind):
    model = build(kind).eval()
    with torch.no_grad():
        model.decoder.head.weight.zero_()
        model.decoder.head.bias.copy_(torch.tensor([0.0, -2.0, 3.0]))
    X = torch.randn(1, 41, 480)
    expected = model.predict_proba(X)
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = build(kind).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    torch.testing.assert_close(restored.predict_proba(X), expected, atol=0, rtol=0)
    assert restored.predict(X).tolist() == [2]
    clone = copy.deepcopy(restored)
    before = {key: value.clone() for key, value in restored.state_dict().items()}
    clone.predict(X)
    for key, value in before.items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize("settings", [{"kernel_length": 30}, {"n_filters": 0},
                                      {"pool_bins": 481}, {"dropout": float("nan")},
                                      {"model": "gmm"}])
def test_invalid_configuration_is_rejected(settings):
    kind = settings.get("model", "sinc")
    with pytest.raises(ValueError):
        build(kind, **{key: value for key, value in settings.items() if key != "model"})
