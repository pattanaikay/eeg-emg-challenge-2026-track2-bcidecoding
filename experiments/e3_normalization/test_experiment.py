"""Behavioral checks for train-only priors and chronological, resettable inference."""

import copy
import io

import pytest
import torch

from experiments.e3_normalization.training import fit_normalizer
from track2.experiments import inference_module


META = {"n_chans": 3, "n_times": 48, "n_classes": 3, "sfreq": 120.0,
        "ch_names": ["C3", "Cz", "C4"]}


def normalizer(mode, X=None):
    config = {"experiment": "e3_normalization", "model": "linear", "normalization": mode}
    instance = inference_module(config).InputNormalizer(META, config)
    if X is not None:
        fit_normalizer(instance, X, config)
    return instance


@pytest.mark.parametrize("mode", ["none", "source-zscore", "window-robust", "causal-zscore", "causal-ea"])
def test_prefix_is_independent_of_future_and_batch_boundaries(mode):
    source = torch.randn(9, 3, 48, generator=torch.Generator().manual_seed(30))
    X = torch.randn(6, 3, 48, generator=torch.Generator().manual_seed(31))
    start = normalizer(mode, source)
    prefix = copy.deepcopy(start)(X[:3])
    changed_future = torch.cat([X[:3], X[3:] * 100 + 200])
    torch.testing.assert_close(prefix, copy.deepcopy(start)(changed_future)[:3])
    sequential = copy.deepcopy(start)
    torch.testing.assert_close(copy.deepcopy(start)(X), torch.cat([sequential(x[None]) for x in X]))


@pytest.mark.parametrize("mode", ["causal-zscore", "causal-ea"])
def test_reset_deepcopy_and_reload_restore_starting_predictions(mode):
    start = normalizer(mode, torch.randn(8, 3, 48))
    X = torch.randn(5, 3, 48)
    before = {key: value.clone() for key, value in start.state_dict().items()}
    clone = copy.deepcopy(start)
    expected = clone(X)
    assert clone.running_count.item() == 15
    assert start.running_count.item() == 10
    clone.reset_state()
    torch.testing.assert_close(clone(X), expected)
    for key, value in before.items():
        torch.testing.assert_close(clone.state_dict()[key], value)
    assert not any("running" in key for key in clone.state_dict())
    checkpoint = io.BytesIO()
    torch.save(clone.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = normalizer(mode)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.reset_state()
    torch.testing.assert_close(restored(X), expected)


def test_running_zscore_includes_current_window_with_source_prior():
    instance = normalizer("causal-zscore")
    X = torch.full((1, 3, 48), 12.0)
    transformed = instance(X)
    mean = 12 / 11
    variance = (10 + 144 * 10 / 11) / 11
    torch.testing.assert_close(transformed, torch.full_like(X, (12 - mean) / variance**0.5))
    torch.testing.assert_close(instance.running_mean, torch.full((3, 1), mean, dtype=torch.float64))
    assert instance.running_count.item() == 11


def test_running_ea_includes_current_covariance_before_transform():
    instance = normalizer("causal-ea")
    X = torch.randn(1, 3, 48) * 30
    covariance = instance.covariances(X)[0]
    expected_covariance = (10 * torch.eye(3, dtype=torch.float64) + covariance) / 11
    instance(X)
    torch.testing.assert_close(instance.running_cov, expected_covariance)


def test_source_fit_uses_only_supplied_training_values_and_inference_cannot_change_it():
    training = torch.randn(6, 3, 48, generator=torch.Generator().manual_seed(8)) * 3 + 7
    instance = normalizer("source-zscore", training)
    torch.testing.assert_close(instance.source_mean.flatten(), training.double().mean(dim=(0, 2)))
    torch.testing.assert_close(instance.source_var.flatten(), training.double().var(dim=(0, 2), unbiased=False))
    before = {key: value.clone() for key, value in instance.state_dict().items()}
    instance(torch.full((2, 3, 48), 10000.0))
    for key, value in before.items():
        torch.testing.assert_close(instance.state_dict()[key], value)
    transformed = instance(training)
    torch.testing.assert_close(transformed.mean(dim=(0, 2)), torch.zeros(3), atol=1e-6, rtol=0)


@pytest.mark.parametrize("mode", ["source-zscore", "window-robust", "causal-zscore", "causal-ea"])
def test_constant_and_rank_deficient_windows_are_finite(mode):
    constants = torch.ones(3, 3, 48) * 15
    instance = normalizer(mode, constants)
    assert torch.isfinite(instance(constants)).all()
    rank_one = torch.randn(3, 1, 48).expand(-1, 3, -1)
    instance = normalizer(mode, rank_one)
    assert torch.isfinite(instance(rank_one)).all()


def test_rejects_invalid_modes_priors_and_nonfinite_training_data():
    with pytest.raises(ValueError, match="Unknown normalization"):
        normalizer("whole-session")
    config = {"experiment": "e3_normalization", "model": "linear", "normalization": "causal-ea",
              "prior_windows": 0}
    with pytest.raises(ValueError, match="prior_windows"):
        inference_module(config).InputNormalizer(META, config)
    with pytest.raises(ValueError, match="finite"):
        normalizer("source-zscore", torch.full((2, 3, 48), float("nan")))


def test_wrapper_probability_order_and_reload_for_linear_control():
    config = {"experiment": "e3_normalization", "model": "linear", "normalization": "causal-zscore"}
    module = inference_module(config)
    model = module.build_model("experiment", META, config).eval()
    fit_normalizer(model.normalizer, torch.randn(6, 3, 48), config)
    with torch.no_grad():
        model.decoder.linear.weight.zero_()
        model.decoder.linear.bias.copy_(torch.tensor([-2.0, 3.0, 1.0]))
    X = torch.randn(1, 3, 48)
    expected = torch.tensor([-2.0, 3.0, 1.0]).softmax(dim=0).unsqueeze(0)
    torch.testing.assert_close(copy.deepcopy(model).predict_proba(X), expected)
    assert copy.deepcopy(model).predict(X).tolist() == [1]
    restored = module.build_model("experiment", META, config).eval()
    restored.load_state_dict(model.state_dict())
    restored.reset_state()
    torch.testing.assert_close(restored.predict_proba(X), expected)
