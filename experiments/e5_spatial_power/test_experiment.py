"""Check maintained E5 interfaces, actual bank response and immutable export state."""

import copy
import json
from pathlib import Path

import pytest
import torch

from track2.experiments import inference_module


def e5_config(kind):
    path = Path(__file__).parent / "configs" / f"{kind}.json"
    return json.loads(path.read_text())


def e5_meta():
    return {"n_chans": 41, "n_times": 480, "n_classes": 3,
            "sfreq": 120.0, "ch_names": None}


@pytest.mark.parametrize("kind", ["shallowconvnet", "fbcnet"])
def test_maintained_model_gradient_single_window_and_solver_reload(kind, tmp_path):
    from braindecode.models import FBCNet, ShallowFBCSPNet

    torch.manual_seed(17)
    config, meta = e5_config(kind), e5_meta()
    module = inference_module(config)
    model = module.build_model("experiment", meta, config).train()
    expected_class = ShallowFBCSPNet if kind == "shallowconvnet" else FBCNet
    assert isinstance(model.decoder, expected_class)
    windows = torch.randn(2, 41, 480)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scores = model(windows)
    loss = torch.nn.functional.cross_entropy(scores, torch.tensor([2, 0]))
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.requires_grad]
    assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert any(value.abs().sum() > 0 for value in gradients)
    optimizer.step()
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    probabilities = model.predict_proba(windows[:1])
    assert probabilities.shape == (1, 3)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(1))
    assert model.predict(windows[:1]).shape == (1,)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    torch.testing.assert_close(copy.deepcopy(model).predict_proba(windows[:1]), probabilities,
                               atol=0, rtol=0)
    (tmp_path / "model.json").write_text(json.dumps(
        {"model": "experiment", "config": config, "input": meta}), encoding="utf-8")
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    restored = module.Solver().load_model({**meta, "submission_dir": tmp_path, "device": "cpu"})
    torch.testing.assert_close(restored.predict_proba(windows[:1]), probabilities, atol=0, rtol=0)
    for special in (torch.zeros(1, 41, 480), windows[:1] * 1e4):
        assert torch.isfinite(restored(special)).all()


def test_fbcnet_four_band_response_and_sample_count():
    config = e5_config("fbcnet")
    model = inference_module(config).build_model("experiment", e5_meta(), config).eval()
    time = torch.arange(480) / 120
    frequencies = torch.tensor([6.0, 10.0, 20.0, 37.0])
    windows = torch.sin(2 * torch.pi * frequencies[:, None, None] * time)
    windows = windows.expand(-1, 41, -1)
    with torch.no_grad():
        filtered = model.decoder.spectral_filtering(windows)
    assert filtered.shape == (4, 4, 41, 480)
    interior_power = filtered[..., 100:-100].square().mean(dim=(2, 3))
    assert torch.equal(interior_power.argmax(dim=1), torch.arange(4))
    for parameter in model.decoder.spectral_filtering.parameters():
        assert not parameter.requires_grad


def test_fbcnet_max_norm_parametrization_never_mutates_saved_weights():
    config = e5_config("fbcnet")
    model = inference_module(config).build_model("experiment", e5_meta(), config).eval()
    with torch.no_grad():
        model.decoder.spatial_conv[0].parametrizations.weight.original.mul_(100)
        model.decoder.final_layer.parametrizations.weight.original.mul_(100)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.predict(torch.randn(1, 41, 480))
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert model.decoder.final_layer.weight.norm(dim=1).max() <= 0.500001


@pytest.mark.parametrize("kind", ["shallowconvnet", "fbcnet"])
def test_e5_invalid_widths_and_matched_input_policy(kind):
    config = e5_config(kind)
    other = e5_config("fbcnet" if kind == "shallowconvnet" else "shallowconvnet")
    assert config["normalization"] == other["normalization"] == "none"
    module = inference_module(config)
    with pytest.raises(ValueError, match="positive integer"):
        module.build_model("experiment", e5_meta(), {**config, "n_filters_spat": 0})
    if kind == "fbcnet":
        with pytest.raises(ValueError, match="Nyquist"):
            module.build_model("experiment", e5_meta(), {**config, "bands": [[4, 60]]})
    else:
        with pytest.raises(ValueError, match="drop_prob"):
            module.build_model("experiment", e5_meta(), {**config, "drop_prob": 1.0})
