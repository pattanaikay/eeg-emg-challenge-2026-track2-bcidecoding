"""Probe one extracted submission without fitting or using development modules."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "competition"))

import torch


def probe(directory):
    directory = Path(directory).resolve()
    spec = importlib.util.spec_from_file_location("exported_track2", directory / "submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config_path = directory / "model.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {"model": "constant"}
    meta = {"n_chans": 41, "n_times": 480, "n_classes": 3, "sfreq": 120.0,
            "ch_names": None, "submission_dir": directory, "device": "cpu"}
    if config.get("input"):
        meta.update(config["input"])
    model = module.Solver().load_model(meta)
    windows = torch.randn(3, meta["n_chans"], meta["n_times"])
    baseline = None
    for _ in range(2):
        fresh = copy.deepcopy(model)
        reset = getattr(fresh, "reset_state", None)
        if callable(reset):
            reset()
        before = {k: v.clone() for k, v in fresh.state_dict().items()} if hasattr(fresh, "state_dict") else {}
        predictions = torch.cat([torch.as_tensor(fresh.predict(x[None])).cpu() for x in windows])
        assert predictions.shape == (3,), predictions.shape
        assert predictions.dtype in (torch.int32, torch.int64), predictions.dtype
        assert torch.all((predictions >= 0) & (predictions < meta["n_classes"]))
        if baseline is not None:
            assert torch.equal(predictions, baseline), "Fresh stream changed deterministic baseline predictions"
        baseline = predictions
        for key, value in before.items():
            assert torch.equal(value, fresh.state_dict()[key]), f"Inference mutated {key}"
    print(f"PASS: exported {config['model']} loads, predicts single windows, deep-copies, and preserves state")


if __name__ == "__main__":
    probe(sys.argv[1])
