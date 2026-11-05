"""Offline checks of scientific boundaries and of the real export/reload path.

The synthetic fixture is deliberately learnable; success establishes working
optimization and serialization, never performance on actual EEG.
"""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from track2 import ROOT


class Counter:
    """A test-only stateful model that exposes accidental cross-session carryover."""

    def __init__(self):
        self.seen = 0

    def reset_state(self):
        self.seen = 0

    def predict(self, X):
        assert len(X) == 1, "Evaluator must deliver one window per call"
        self.seen += 1
        return np.array([self.seen])


def metric_and_stream_checks():
    from benchmark_utils.data import make_loader
    from benchmark_utils.metrics import balanced_accuracy, group_scores
    from benchmark_utils.streaming import predict_streams

    # Including a predicted-but-absent class is part of this scorer's semantics.
    assert balanced_accuracy([0, 0], [0, 1]) == 0.25
    y = np.tile([0, 1, 2], 11)
    prediction = y.copy()
    prediction[-3:] = [1, 2, 0]
    groups = np.array([0] * 30 + [1] * 3)
    assert np.isclose(group_scores(balanced_accuracy, y, prediction, groups).mean(), .5)
    assert np.isclose(balanced_accuracy(y, prediction), 10 / 11)
    model = Counter()
    loader = make_loader(np.zeros((6, 2, 3)), np.arange(6), batch_size=4,
                         stream_id=[0, 0, 0, 0, 1, 1], record_id=[0, 0, 1, 1, 2, 2],
                         onset=[0, 1, 0, 1, 0, 1])
    _, predicted, _ = predict_streams(model, loader)
    np.testing.assert_array_equal(predicted, [1, 2, 3, 4, 1, 2])
    assert model.seen == 0, "Evaluator must not mutate the submitted starting model"
    bad_loader = make_loader(np.zeros((2, 2, 3)), [0, 1],
                             stream_id=[0, 0], record_id=[0, 0], onset=[1, 0])
    try:
        predict_streams(model, bad_loader)
    except ValueError as error:
        assert "time order" in str(error)
    else:
        raise AssertionError("Out-of-order windows were accepted")


def smoke(include_eegnet=False, tier1=False):
    from track2.packaging import package, validate
    from track2.workflow import train, evaluate_run

    torch.set_num_threads(2)
    metric_and_stream_checks()
    print("PASS: official metric semantics, stream resets, and chronology checks", flush=True)
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=cache) as scratch:
        scratch = Path(scratch)
        package(scratch / "constant.zip")
        malformed = scratch / "nested.zip"
        with zipfile.ZipFile(malformed, "w") as archive:
            archive.writestr("nested/submission.py", "")
        try:
            validate(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Nested ZIP was accepted")
        for kind in (["linear", "eegnet"] if include_eegnet else ["linear"]):
            run = scratch / kind
            args = SimpleNamespace(model=kind, data="simulated", data_dir=ROOT / "data",
                                   output=run, seed=33, epochs=3 if kind == "linear" else 1,
                                   lr=None, batch_size=16, device="cpu", num_workers=0,
                                   command="train")
            train(args)
            with (run / "history.csv").open(newline="") as stream:
                history = list(csv.DictReader(stream))
            if kind == "linear":
                assert float(history[-1]["train_loss"]) < float(history[0]["train_loss"])
            expected = json.loads((run / "test_metrics.json").read_text())
            actual = evaluate_run(SimpleNamespace(run=run, data="simulated", data_dir=ROOT / "data",
                                                  device="cpu", split="test"))
            assert actual == expected, "Saved checkpoint replay changed the measured outcome"
            package(scratch / f"{kind}_synthetic_check.zip", run)
            print(f"PASS: {kind} optimization, best-checkpoint selection, export, and fresh replay", flush=True)
    if tier1:
        tier1_smoke()
    print(json.dumps({"status": "PASS", "real_eeg_tested": False,
                      "eegnet_tested": include_eegnet,
                      "tier1_tested": tier1,
                      "scope": "offline scientific-boundary and workflow checks"}, indent=2), flush=True)


def tier1_smoke():
    """Exercise all named comparisons through the real fit/reload/flat-ZIP path."""
    from track2.packaging import package
    from track2.workflow import evaluate_run, train

    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="tier1-smoke-", dir=cache) as scratch:
        scratch = Path(scratch)
        configs = [*(ROOT / "experiments").glob("*/configs/*.json"),
                   *(ROOT / "experiments/controls").glob("*.json")]
        for config in sorted(configs):
            folder = config.parent.name if config.parent.name == "controls" else config.parent.parent.name
            name = f"{folder}-{config.stem}"
            run = scratch / name
            train(SimpleNamespace(config=config, model=None, data="simulated", data_dir=ROOT / "data",
                                  output=run, seed=33, epochs=1, lr=None, batch_size=16,
                                  device="cpu", num_workers=0, protocol="screen", fold=None))
            completed = json.loads((run / "completed.json").read_text())
            for split, expected in (("val", completed["validation"]), ("test", completed["test"])):
                actual = evaluate_run(SimpleNamespace(run=run, data="simulated", data_dir=ROOT / "data",
                                                     device="cpu", split=split))
                assert actual == expected, f"{name}: saved checkpoint metric parity failed"
                assert (run / f"{split}_predictions.csv").read_bytes() == (run / "replay" / f"{split}_predictions.csv").read_bytes(), name
            package(scratch / f"{name}.zip", run)
            results.append({"preset": str(config.relative_to(ROOT)),
                            "parameters": json.loads((run / "parameters.json").read_text()),
                            "training_and_scoring_seconds": completed["seconds"]})
            print(f"PASS: {name} fit, checkpoint selection/reload, prediction replay and isolated ZIP", flush=True)
    (cache / "tier1-smoke-results.json").write_text(json.dumps({
        "real_eeg_tested": False, "presets": results}, indent=2), encoding="utf-8")
