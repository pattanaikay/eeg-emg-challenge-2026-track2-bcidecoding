"""Train, select on validation, then score once through the official evaluator.

No test labels enter optimization or checkpoint selection. Saved run artifacts
carry the inference source and environment so a ZIP cannot silently use a newer
architecture than the checkpoint that was trained.
"""

import csv
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

from track2 import ROOT


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def environment():
    versions = {}
    for name in ("torch", "numpy", "scikit-learn", "benchopt", "neuralbench", "neuralset", "neuralfetch", "neuraltrain", "braindecode", "mne", "moabb", "exca", "nemar"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "packages": versions, "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def doctor():
    """Read versions/device availability without contacting cloud accounts."""
    print(json.dumps(environment(), indent=2), flush=True)


def device_for(choice):
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a local smoke test")
    return choice


def _configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def score(model, loader, output, prefix):
    """Use unchanged upstream stream boundaries and metric semantics."""
    from benchmark_utils.metrics import balanced_accuracy, group_scores
    from benchmark_utils.streaming import predict_streams

    y, prediction, info = predict_streams(model, loader)
    if y.ndim != 1 or prediction.shape != y.shape:
        raise ValueError(f"Expected one class per window, got {y.shape}, {prediction.shape}")
    if prediction.dtype.kind not in "iu" or np.any((prediction < 0) | (prediction > 2)):
        raise ValueError("Predictions must be integer command indices 0, 1 or 2")
    context = info.get("context_id", np.zeros(len(y), dtype=np.int64))
    cells = np.stack([info["stream_id"], context], axis=1)
    scores = group_scores(balanced_accuracy, y, prediction, cells)
    result = {"balanced_accuracy_cell_mean": float(scores.mean()),
              "balanced_accuracy_session_mean": float(group_scores(balanced_accuracy, y, prediction, info["stream_id"]).mean()),
              "balanced_accuracy_pooled": balanced_accuracy(y, prediction),
              "accuracy": float(np.mean(y == prediction)), "n_windows": len(y), "n_cells": len(scores)}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / f"{prefix}_metrics.json", result)
        with (output / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["y_true", "y_pred", *info]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(len(y)):
                writer.writerow({"y_true": int(y[i]), "y_pred": int(prediction[i]),
                                 **{k: v[i].item() for k, v in info.items()}})
        keys = np.unique(cells, axis=0)
        with (output / f"{prefix}_cells.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["stream_id", "context_id", "n_windows", "balanced_accuracy"])
            for key, value in zip(keys, scores):
                writer.writerow([*key, int(np.all(cells == key, axis=1).sum()), value])
        confusion = np.zeros((3, 3), dtype=int)
        np.add.at(confusion, (y.astype(int), prediction.astype(int)), 1)
        write_json(output / f"{prefix}_confusion.json", {"rows_true_columns_predicted": confusion.tolist(),
                   "labels": ["motor_imagery", "mental_subtraction", "word_generation"]})
    return result


def train(args):
    from submission import build_model
    from track2.data import load, split_rows

    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    device = device_for(args.device)
    _configure(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    settings = {**vars(args), "epochs": args.epochs or (30 if args.model == "linear" else 20),
                "lr": args.lr if args.lr is not None else (1e-2 if args.model == "linear" else 1e-3),
                "resolved_device": device, "checkpoint_selection": "maximum validation cell-averaged balanced accuracy"}
    if settings["lr"] <= 0:
        raise ValueError("learning rate must be positive")
    write_json(output / "training.json", settings)
    write_json(output / "environment.json", environment())
    shutil.copyfile(ROOT / "UPSTREAM.json", output / "UPSTREAM.json")
    loaders, meta = load(args.data, args.data_dir, device=device, batch_size=args.batch_size,
                         seed=args.seed, num_workers=args.num_workers)
    manifest = {"model": args.model, "input": meta, "data": args.data,
                "purpose": "synthetic workflow check only" if args.data == "simulated" else "public PROTEUS supervised baseline",
                "seed": args.seed}
    write_json(output / "model.json", manifest)
    shutil.copyfile(ROOT / "submission.py", output / "submission.py")
    rows = split_rows(loaders)
    if rows:
        with (output / "split_windows.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    model = build_model(args.model, meta).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    best = -float("inf")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "val_balanced_accuracy_cell_mean"])
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for X, y, _info in loaders["train"]:
                if not torch.isfinite(X).all():
                    raise ValueError("Non-finite training EEG; inspect the data audit")
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(X), y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss; do not export a broken model")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(y)
                count += len(y)
            if not count:
                raise ValueError("Empty training loader")
            model.eval()
            validation = score(model, loaders["val"], None, "val")
            metric = validation["balanced_accuracy_cell_mean"]
            writer.writerow([epoch, loss_sum / count, metric])
            stream.flush()
            print(f"epoch={epoch}/{settings['epochs']} train_loss={loss_sum/count:.5f} val_cell_bal_acc={metric:.5f}", flush=True)
            if metric > best:
                best = metric
                torch.save(model.state_dict(), output / "weights.pt")
                write_json(output / "selection.json", {"epoch": epoch, "val_balanced_accuracy_cell_mean": best})
    model.load_state_dict(torch.load(output / "weights.pt", map_location=device, weights_only=True))
    model.eval()
    validation = score(model, loaders["val"], output, "val")
    test = score(model, loaders["test"], output, "test")
    write_json(output / "completed.json", {"seconds": time.monotonic() - started, "validation": validation, "test": test})
    print(json.dumps({"run": str(output), "test": test, "warning": manifest["purpose"]}, indent=2), flush=True)
    return output


def evaluate_run(args):
    from track2.data import load

    run = Path(args.run).resolve()
    config = json.loads((run / "training.json").read_text())
    if args.data != config["data"]:
        raise ValueError("Evaluation data kind differs from the training run")
    device = device_for(args.device)
    loaders, meta = load(args.data, args.data_dir, device=device, seed=config["seed"])
    spec = importlib.util.spec_from_file_location("saved_track2_submission", run / "submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.Solver().load_model({**meta, "submission_dir": run, "device": device})
    result = score(model, loaders[args.split], run / "replay", args.split)
    print(json.dumps(result, indent=2), flush=True)
    return result
