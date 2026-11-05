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
    for name in ("torch", "numpy", "scipy", "scikit-learn", "pandas", "h5py", "numcodecs", "PyYAML", "pytest", "benchopt", "neuralbench", "neuralset", "neuralfetch", "neuraltrain", "braindecode", "mne", "moabb", "exca", "nemar-py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": versions, "cuda_available": torch.cuda.is_available(),
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
    result["command_recalls"] = [float(np.mean(prediction[y == label] == label))
                                 if np.any(y == label) else None for label in range(3)]
    sessions = group_scores(balanced_accuracy, y, prediction, info["stream_id"])
    result["worst_session_balanced_accuracy"] = float(sessions.min())
    result["session_score_std"] = float(sessions.std())
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
        for grouping, column in (("sessions", "stream_id"), ("participants", "subject_id")):
            if column not in info:
                continue
            with (output / f"{prefix}_{grouping}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([column, "n_windows", "cell_balanced_accuracy", "recall_0", "recall_1", "recall_2"])
                for key in np.unique(info[column]):
                    mask = info[column] == key
                    recalls = [float(np.mean(prediction[mask & (y == label)] == label))
                               if np.any(mask & (y == label)) else "" for label in range(3)]
                    value = float(group_scores(balanced_accuracy, y[mask], prediction[mask], cells[mask]).mean())
                    writer.writerow([key, int(mask.sum()), value, *recalls])
        with (output / f"{prefix}_startup.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["stream_id", "segment", "class", "n_windows", "errors", "recall"])
            for key in np.unique(info["stream_id"]):
                indices = np.flatnonzero(info["stream_id"] == key)
                for segment, selected in (("first_window", indices[:1]), ("first_ten", indices[:10]),
                                          ("after_ten", indices[10:])):
                    for label in range(3):
                        chosen = selected[y[selected] == label]
                        errors = int(np.sum(prediction[chosen] != label))
                        writer.writerow([key, segment, label, len(chosen), errors,
                                         1 - errors / len(chosen) if len(chosen) else ""])
    return result



STATISTICAL_MODELS = {"bandpower", "bandpower-low", "covariance-diagonal", "covariance"}


def fit_source_normalizer(model, loader):
    """Fit channel mean/std from fitting windows only."""
    from torch.utils.data import DataLoader

    base = getattr(loader, "loader", loader)
    scan = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False,
                      num_workers=0, collate_fn=base.collate_fn,
                      generator=torch.Generator().manual_seed(0))
    totals = torch.zeros(model.input_mean.shape[1], dtype=torch.float64)
    squares = torch.zeros_like(totals)
    count = 0
    for X, _y, _info in scan:
        values = X.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite training EEG; inspect the data audit")
        totals += values.sum(dim=(0, 2))
        squares += values.square().sum(dim=(0, 2))
        count += values.shape[0] * values.shape[-1]
    if not count:
        raise ValueError("Cannot fit normalization from an empty training split")
    mean = totals / count
    scale = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
    model.input_mean.copy_(mean.to(model.input_mean.dtype).view(1, -1, 1))
    model.input_scale.copy_(scale.to(model.input_scale.dtype).view(1, -1, 1))


def fit_statistical_model(model, loader):
    """Fit the fixed feature scaler and L2 logistic head using training windows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    windows, labels = [], []
    for X, y, _info in loader:
        if not torch.isfinite(X).all():
            raise ValueError("Non-finite training EEG; inspect the data audit")
        windows.append(X.detach().cpu())
        labels.append(y.detach().cpu())
    if not windows:
        raise ValueError("Cannot fit a decoder from an empty training split")
    X = torch.cat(windows)
    y = torch.cat(labels).numpy()
    if not np.array_equal(np.unique(y), np.arange(model.linear.out_features)):
        raise ValueError("Training split is missing a command class")
    covariances = None
    if not model.kind.startswith("bandpower"):
        from submission import shrinkage_covariances
        covariances = shrinkage_covariances(X)
        if model.kind == "covariance":
            model.reference_cov.copy_(covariances.mean(dim=0))
    features = model.extract_features(X, covariances).numpy()
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    classifier = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(standardized, y)
    if not np.array_equal(classifier.classes_, np.arange(model.linear.out_features)):
        raise ValueError("Logistic head class order differs from command indices")
    with torch.no_grad():
        model.feature_mean.copy_(torch.as_tensor(scaler.mean_, dtype=model.feature_mean.dtype))
        model.feature_scale.copy_(torch.as_tensor(scaler.scale_, dtype=model.feature_scale.dtype))
        model.linear.weight.copy_(torch.as_tensor(classifier.coef_, dtype=model.linear.weight.dtype))
        model.linear.bias.copy_(torch.as_tensor(classifier.intercept_, dtype=model.linear.bias.dtype))

def train(args):
    from submission import build_model
    from track2.data import load, split_rows
    from track2.experiments import (STATISTICAL, fit_decoder, fit_inputs, fitting_windows,
                                    inference_module, load_config)

    config = load_config(args.config) if getattr(args, "config", None) else None
    if config is None and getattr(args, "protocol", "screen") != "screen":
        raise ValueError("--protocol outer requires --config; legacy --model commands keep their original split")
    if config is not None:
        args.model = "experiment"
    protocol = getattr(args, "protocol", "screen") if config is not None else None
    fold = getattr(args, "fold", None)
    if protocol == "outer" and fold not in range(4):
        raise ValueError("Outer protocol requires --fold 0, 1, 2 or 3")
    if protocol != "outer" and fold is not None:
        raise ValueError("--fold requires an experiment config with --protocol outer")

    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    device = device_for(args.device)
    _configure(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    statistical = (config["experiment"] in STATISTICAL if config else args.model in STATISTICAL_MODELS)
    linear_control = args.model == "linear" or (config and config["model"] == "linear")
    epochs = 0 if statistical else (
        args.epochs if args.epochs is not None else (30 if linear_control else 20))
    lr = None if statistical else (
        args.lr if args.lr is not None else (1e-2 if linear_control else 1e-3))
    if lr is not None and lr <= 0:
        raise ValueError("learning rate must be positive")
    settings = {**vars(args), "epochs": epochs, "lr": lr,
                "resolved_device": device,
                "checkpoint_selection": "maximum validation cell-averaged balanced accuracy",
                "experiment_config": config}
    write_json(output / "training.json", settings)
    write_json(output / "environment.json", environment())
    shutil.copyfile(ROOT / "UPSTREAM.json", output / "UPSTREAM.json")
    reserve = config is not None or bool(getattr(args, "reserve_p10_later", False))
    loaders, meta = load(args.data, args.data_dir, device=device, batch_size=args.batch_size,
                         seed=args.seed, num_workers=args.num_workers,
                         reserve_p10_later=reserve, protocol=protocol, fold=fold)
    manifest = {"model": args.model, "input": meta, "data": args.data,
                "purpose": "synthetic workflow check only" if args.data == "simulated" else "public PROTEUS supervised experiment",
                "seed": args.seed,
                "fit_split": "train", "config": config,
                "split_policy": "reserve P10 sessions 2-3" if reserve else "official participant split"}
    write_json(output / "model.json", manifest)
    module = inference_module(config) if config is not None else None
    if module is not None:
        (output / "submission.py").write_text(module.inference_source, encoding="utf-8")
    else:
        shutil.copyfile(ROOT / "submission.py", output / "submission.py")
    rows = split_rows(loaders)
    if rows:
        with (output / "split_windows.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if config is not None:
        from track2.splits import description
        write_json(output / "split.json", description(loaders, protocol, fold))
    model = (module.build_model(args.model, meta, config) if module else build_model(args.model, meta))
    training_loader = loaders["train"]
    if config is not None:
        from torch.utils.data import DataLoader, TensorDataset
        X, y, info = fitting_windows(loaders["train"])
        transformed = fit_inputs(model, X, info, config)
        if statistical:
            fit_decoder(model, transformed, y, config)
        else:
            training_loader = DataLoader(TensorDataset(transformed, y, torch.arange(len(y))),
                                         batch_size=args.batch_size, shuffle=True)
        write_json(output / "fit_membership.json", {
            "split": "train", "n_windows": len(y),
            "normalization": config.get("normalization", "none"),
            "decoder_fitter": f"experiments.{config['experiment']}.training.fit" if statistical else "shared gradient trainer",
            "membership_file": "split_windows.csv" if rows else "synthetic independent training partition",
            "validation_used_for": "checkpoint selection only; fixed statistical configs have no tuning"})
    elif statistical:
        fit_statistical_model(model, loaders["train"])
    elif args.model == "eegnet-source-zscore":
        fit_source_normalizer(model, loaders["train"])

    model.to(device)
    write_json(output / "parameters.json", {
        "total": sum(value.numel() for value in model.parameters()),
        "trainable": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "persistent_buffer_scalars": sum(value.numel() for name, value in model.state_dict().items()
                                         if name not in dict(model.named_parameters())),
        "note": "Tensor parameter counts; fitted statistical transforms are recorded as buffers."})

    if not statistical:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        best = -float("inf")
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "train_loss", "val_balanced_accuracy_cell_mean"])
            for epoch in range(1, epochs + 1):
                model.train()
                loss_sum, count = 0.0, 0
                for X, y, _info in training_loader:
                    X, y = X.to(device), y.to(device)
                    if not torch.isfinite(X).all():
                        raise ValueError("Non-finite training EEG; inspect the data audit")
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model.decoder(X) if config is not None else model(X), y)
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
                print(f"epoch={epoch}/{epochs} train_loss={loss_sum/count:.5f} val_cell_bal_acc={metric:.5f}", flush=True)
                if metric > best:
                    best = metric
                    torch.save(model.state_dict(), output / "weights.pt")
                    write_json(output / "selection.json", {"epoch": epoch, "val_balanced_accuracy_cell_mean": best})
    else:
        torch.save(model.state_dict(), output / "weights.pt")

    model.load_state_dict(torch.load(output / "weights.pt", map_location=device, weights_only=True))
    model.eval()
    validation = score(model, loaders["val"], output, "val")
    if statistical:
        write_json(output / "selection.json", {
            "epoch": 0, "selection": "fixed statistical configuration; no validation tuning",
            "val_balanced_accuracy_cell_mean": validation["balanced_accuracy_cell_mean"]})
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "train_loss", "val_balanced_accuracy_cell_mean"])
            writer.writerow([0, "", validation["balanced_accuracy_cell_mean"]])
    test = score(model, loaders["test"], output, "test")
    completed = {"seconds": time.monotonic() - started,
                 "validation": validation, "test": test}
    if "longitudinal" in loaders:
        completed["longitudinal"] = score(model, loaders["longitudinal"], output, "longitudinal")
    if config is not None:
        completed["longitudinal_known_person"] = json.loads((output / "split.json").read_text())["longitudinal_known_person"]
    write_json(output / "completed.json", completed)
    print(json.dumps({"run": str(output), **completed, "warning": manifest["purpose"]}, indent=2), flush=True)
    return output

def evaluate_run(args):
    from track2.data import load

    run = Path(args.run).resolve()
    config = json.loads((run / "training.json").read_text())
    if args.data != config["data"]:
        raise ValueError("Evaluation data kind differs from the training run")
    device = device_for(args.device)
    loaders, meta = load(args.data, args.data_dir, device=device, seed=config["seed"],
                         reserve_p10_later=bool(config.get("reserve_p10_later", False)) or bool(config.get("experiment_config")),
                         protocol=config.get("protocol", "screen") if config.get("experiment_config") else None,
                         fold=config.get("fold"))
    if args.split not in loaders:
        raise ValueError(f"Split {args.split!r} is unavailable for this run; choose from {list(loaders)}")
    spec = importlib.util.spec_from_file_location("saved_track2_submission", run / "submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.Solver().load_model({**meta, "submission_dir": run, "device": device})
    result = score(model, loaders[args.split], run / "replay", args.split)
    print(json.dumps(result, indent=2), flush=True)
    return result


def experiment_batch(args):
    """Run one fixed Tier 1 phase and exact saved-checkpoint replays."""
    from types import SimpleNamespace

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / args.phase
    output.mkdir(exist_ok=False)
    phases = {
        "cpu": ("linear", "bandpower", "bandpower-low",
                "covariance-diagonal", "covariance"),
        "neural": ("eegnet", "eegnet-source-zscore", "eegnet-window-robust"),
    }
    kinds = phases[args.phase]
    device = "cpu" if args.phase == "cpu" else args.device
    results = {}
    for kind in kinds:
        run = output / kind
        fit_epochs = args.epochs if kind == "linear" or kind.startswith("eegnet") else None
        fit_lr = args.lr if kind == "linear" or kind.startswith("eegnet") else None
        run_args = SimpleNamespace(
            command="train", model=kind, data="proteus", data_dir=args.data_dir,
            output=run, seed=args.seed, epochs=fit_epochs, lr=fit_lr,
            batch_size=args.batch_size, device=device,
            num_workers=args.num_workers, reserve_p10_later=True)
        train(run_args)
        completed = json.loads((run / "completed.json").read_text(encoding="utf-8"))
        for split in ("val", "test", "longitudinal"):
            replay = evaluate_run(SimpleNamespace(
                run=run, data="proteus", data_dir=args.data_dir,
                device=device, split=split))
            if replay != completed[split]:
                raise RuntimeError(f"{kind} {split} checkpoint replay changed its metrics")
        results[kind] = completed
    write_json(output / "batch_results.json", results)
    print(json.dumps({"batch": str(output), "phase": args.phase, "models": list(kinds)},
                     indent=2), flush=True)
    return output
