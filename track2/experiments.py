"""Explicit Tier 1 config dispatch, tensor heads and single-file inference export."""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

from track2 import ROOT


EXPERIMENTS = ("e1_spectral", "e2_covariance", "e3_normalization", "e4_csp",
               "e5_spatial_power", "e6_sinc")
STATISTICAL = {"e1_spectral", "e2_covariance", "e4_csp"}


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("experiment") not in (*EXPERIMENTS, "control"):
        raise ValueError("Config must select a known Tier 1 experiment or control")
    if not isinstance(config.get("model"), str):
        raise ValueError("Config must name its model variant")
    if config.get("normalization", "none") not in {
            "none", "source-zscore", "window-robust", "causal-zscore", "causal-ea"}:
        raise ValueError("Unknown normalization mode")
    if config["experiment"] == "e2_covariance" and config.get("normalization") == "causal-ea":
        raise ValueError("Do not whiten away the covariance representation in the E2 comparison")
    return config


def inference_module(config):
    """Use the very same assembled source for fitting, saved reload and ZIPs."""
    sources = [(ROOT / "submission.py").read_text(encoding="utf-8")]
    experiment = config["experiment"]
    if experiment == "e3_normalization" or config.get("normalization", "none") != "none":
        sources.append((ROOT / "experiments/e3_normalization/inference.py").read_text(encoding="utf-8"))
    if experiment not in {"control", "e3_normalization"}:
        sources.append((ROOT / "experiments" / experiment / "inference.py").read_text(encoding="utf-8"))
    source = "\n\n".join(sources)
    name = "tier1_" + hashlib.sha256(source.encode()).hexdigest()[:16]
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / "submission.py")
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    module.inference_source = source
    return module


def fit_head(model, features, y, config):
    """Fit source-only standardization and copy a three-class linear head to tensors."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(torch.as_tensor(features).detach().cpu(), dtype=np.float64)
    labels = np.asarray(torch.as_tensor(y).detach().cpu())
    if features.ndim != 2 or not len(features) or len(features) != len(labels) or not np.isfinite(features).all():
        raise ValueError("Head requires nonempty finite features aligned with labels")
    if not np.array_equal(np.unique(labels), [0, 1, 2]):
        raise ValueError("Fitting split must contain exactly command classes 0, 1, 2")
    scaler = StandardScaler().fit(features)
    head = config.get("head", "logistic")
    if head == "logistic":
        regularization = float(config.get("C", 1.0))
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("Logistic C must be finite and positive")
        estimator = LogisticRegression(C=regularization, max_iter=1000, solver="lbfgs")
    elif head == "lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(f"Unknown head: {head}")
    estimator.fit(scaler.transform(features), labels)
    if not np.array_equal(estimator.classes_, [0, 1, 2]):
        raise ValueError("Classifier command ordering changed")
    with torch.no_grad():
        for destination, values in ((model.feature_mean, scaler.mean_),
                                    (model.feature_scale, scaler.scale_),
                                    (model.linear.weight, estimator.coef_),
                                    (model.linear.bias, estimator.intercept_)):
            destination.copy_(torch.as_tensor(values, dtype=destination.dtype, device=destination.device))
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Fitting produced non-finite parameters")


def fitting_windows(loader):
    """Collect only the supplied fitting partition, sorted for causal source replay."""
    windows, labels, metadata = [], [], {}
    for X, y, info in loader:
        windows.append(X.detach().cpu())
        labels.append(y.detach().cpu())
        for key, values in info.items():
            metadata.setdefault(key, []).extend(torch.as_tensor(values).cpu().tolist())
    if not windows:
        raise ValueError("Empty fitting partition")
    X, y = torch.cat(windows), torch.cat(labels)
    if not torch.isfinite(X).all():
        raise ValueError("Non-finite fitting windows")
    info = {key: np.asarray(values) for key, values in metadata.items()}
    order = np.lexsort((info["onset"], info["record_id"], info["stream_id"]))
    return X[order], y[order], {key: values[order] for key, values in info.items()}


def fit_inputs(model, X, info, config):
    """Fit source quantities, then replay each training session from its source prior."""
    if config.get("normalization", "none") == "none":
        return X
    from experiments.e3_normalization.training import fit_normalizer

    fit_normalizer(model.normalizer, X, config)
    transformed, stream = [], None
    with torch.no_grad():
        for index, window in enumerate(X):
            if info["stream_id"][index] != stream:
                model.reset_state()
                stream = info["stream_id"][index]
            transformed.append(model.normalizer(window[None]))
    model.reset_state()
    result = torch.cat(transformed)
    if not torch.isfinite(result).all():
        raise ValueError("Normalization produced non-finite fitting data")
    return result


def fit_decoder(model, X, y, config):
    module = importlib.import_module(f"experiments.{config['experiment']}.training")
    module.fit(model.decoder, X, y, config)
