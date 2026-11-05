"""Training-only source reference and regularized classifier fitting for E2."""

import torch

from submission import shrinkage_covariances
from track2.experiments import fit_head


def fit(model, X, y, config):
    if (X.ndim != 3 or not len(X) or X.shape[1] != model.n_chans
            or X.shape[-1] < 2 or not torch.isfinite(X).all()):
        raise ValueError("E2 fitting needs nonempty, finite EEG windows with matching channels")
    if model.kind != config.get("model", "covariance"):
        raise ValueError("E2 fitting config differs from the constructed decoder")
    with torch.no_grad():
        covariances = None
        if model.kind == "covariance":
            covariances = shrinkage_covariances(X)
            model.reference_cov.copy_(covariances.mean(dim=0))
        features = model.extract_features(X, covariances)
    fit_head(model, features, y, config)
