"""Training-only feature scaling and regularized E1 head fitting."""

import torch

from track2.experiments import fit_head


def fit(model, X, y, config):
    with torch.no_grad():
        features = model.extract_features(X)
    fit_head(model, features, y, config)
