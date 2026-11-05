"""Training-only regularized OVR covariance, CSP and linear-head fitting."""

import numpy as np
from scipy.linalg import eigh
import torch

from track2.experiments import fit_head
from submission import shrinkage_covariances


def fit(model, X, y, config):
    X, y = torch.as_tensor(X), torch.as_tensor(y)
    if X.ndim != 3 or X.shape[1] != model.n_chans or not len(X) or not torch.isfinite(X).all():
        raise ValueError("CSP requires nonempty finite training windows with the expected channels")
    if y.ndim != 1 or len(y) != len(X) or not torch.equal(torch.unique(y).cpu(), torch.tensor([0, 1, 2])):
        raise ValueError("CSP fitting requires aligned labels containing command classes 0, 1, 2")
    if model.kind != "bandpower-control":
        with torch.no_grad():
            for band in range(len(model.bands)):
                covariances = shrinkage_covariances(model.filter_band(X, band)).cpu().numpy()
                covariances /= np.trace(covariances, axis1=1, axis2=2)[:, None, None]
                projections = []
                half = model.components_per_class // 2
                indices = [*range(half), *range(model.n_chans - half, model.n_chans)]
                for command in range(3):
                    selected = y.cpu().numpy() == command
                    positive = covariances[selected].mean(axis=0)
                    rest = covariances[~selected].mean(axis=0)
                    _values, vectors = eigh(positive, positive + rest, check_finite=True)
                    filters = vectors[:, indices].T
                    # Eigenvector signs are arbitrary; canonicalize stored coefficients.
                    signs = np.sign(filters[np.arange(len(filters)), np.abs(filters).argmax(axis=1)])
                    projections.extend(filters * signs[:, None])
                values = torch.as_tensor(np.asarray(projections), dtype=model.spatial_filters.dtype)
                model.spatial_filters[band].copy_(values)
            model.spatial_fitted.fill_(True)
    with torch.no_grad():
        features = model.extract_features(X)
    fit_head(model, features, y, config)
