"""Fit source priors from the coordinator's fitting partition only."""

import torch


@torch.no_grad()
def fit_normalizer(normalizer, X, config):
    if X.ndim != 3 or len(X) == 0 or X.shape[1] != normalizer.n_chans or X.shape[-1] < 2:
        raise ValueError("Normalization fit requires nonempty (N, n_chans, T) training EEG")
    if not X.is_floating_point() or not torch.isfinite(X).all():
        raise ValueError("Normalization fit requires finite floating-point training EEG")
    values = X.to(device=normalizer.source_mean.device, dtype=torch.float64)
    if normalizer.mode in {"source-zscore", "causal-zscore"}:
        normalizer.source_mean.copy_(values.mean(dim=(0, 2)).unsqueeze(-1))
        normalizer.source_var.copy_(values.var(dim=(0, 2), unbiased=False).unsqueeze(-1).clamp_min(1e-12))
    elif normalizer.mode == "causal-ea":
        total = torch.zeros_like(normalizer.source_cov)
        for batch in values.split(32):
            total.add_(normalizer.covariances(batch).sum(dim=0))
        normalizer.source_cov.copy_(total / len(values))
    normalizer.reset_state()
    return normalizer
