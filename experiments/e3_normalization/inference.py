"""Independent input-normalization choices; appended to standalone submission.py."""


class InputNormalizer(nn.Module):
    """Source-fitted priors plus optional current/past-window session statistics."""

    MODES = {"none", "source-zscore", "window-robust", "causal-zscore", "causal-ea"}

    def __init__(self, meta, config):
        super().__init__()
        self.mode = config.get("normalization", "none")
        if self.mode not in self.MODES:
            raise ValueError(f"Unknown normalization mode: {self.mode!r}")
        self.n_chans = int(meta["n_chans"])
        self.prior_windows = float(config.get("prior_windows", 10))
        if not 0 < self.prior_windows < float("inf"):
            raise ValueError("prior_windows must be finite and positive")
        self.register_buffer("source_mean", torch.zeros(self.n_chans, 1, dtype=torch.float64))
        self.register_buffer("source_var", torch.ones(self.n_chans, 1, dtype=torch.float64))
        self.register_buffer("source_cov", torch.eye(self.n_chans, dtype=torch.float64))
        self.register_buffer("running_mean", self.source_mean.clone(), persistent=False)
        self.register_buffer("running_var", self.source_var.clone(), persistent=False)
        self.register_buffer("running_cov", self.source_cov.clone(), persistent=False)
        self.register_buffer("running_count", torch.tensor(self.prior_windows, dtype=torch.float64),
                             persistent=False)

    @torch.no_grad()
    def reset_state(self):
        self.running_mean.copy_(self.source_mean)
        self.running_var.copy_(self.source_var)
        self.running_cov.copy_(self.source_cov)
        self.running_count.fill_(self.prior_windows)

    @staticmethod
    def covariances(X):
        return shrinkage_covariances(X)

    @torch.no_grad()
    def forward(self, X):
        if X.ndim != 3 or X.shape[1] != self.n_chans or X.shape[-1] < 2:
            raise ValueError("Normalizer expects (B, n_chans, T) with at least two samples")
        if not X.is_floating_point() or not torch.isfinite(X).all():
            raise ValueError("Normalizer expects finite floating-point EEG")
        if self.mode == "none":
            return X
        if self.mode == "window-robust":
            return robust_window_normalize(X)
        values = X.to(torch.float64)
        if self.mode == "source-zscore":
            result = (values - self.source_mean) / self.source_var.clamp_min(1e-12).sqrt()
            return result.to(X.dtype)
        transformed = []
        for window in values:
            count = self.running_count + 1
            if self.mode == "causal-zscore":
                mean = window.mean(dim=-1, keepdim=True)
                variance = window.var(dim=-1, keepdim=True, unbiased=False)
                delta = mean - self.running_mean
                # Merge equal-duration window moments without subtracting large squared means.
                self.running_var.copy_((self.running_count * self.running_var + variance
                                        + delta.square() * self.running_count / count) / count)
                self.running_mean.add_(delta / count)
                result = (window - self.running_mean) / self.running_var.clamp_min(1e-12).sqrt()
            else:
                covariance = self.covariances(window.unsqueeze(0))[0]
                self.running_cov.add_((covariance - self.running_cov) / count)
                symmetric = (self.running_cov + self.running_cov.T) / 2
                eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
                floor = (eigenvalues.abs().max() * 1e-8).clamp_min(1e-8)
                inverse_sqrt = (eigenvectors * eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.T
                result = inverse_sqrt @ (window - window.mean(dim=-1, keepdim=True))
            self.running_count.copy_(count)
            transformed.append(result)
        return torch.stack(transformed).to(X.dtype) if transformed else X.clone()


def build_e3(meta, config):
    kind = config.get("model", "eegnet")
    if kind == "linear":
        return TorchLinear(meta["n_chans"], meta["n_classes"])
    if kind != "eegnet":
        raise ValueError("E3 model must be eegnet or the linear smoke/control decoder")
    return EEGNetModel(meta["n_chans"], meta["n_times"], meta["n_classes"], meta["sfreq"])
