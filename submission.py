"""All model inference code required by the Track 2 Codabench worker.

Only this Python file enters a submission ZIP. Training code lives elsewhere,
but reuses these exact architectures. A missing manifest selects the explicitly
named constant contract check; trained manifests always require their weights.
"""

import json
from pathlib import Path

import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver


class ConstantModel:
    """Upload/streaming check: always return motor-imagery class index zero."""

    def predict(self, X):
        if X.ndim != 3:
            raise ValueError("Expected EEG windows with shape (B, C, T)")
        return torch.zeros(len(X), dtype=torch.long, device=X.device)


class TorchLinear(nn.Module):
    """Official reference architecture: channel means followed by one linear layer.

    This deliberately discards temporal structure. It provides a small supervised
    comparison, not a claim that channel means are sufficient for mental commands.
    """

    def __init__(self, n_chans, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_chans, n_classes)

    def forward(self, X):
        return self.linear(X.mean(dim=-1))

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X).argmax(dim=1)


class EEGNetModel(nn.Module):
    """Braindecode EEGNet with an optional, train-only input normalizer."""

    def __init__(self, n_chans, n_times, n_classes, sfreq, kind="eegnet"):
        super().__init__()
        from braindecode.models import EEGNet

        self.kind = kind
        if kind == "eegnet-source-zscore":
            self.register_buffer("input_mean", torch.zeros(1, n_chans, 1))
            self.register_buffer("input_scale", torch.ones(1, n_chans, 1))
        self.net = EEGNet(n_chans=n_chans, n_times=n_times,
                          n_outputs=n_classes, sfreq=sfreq)

    def forward(self, X):
        if self.kind == "eegnet-source-zscore":
            X = (X - self.input_mean) / self.input_scale
        elif self.kind == "eegnet-window-robust":
            X = robust_window_normalize(X)
        return self.net(X)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X).argmax(dim=1)


BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
BANDS_WITH_LOW = BANDS + ((1.0, 4.0),)


def robust_window_normalize(X):
    """Normalize each channel from only the samples in the current window."""
    quartiles = torch.quantile(
        X, torch.tensor([0.25, 0.5, 0.75], dtype=X.dtype, device=X.device), dim=-1)
    median = quartiles[1].unsqueeze(-1)
    scale = ((quartiles[2] - quartiles[0]).unsqueeze(-1) / 1.349).clamp_min(1e-6)
    return (X - median) / scale


def band_power_features(X, sfreq, bands=BANDS):
    """Hann-tapered log band power, ordered by channel then band."""
    if X.ndim != 3:
        raise ValueError("Expected EEG windows with shape (B, C, T)")
    n_times = X.shape[-1]
    window = torch.hann_window(n_times, periodic=False, dtype=X.dtype, device=X.device)
    centered = X - X.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(centered * window, dim=-1).abs().square()
    power = spectrum / (sfreq * window.square().sum())
    power[..., 1:-1] *= 2
    frequencies = torch.fft.rfftfreq(n_times, d=1.0 / sfreq, device=X.device)
    resolution = sfreq / n_times
    features = []
    for low, high in bands:
        bins = (frequencies >= low) & (frequencies < high)
        if not bins.any():
            raise ValueError(f"Empty spectral band {low:g}–{high:g} Hz")
        features.append(torch.log10((power[..., bins].sum(dim=-1) * resolution).clamp_min(1e-12)))
    return torch.stack(features, dim=-1).reshape(len(X), -1)


def shrinkage_covariances(X, diagonal_only=False):
    """Per-window OAS covariance with an isotropic, positive-definite floor."""
    if X.ndim != 3:
        raise ValueError("Expected EEG windows with shape (B, C, T)")
    values = X.to(torch.float64)
    values = values - values.mean(dim=-1, keepdim=True)
    n_channels, n_times = values.shape[1:]
    covariance = values @ values.transpose(-1, -2) / (n_times - 1)
    if diagonal_only:
        covariance = torch.diag_embed(covariance.diagonal(dim1=-2, dim2=-1))
    mu = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    alpha = covariance.square().mean(dim=(-2, -1))
    denominator = (n_times + 1) * (alpha - mu.square() / n_channels)
    numerator = alpha + mu.square()
    shrinkage = torch.where(denominator > 1e-30,
                            numerator / denominator.clamp_min(1e-30),
                            torch.ones_like(denominator)).clamp(max=1.0)
    identity = torch.eye(n_channels, dtype=values.dtype, device=values.device)
    covariance = ((1 - shrinkage[:, None, None]) * covariance
                  + shrinkage[:, None, None] * mu[:, None, None] * identity)
    covariance += (mu.abs() * 1e-8 + 1e-8)[:, None, None] * identity
    return covariance


class FeatureDecoder(nn.Module):
    """Fixed spectral or tangent-covariance features with a regularized linear head."""

    def __init__(self, kind, n_chans, n_classes, sfreq):
        super().__init__()
        self.kind = kind
        self.n_chans = n_chans
        self.sfreq = sfreq
        self.bands = BANDS_WITH_LOW if kind == "bandpower-low" else BANDS
        n_features = n_chans * len(self.bands) if kind.startswith("bandpower") else (
            n_chans if kind == "covariance-diagonal" else n_chans * (n_chans + 1) // 2)
        self.register_buffer("reference_cov", torch.eye(n_chans, dtype=torch.float64))
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_scale", torch.ones(n_features))
        self.linear = nn.Linear(n_features, n_classes)

    def extract_features(self, X, covariances=None):
        if self.kind.startswith("bandpower"):
            return band_power_features(X, self.sfreq, self.bands)
        covariances = shrinkage_covariances(X) if covariances is None else covariances
        if self.kind == "covariance-diagonal":
            return torch.log(covariances.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)).to(X.dtype)
        eigenvalues, eigenvectors = torch.linalg.eigh(self.reference_cov.to(X.device))
        inverse_sqrt = (eigenvectors * eigenvalues.clamp_min(1e-12).rsqrt()) @ eigenvectors.T
        whitened = inverse_sqrt @ covariances @ inverse_sqrt
        values, vectors = torch.linalg.eigh(whitened)
        log_covariance = (vectors * values.clamp_min(1e-12).log().unsqueeze(-2)) @ vectors.transpose(-1, -2)
        rows, cols = torch.triu_indices(self.n_chans, self.n_chans, device=X.device)
        scale = torch.where(rows == cols, 1.0, 2.0 ** 0.5)
        return (log_covariance[:, rows, cols] * scale).to(X.dtype)

    def forward(self, X):
        features = self.extract_features(X)
        features = (features - self.feature_mean) / self.feature_scale
        return self.linear(features)

    @torch.inference_mode()
    def predict(self, X):
        self.eval()
        return self(X).argmax(dim=1)


class Tier1Model(nn.Module):
    """Shared validated decoder with independently resettable session preprocessing."""

    def __init__(self, meta, config):
        super().__init__()
        self.input_shape = (meta["n_chans"], meta["n_times"])
        self.n_classes = meta["n_classes"]
        experiment = config["experiment"]
        if experiment == "control":
            self.decoder = build_model(config["model"], meta)
        else:
            self.decoder = globals()[f"build_{experiment[:2]}"](meta, config)
        self.normalizer = (InputNormalizer(meta, config)
                           if config.get("normalization", "none") != "none" else nn.Identity())

    def reset_state(self):
        reset = getattr(self.normalizer, "reset_state", None)
        if callable(reset):
            reset()

    def forward(self, X):
        if X.ndim != 3 or tuple(X.shape[1:]) != self.input_shape or not len(X):
            raise ValueError(f"Expected EEG windows (B, {self.input_shape[0]}, {self.input_shape[1]})")
        if not torch.isfinite(X).all():
            raise ValueError("Non-finite EEG input")
        scores = self.decoder(self.normalizer(X))
        if scores.shape != (len(X), self.n_classes) or not torch.isfinite(scores).all():
            raise ValueError("Decoder must return finite (B, 3) scores")
        return scores

    @torch.inference_mode()
    def predict_proba(self, X):
        self.eval()
        return self(X).softmax(dim=1)

    @torch.inference_mode()
    def predict(self, X):
        return self.predict_proba(X).argmax(dim=1)


def build_model(kind, meta, config=None):
    """Construct exactly the architecture identified by a shipped manifest."""
    if kind == "experiment":
        if not isinstance(config, dict):
            raise ValueError("Experiment checkpoint requires its configuration")
        return Tier1Model(meta, config)
    if kind == "constant":
        return ConstantModel()
    if kind == "linear":
        return TorchLinear(meta["n_chans"], meta["n_classes"])
    if kind in {"eegnet", "eegnet-source-zscore", "eegnet-window-robust"}:
        return EEGNetModel(meta["n_chans"], meta["n_times"],
                           meta["n_classes"], meta["sfreq"], kind=kind)
    if kind in {"bandpower", "bandpower-low", "covariance-diagonal", "covariance"}:
        return FeatureDecoder(kind, meta["n_chans"], meta["n_classes"], meta["sfreq"])
    raise ValueError(f"Unknown model kind: {kind!r}")

class Solver(CompetSolver):
    """Reconstruct the model without fitting, downloads, or submission writes."""

    name = "Track2-BCI"

    def load_model(self, meta):
        directory = Path(meta["submission_dir"])
        manifest_path = directory / "model.json"
        if not manifest_path.exists():
            return ConstantModel()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kind = manifest["model"]
        if kind == "constant":
            return ConstantModel()
        expected = manifest["input"]
        for key in ("n_chans", "n_times", "n_classes", "sfreq"):
            if expected[key] != meta[key]:
                raise ValueError(f"Checkpoint {key}={expected[key]} but worker supplies {meta[key]}")
        # An identical channel count is insufficient if channel order differs.
        if expected.get("ch_names") and list(meta.get("ch_names") or []) != expected["ch_names"]:
            raise ValueError("Checkpoint and worker channel order differ")
        model = build_model(kind, meta, manifest.get("config")).to(meta["device"])
        state = torch.load(directory / "weights.pt", map_location=meta["device"],
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        reset = getattr(model, "reset_state", None)
        if callable(reset):
            reset()
        return model.eval()
