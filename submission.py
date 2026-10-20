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
    """Braindecode's EEGNet, preserving sample order and channel identity."""

    def __init__(self, n_chans, n_times, n_classes, sfreq):
        super().__init__()
        from braindecode.models import EEGNet

        self.net = EEGNet(n_chans=n_chans, n_times=n_times,
                          n_outputs=n_classes, sfreq=sfreq)

    def forward(self, X):
        return self.net(X)

    @torch.inference_mode()
    def predict(self, X):
        # eval() freezes BatchNorm state; the initial baseline performs no TTA.
        self.eval()
        return self(X).argmax(dim=1)


def build_model(kind, meta):
    """Construct exactly the architecture identified by a shipped manifest."""
    if kind == "constant":
        return ConstantModel()
    if kind == "linear":
        return TorchLinear(meta["n_chans"], meta["n_classes"])
    if kind == "eegnet":
        return EEGNetModel(meta["n_chans"], meta["n_times"],
                           meta["n_classes"], meta["sfreq"])
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
        model = build_model(kind, meta).to(meta["device"])
        state = torch.load(directory / "weights.pt", map_location=meta["device"],
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.eval()
