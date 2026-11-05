"""Matched compact power decoder with Sinc or spectrally unrestricted temporal taps."""

import torch.nn.functional as e6_functional


def e6_sinc_kernels(low, high, offsets, sfreq, taper):
    lower = 2 * low[:, None] / sfreq
    upper = 2 * high[:, None] / sfreq
    return (upper * torch.sinc(upper * offsets) - lower * torch.sinc(lower * offsets)) * taper


class E6TemporalFilters(nn.Module):
    def __init__(self, kind, n_filters, kernel_length, sfreq):
        super().__init__()
        self.kind = kind
        self.sfreq = float(sfreq)
        if not 0 < self.sfreq < float("inf"):
            raise ValueError("Sinc sampling rate must be finite and positive")
        self.register_buffer("offsets", torch.arange(kernel_length, dtype=torch.float32) - kernel_length // 2)
        self.register_buffer("taper", torch.hamming_window(kernel_length, periodic=False).unsqueeze(0))
        nyquist = self.sfreq / 2
        low = torch.linspace(0.07, 0.65, n_filters) * nyquist
        high = low + 0.10 * nyquist
        if kind == "sinc":
            low_fraction = (low / nyquist - 0.005) / 0.965
            width_fraction = (high - low - 0.02 * nyquist) / (0.99 * nyquist - low - 0.02 * nyquist)
            self.raw_low = nn.Parameter(torch.logit(low_fraction))
            self.raw_width = nn.Parameter(torch.logit(width_fraction))
        elif kind == "free":
            self.taps = nn.Parameter(e6_sinc_kernels(low, high, self.offsets, self.sfreq, self.taper))
        else:
            raise ValueError("E6 temporal model must be sinc or free")

    def cutoffs(self):
        if self.kind != "sinc":
            raise ValueError("Free temporal kernels have no enforced frequency cutoffs")
        nyquist = self.sfreq / 2
        low = nyquist * (0.005 + 0.965 * self.raw_low.sigmoid())
        # A positive minimum width and upper margin remain valid even if sigmoid saturates.
        high = low + 0.02 * nyquist + (0.99 * nyquist - low - 0.02 * nyquist) * self.raw_width.sigmoid()
        return low, high

    def kernels(self):
        if self.kind == "sinc":
            low, high = self.cutoffs()
            taps = e6_sinc_kernels(low, high, self.offsets, self.sfreq, self.taper)
        else:
            taps = self.taps
        return (taps / taps.norm(dim=-1, keepdim=True).clamp_min(1e-6)).unsqueeze(1)

    def forward(self, X):
        return e6_functional.conv1d(X, self.kernels(), padding=self.offsets.numel() // 2)


class E6PowerDecoder(nn.Module):
    def __init__(self, meta, config):
        super().__init__()
        settings = {}
        for key, default in (("n_filters", 8), ("kernel_length", 31),
                             ("spatial_multiplier", 2), ("pool_bins", 4)):
            value = config.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            settings[key] = value
        if settings["kernel_length"] < 3 or settings["kernel_length"] % 2 != 1:
            raise ValueError("kernel_length must be odd and at least 3")
        if settings["kernel_length"] > meta["n_times"] or settings["pool_bins"] > meta["n_times"]:
            raise ValueError("Temporal kernel and pooling bins must fit the supplied window")
        dropout = float(config.get("dropout", 0.25))
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.n_filters = settings["n_filters"]
        n_spatial = self.n_filters * settings["spatial_multiplier"]
        self.temporal = E6TemporalFilters(config.get("model", "sinc"), self.n_filters,
                                           settings["kernel_length"], meta["sfreq"])
        self.spatial = nn.Conv2d(self.n_filters, n_spatial, (meta["n_chans"], 1),
                                  groups=self.n_filters, bias=False)
        self.normalization = nn.BatchNorm2d(n_spatial)
        self.power_pool = nn.AdaptiveAvgPool2d((1, settings["pool_bins"]))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n_spatial * settings["pool_bins"], meta["n_classes"])

    def forward(self, X):
        batch, n_chans, n_times = X.shape
        temporal = self.temporal(X.reshape(batch * n_chans, 1, n_times))
        temporal = temporal.reshape(batch, n_chans, self.n_filters, n_times).permute(0, 2, 1, 3)
        spatial = self.normalization(self.spatial(temporal))
        log_power = self.power_pool(spatial.square()).clamp_min(1e-6).log()
        return self.head(self.dropout(log_power.flatten(1)))


def build_e6(meta, config):
    return E6PowerDecoder(meta, config)
