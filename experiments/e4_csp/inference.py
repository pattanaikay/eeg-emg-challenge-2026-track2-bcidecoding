"""Fixed current-window filtering and fitted one-vs-rest CSP projections."""

from math import isfinite as _e4_isfinite


def e4_filter_band(X, sfreq, band):
    """Hann-tapered FFT band signal, with the shared E1 power normalization."""
    n_times = X.shape[-1]
    if n_times < 2:
        raise ValueError("CSP requires at least two time samples")
    taper = torch.hann_window(n_times, periodic=False, dtype=X.dtype, device=X.device)
    spectrum = torch.fft.rfft((X - X.mean(dim=-1, keepdim=True)) * taper, dim=-1)
    frequencies = torch.fft.rfftfreq(n_times, d=1 / sfreq, device=X.device)
    mask = (frequencies >= band[0]) & (frequencies < band[1])
    if not mask.any():
        raise ValueError("CSP band contains no Fourier bins")
    # Parseval makes unprojected variance equal the shared E1 integrated power.
    filtered = torch.fft.irfft(spectrum * mask, n=n_times, dim=-1)
    return filtered * (n_times / taper.square().sum()).sqrt()


class E4CSPDecoder(FeatureDecoder):
    """Few regularized class-vs-rest projections, log variance and linear head."""

    def __init__(self, meta, config):
        super().__init__("bandpower", meta["n_chans"], meta["n_classes"], meta["sfreq"])
        self.kind = config.get("model", "csp")
        if self.kind not in {"csp", "fbcsp", "bandpower-control"}:
            raise ValueError("E4 model must be csp, fbcsp or bandpower-control")
        if meta["n_classes"] != 3:
            raise ValueError("E4 one-vs-rest CSP requires three command classes")
        if config.get("head", "logistic") not in {"logistic", "lda"}:
            raise ValueError("E4 head must be logistic or lda")
        raw_bands = config.get("bands", ((4, 45),) if self.kind == "csp" else BANDS)
        if not isinstance(raw_bands, (list, tuple)) or not raw_bands:
            raise ValueError("CSP bands must be nonempty [low, high] pairs")
        bands = []
        for band in raw_bands:
            if not isinstance(band, (list, tuple)) or len(band) != 2:
                raise ValueError("CSP bands must contain [low, high] pairs")
            try:
                low, high = map(float, band)
            except (TypeError, ValueError) as error:
                raise ValueError("CSP band limits must be finite numbers") from error
            if not (_e4_isfinite(low) and _e4_isfinite(high)
                    and 0 < low < high <= self.sfreq / 2):
                raise ValueError("CSP bands must satisfy 0 < low < high <= Nyquist")
            bands.append((low, high))
        self.bands = tuple(bands)
        if self.kind == "csp" and len(self.bands) != 1:
            raise ValueError("CSP uses one band; select fbcsp for multiple bands")
        if self.kind == "fbcsp" and len(self.bands) < 2:
            raise ValueError("FBCSP requires at least two bands")
        components = config.get("components_per_class", 2)
        if (not isinstance(components, int) or isinstance(components, bool)
                or components < 2 or components % 2 or components > self.n_chans):
            raise ValueError("CSP components_per_class must be an even integer between 2 and n_chans")
        self.components_per_class = components
        n_projections = 0 if self.kind == "bandpower-control" else 3 * components
        self.register_buffer("spatial_filters", torch.zeros(len(self.bands), n_projections, self.n_chans))
        self.register_buffer("spatial_fitted", torch.tensor(self.kind == "bandpower-control"))
        n_features = len(self.bands) * (self.n_chans if self.kind == "bandpower-control" else n_projections)
        self.feature_mean = torch.zeros(n_features)
        self.feature_scale = torch.ones(n_features)
        self.linear = nn.Linear(n_features, 3)

    def filter_band(self, X, index):
        return e4_filter_band(X, self.sfreq, self.bands[index])

    def extract_features(self, X, covariances=None):
        if not bool(self.spatial_fitted):
            raise RuntimeError("CSP spatial filters have not been fitted")
        features = []
        for index in range(len(self.bands)):
            filtered = self.filter_band(X, index)
            if self.kind != "bandpower-control":
                filtered = torch.einsum("kc,bct->bkt", self.spatial_filters[index].to(X.dtype), filtered)
            features.append(filtered.var(dim=-1, unbiased=False).clamp_min(1e-12).log10())
        # Projection/channel first, then band; control ordering is identical to E1.
        return torch.stack(features, dim=-1).reshape(len(X), -1)


def build_e4(meta, config):
    return E4CSPDecoder(meta, config)
