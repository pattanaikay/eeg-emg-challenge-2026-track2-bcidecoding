"""E1 additions to the shared standalone inference source."""

from math import isfinite as _e1_isfinite


class E1SpectralDecoder(FeatureDecoder):
    """Reuse shared Hann-tapered log power with an explicitly configured bank."""

    def __init__(self, meta, config):
        kind = config.get("model", "bandpower")
        if kind not in {"bandpower", "bandpower-low"}:
            raise ValueError("E1 model must be bandpower or bandpower-low")
        if config.get("head", "logistic") not in {"logistic", "lda"}:
            raise ValueError("E1 head must be logistic or lda")
        super().__init__(kind, meta["n_chans"], meta["n_classes"], meta["sfreq"])
        raw_bands = config.get("bands", self.bands)
        if not isinstance(raw_bands, (list, tuple)) or not raw_bands:
            raise ValueError("E1 bands must be a nonempty list of [low, high] pairs")
        bands = []
        for band in raw_bands:
            if not isinstance(band, (list, tuple)) or len(band) != 2:
                raise ValueError("E1 bands must contain [low, high] pairs")
            try:
                low, high = map(float, band)
            except (TypeError, ValueError) as error:
                raise ValueError("E1 band limits must be finite numbers") from error
            if not (_e1_isfinite(low) and _e1_isfinite(high)
                    and 0 <= low < high <= self.sfreq / 2):
                raise ValueError("E1 bands must satisfy 0 <= low < high <= Nyquist")
            bands.append((low, high))
        self.bands = tuple(bands)
        n_features = self.n_chans * len(self.bands)
        self.feature_mean = torch.zeros(n_features)
        self.feature_scale = torch.ones(n_features)
        self.linear = nn.Linear(n_features, meta["n_classes"])


def build_e1(meta, config):
    return E1SpectralDecoder(meta, config)
