"""E2 inference fragment; shared covariance math lives in submission.py."""


class E2DiagonalDecoder(FeatureDecoder):
    def extract_features(self, X, covariances=None):
        diagonal = shrinkage_covariances(X, diagonal_only=True)
        return super().extract_features(X, diagonal)


def build_e2(meta, config):
    kind = config.get("model", "covariance")
    if kind not in {"covariance", "covariance-diagonal"}:
        raise ValueError(f"Unknown E2 model: {kind!r}")
    if config.get("head", "logistic") not in {"logistic", "lda"}:
        raise ValueError("E2 head must be logistic or lda")
    if meta["n_classes"] != 3 or meta["n_chans"] < 1 or meta["n_times"] < 2:
        raise ValueError("E2 requires three classes, EEG channels and at least two samples")
    decoder = E2DiagonalDecoder if kind == "covariance-diagonal" else FeatureDecoder
    return decoder(kind, meta["n_chans"], meta["n_classes"], meta["sfreq"])
