"""Small maintained spatial/power decoders; appended to submission.py."""


def build_e5(meta, config):
    from braindecode.models import FBCNet, ShallowFBCSPNet

    kind = config.get("model", "shallowconvnet")
    if kind not in {"shallowconvnet", "fbcnet"}:
        raise ValueError(f"Unknown E5 model: {kind!r}")
    if meta["n_classes"] != 3 or meta["n_chans"] < 1 or meta["sfreq"] <= 0:
        raise ValueError("E5 requires three classes and positive channel/sample rate metadata")
    arguments = {"n_chans": meta["n_chans"], "n_times": meta["n_times"],
                 "n_outputs": meta["n_classes"], "sfreq": meta["sfreq"]}
    widths = {"n_filters_spat": config.get("n_filters_spat", 8 if kind == "shallowconvnet" else 4)}
    if kind == "shallowconvnet":
        widths["n_filters_time"] = config.get("n_filters_time", 8)
    else:
        widths["stride_factor"] = config.get("stride_factor", 4)
    for key, value in widths.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"E5 {key} must be a positive integer")
    if kind == "shallowconvnet":
        dropout = float(config.get("drop_prob", 0.5))
        if not 0 <= dropout < 1 or meta["n_times"] < 99:
            raise ValueError("E5 ShallowConvNet needs 0 <= drop_prob < 1 and at least 99 samples")
        return ShallowFBCSPNet(**arguments, **widths, filter_time_length=25,
                              pool_time_length=75, pool_time_stride=15,
                              final_conv_length="auto", drop_prob=dropout)
    bands = config.get("bands", BANDS)
    if not isinstance(bands, (list, tuple)) or not bands:
        raise ValueError("E5 FBCNet requires a nonempty list of band pairs")
    checked = []
    for band in bands:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError("E5 bands must contain lower/upper pairs")
        low, high = map(float, band)
        if not 0 < low < high < meta["sfreq"] / 2:
            raise ValueError("E5 bands must satisfy 0 < lower < upper < Nyquist")
        checked.append((low, high))
    if meta["n_times"] < widths["stride_factor"] * 2:
        raise ValueError("E5 FBCNet requires at least two samples in each variance segment")
    return FBCNet(**arguments, **widths, n_bands=checked, temporal_layer="LogVarLayer",
                  filter_parameters={"method": "fir", "phase": "zero"})
