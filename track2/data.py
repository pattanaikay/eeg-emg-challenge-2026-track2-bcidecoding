"""Public-data preparation, official split loading, and a readable BIDS audit.

Do not replace the official loader with randomly split EEG windows: that would
mix participant/session conditions and answer a different transfer question.
"""

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

from track2 import ROOT


def configure_neuralbench(data_dir):
    """Avoid the first-run path wizard in detached jobs without editing home config."""
    data_dir = Path(data_dir).resolve()
    config_path = ROOT / ".cache" / "neuralbench-config.json"
    explicit = os.environ.get("NEURALBENCH_CONFIG")
    if explicit and Path(explicit).resolve() != config_path.resolve():
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"Explicit NEURALBENCH_CONFIG does not exist: {explicit}")
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"USER": "track2", "ENTITY_NAME": "", "PROJECT_NAME": "track2-bci",
              "DATA_DIR": str(data_dir), "CACHE_DIR": str(data_dir / "cache"),
              "SAVE_DIR": str(ROOT / "runs" / "neuralbench"),
              "N_CPUS": 1, "CLUSTER": None, "WANDB_HOST": ""}
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.environ["NEURALBENCH_CONFIG"] = str(config_path)


def prepare(data_dir):
    """Download PROTEUS once, then warm the pinned official extraction cache."""
    if sys.version_info < (3, 12):
        raise RuntimeError("PROTEUS/neuralbench 0.4.0 requires Python 3.12+; the offline core checks can run on 3.11")
    from benchmark_utils.nb_task import download_study

    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    configure_neuralbench(data_dir)
    download_study("eeg", "_motor_imagery_stream", data_dir)
    loaders, meta = load("proteus", data_dir, device="cpu")
    print(json.dumps({"prepared": str(data_dir), "windows": {k: len(v.dataset) for k, v in loaders.items()},
                      "input": meta}, indent=2, default=str), flush=True)
    audit(data_dir)


def load(kind, data_dir, device="cpu", batch_size=32, seed=33, num_workers=0):
    """Return train/val/test loaders while keeping metadata outside predict(X)."""
    if kind == "simulated":
        return simulated(device, batch_size, seed)
    if kind != "proteus":
        raise ValueError(kind)
    if sys.version_info < (3, 12):
        raise RuntimeError("PROTEUS/neuralbench 0.4.0 requires Python 3.12+")
    configure_neuralbench(data_dir)
    from benchmark_utils.nb_task import load_task, require_prepared
    from vendor.competition.tracks.bci_decoding.datasets.bci_studies import (
        _proteus_interface, _proteus_run,
    )

    data_dir = Path(data_dir).resolve()
    require_prepared("eeg", "_motor_imagery_stream", data_dir)
    loaders, meta = load_task(
        "eeg", "_motor_imagery_stream", data_dir=data_dir, device=device,
        batch_size=batch_size, seed=seed, num_workers=num_workers,
        target_transform=lambda y: y.argmax(-1),
        context_of=_proteus_interface, order_of=_proteus_run,
        # Serialize signal extraction without attaching infra to LabelEncoder.
        overrides={"neuro.infra.cpus_per_task": 1},
    )
    meta["n_classes"] = int(meta["raw_target_shape"][-1])
    if (meta["n_chans"], meta["n_times"], meta["n_classes"]) != (41, 480, 3):
        raise ValueError(f"PROTEUS contract changed: {meta}. Review before training.")
    check_subject_split(loaders)
    return loaders, serializable_meta(meta)


def serializable_meta(meta):
    """Keep only the model-facing fields needed to reconstruct inference."""
    return {"n_chans": int(meta["n_chans"]), "n_times": int(meta["n_times"]),
            "n_classes": int(meta["n_classes"]), "sfreq": float(meta["sfreq"]),
            "ch_names": list(meta["ch_names"]) if meta.get("ch_names") else None}


def split_rows(loaders):
    """Read identifiers from segment metadata, without reading every EEG tensor."""
    rows = []
    for split, loader in loaders.items():
        ds = loader.dataset
        if not hasattr(ds, "seg_ds"):
            continue  # synthetic data have no human participant identity
        for position in ds.index:
            segment = ds.seg_ds.segments[int(position)]
            event = segment.trigger.to_dict()
            rows.append({"split": split, "subject": str(event.get("subject", "")),
                         "session": str(event.get("session", "")),
                         "task": str(event.get("task", "")), "run": str(event.get("run", "")),
                         "onset": float(segment.start)})
    return rows


def check_subject_split(loaders):
    rows = split_rows(loaders)
    groups = {split: {r["subject"] for r in rows if r["split"] == split} for split in loaders}
    for left, subjects in groups.items():
        if "" in subjects:
            raise ValueError("Prepared windows lack participant identity for the split audit")
        for right, other in groups.items():
            if left != right and subjects & other:
                raise ValueError(f"Participant leakage between {left} and {right}: {subjects & other}")


def simulated(device="cpu", batch_size=32, seed=33):
    """Independent synthetic splits at real tensor dimensions, not EEG evidence."""
    from benchmark_utils.data import make_loader

    rng = np.random.default_rng(seed)
    templates = rng.standard_normal((3, 41)).astype(np.float32)
    loaders = {}
    for split, n, offset in (("train", 96, 0), ("val", 48, 20), ("test", 48, 40)):
        y = np.tile(np.arange(3), n // 3).astype(np.int64)
        X = rng.standard_normal((n, 41, 480)).astype(np.float32)
        X += templates[y, :, None]
        record = np.arange(n) // (n // 4) + offset
        onset = np.arange(n) % (n // 4)
        loaders[split] = make_loader(
            X, y, batch_size=batch_size, shuffle=split == "train", device=device,
            record_id=record, onset=onset, stream_id=record // 2, context_id=record % 2,
        )
    return loaders, {"n_chans": 41, "n_times": 480, "n_classes": 3,
                     "sfreq": 120.0, "ch_names": None}


def audit(data_dir):
    """Inventory public runs/events and EEG channel tables using only the stdlib."""
    data_dir = Path(data_dir).resolve()
    rows, channel_orders = [], set()
    for events in sorted(data_dir.rglob("*_events.tsv")):
        entities = dict(part.split("-", 1) for part in events.name.removesuffix("_events.tsv").split("_") if "-" in part)
        with events.open(encoding="utf-8-sig", newline="") as stream:
            table = list(csv.DictReader(stream, delimiter="\t"))
        counts = {"mi": 0, "sub": 0, "word": 0}
        for event in table:
            # PROTEUS BIDS labels are authoritative; OpenViBE annotations can
            # coexist in the EDF and must not be counted as extra trials.
            label = event.get("trial_type", event.get("value", ""))
            if label in counts:
                counts[label] += 1
        prefix = events.name.removesuffix("_events.tsv")
        channels = events.with_name(prefix + "_channels.tsv")
        names = []
        units = []
        if channels.exists():
            with channels.open(encoding="utf-8-sig", newline="") as stream:
                eeg_rows = [r for r in csv.DictReader(stream, delimiter="\t") if r.get("type", "").upper() == "EEG"]
                names = [r["name"] for r in eeg_rows]
                units = sorted({r.get("units", "unverified") for r in eeg_rows})
            channel_orders.add(tuple(names))
        sidecar = events.with_name(prefix + "_eeg.json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        edf = events.with_name(prefix + "_eeg.edf")
        rows.append({"subject": entities.get("sub", ""), "session": entities.get("ses", ""),
                     "task": entities.get("task", ""), "run": entities.get("run", ""),
                     **counts, "eeg_channels": len(names), "units": "|".join(units),
                     "recorded_sfreq_hz": metadata.get("SamplingFrequency", "unverified"),
                     "reference": metadata.get("EEGReference", "unverified"),
                     "edf_present": edf.is_file(), "edf_bytes": edf.stat().st_size if edf.is_file() else 0,
                     "events_file": str(events.relative_to(data_dir))})
    if not rows:
        raise FileNotFoundError(f"No BIDS *_events.tsv found under {data_dir}; run prepare first")
    destination = ROOT / "runs" / "data-audit"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"participants": len({r["subject"] for r in rows}),
               "participant_sessions": len({(r["subject"], r["session"]) for r in rows}),
               "runs_with_event_tables": len(rows), "distinct_eeg_channel_orders": len(channel_orders),
               "edf_files_present": sum(r["edf_present"] for r in rows),
               "recorded_sampling_rates": sorted({str(r["recorded_sfreq_hz"]) for r in rows}),
               "references": sorted({str(r["reference"]) for r in rows}),
               "units": sorted({r["units"] for r in rows}),
               "channel_orders": [list(order) for order in sorted(channel_orders)],
               "command_events": {k: sum(r[k] for r in rows) for k in ("mi", "sub", "word")},
               "expected_full_public_release": {"participants": 10, "participant_sessions": 14, "runs": 112},
               "note": "Channel orders here are raw BIDS orders; model.json records the extracted model-facing order. An inventory is not proof of correct filtering, epoching, or evaluator ordering."}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
