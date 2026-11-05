"""Fixed participant protocols; reserve P10 later visits before any fitting."""

import numpy as np
from torch.utils.data import DataLoader, Dataset


OUTER_PAIRS = (("P01", "P02"), ("P03", "P05"), ("P06", "P07"), ("P08", "P10"))


def participant(value):
    return str(value).rsplit("/", 1)[-1].removeprefix("sub-")


def number(value):
    return int(str(value).rsplit("-", 1)[-1])


def membership(rows, protocol="screen", fold=None):
    """Metadata-only assignment, independent of EEG, labels and initialization seed."""
    if protocol not in {"screen", "outer"}:
        raise ValueError("Protocol must be screen or outer")
    if protocol == "outer" and fold not in range(4):
        raise ValueError("Outer protocol requires fold 0, 1, 2 or 3")
    if protocol == "screen" and fold is not None:
        raise ValueError("Screen protocol has no fold")
    assigned = []
    for row in rows:
        subject, session = participant(row["subject"]), number(row["session"])
        if subject == "P10" and session in {2, 3}:
            split = "longitudinal"
        elif protocol == "screen":
            if subject in {"P05", "P08"}:
                split = "val"
            elif subject in {"P04", "P09"}:
                split = "test"
            elif subject in {name for pair in OUTER_PAIRS for name in pair}:
                split = "train"
            else:
                raise ValueError(f"Unexpected development participant {subject}")
        elif subject in {"P04", "P09"}:
            split = "diagnostic"
        elif subject in OUTER_PAIRS[fold]:
            split = "test"
        elif subject in OUTER_PAIRS[(fold + 1) % 4]:
            split = "val"
        elif subject in {name for pair in OUTER_PAIRS for name in pair}:
            split = "train"
        else:
            raise ValueError(f"Unexpected development participant {subject}")
        assigned.append({**row, "split": split})
    seen, groups = set(), {}
    for row in assigned:
        identity = tuple(row[key] for key in ("subject", "session", "task", "run", "onset"))
        if identity in seen:
            raise ValueError("A window occurs more than once across partitions")
        seen.add(identity)
        groups.setdefault(row["split"], set()).add(participant(row["subject"]))
    for left in ("train", "val", "test", "diagnostic"):
        for right in ("train", "val", "test", "diagnostic"):
            if left != right and groups.get(left, set()) & groups.get(right, set()):
                raise ValueError(f"Participant leakage between {left} and {right}")
    return assigned


class PartitionDataset(Dataset):
    def __init__(self, entries, rows, identities):
        self.entries = entries
        self.rows = rows
        self.identities = identities

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        dataset, position = self.entries[index]
        X, y, _ = dataset[position]
        row, ids = self.rows[index], self.identities[index]
        return X, y, {**ids, "onset": float(row["onset"])}


def partition(loaders, protocol="screen", fold=None, num_workers=0):
    from benchmark_utils.nb_task import _DeviceLoader
    from vendor.competition.tracks.bci_decoding.datasets.bci_studies import _proteus_interface

    entries, rows = [], []
    for loader in loaders.values():
        dataset = loader.dataset
        for position, segment_position in enumerate(dataset.index):
            segment = dataset.seg_ds.segments[int(segment_position)]
            trigger = segment.trigger.to_dict()
            entries.append((dataset, position))
            rows.append({key: str(trigger.get(key, "")) for key in ("subject", "session", "task", "run")}
                        | {"onset": float(segment.start)})
    assigned = membership(rows, protocol, fold)
    counts = {key: sum(row["split"] == key for row in assigned)
              for key in ("train", "val", "test", "longitudinal")}
    if counts["longitudinal"] != 385:
        raise ValueError(f"Expected 385 reserved P10 windows; found {counts['longitudinal']}")
    if protocol == "screen" and tuple(counts[key] for key in ("train", "val", "test")) != (1136, 379, 755):
        raise ValueError(f"Unexpected prospective screen allocation: {counts}")
    order = sorted(range(len(rows)), key=lambda i: (
        rows[i]["subject"], number(rows[i]["session"]), number(rows[i]["run"]),
        rows[i]["task"], rows[i]["onset"]))
    subjects, sessions, records = {}, {}, {}
    grouped = {}
    for index in order:
        row = assigned[index]
        subject, session = row["subject"], row["session"]
        record = (subject, session, row["task"], row["run"])
        ids = {"subject_id": subjects.setdefault(subject, len(subjects)),
               "stream_id": sessions.setdefault((subject, session), len(sessions)),
               "record_id": records.setdefault(record, len(records)),
               "context_id": 0 if _proteus_interface(row) == "Graz" else 1}
        grouped.setdefault(row["split"], []).append((entries[index], row, ids))
    source = loaders["train"]
    result = {}
    for split, values in grouped.items():
        dataset = PartitionDataset([value[0] for value in values],
                                   [value[1] for value in values], [value[2] for value in values])
        result[split] = _DeviceLoader(DataLoader(
            dataset, batch_size=source.batch_size, shuffle=split == "train",
            num_workers=num_workers, pin_memory=source.pin_memory,
            persistent_workers=num_workers > 0), source.device)
    if not all(key in result for key in ("train", "val", "test")):
        raise ValueError("Protocol produced an empty fitting, validation or evaluation partition")
    return result


def description(loaders, protocol, fold):
    from track2.data import split_rows

    rows = split_rows(loaders)
    participants = {split: sorted({participant(row["subject"]) for row in rows if row["split"] == split})
                    for split in loaders}
    return {"protocol": protocol, "fold": fold, "participants": participants,
            "windows": {split: len(loader.dataset) for split, loader in loaders.items()},
            "longitudinal_known_person": "P10" in participants.get("train", []),
            "previously_viewed_participants": ["P04", "P09"],
            "selection": "inner validation only; outer/test/longitudinal results are diagnostics"}
