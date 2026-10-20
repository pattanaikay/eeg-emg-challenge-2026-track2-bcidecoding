"""Create flat ZIPs and test the exact exported module in a fresh subprocess."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from track2 import ROOT


def package(output, run=None):
    """Package a constant reference, or a trained run with its matching source."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing submission: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if run is None:
        files = {"submission.py": ROOT / "submission.py"}
        manifest = {"model": "constant", "purpose": "warm-up upload contract check; no training"}
    else:
        run = Path(run).resolve()
        if not (run / "completed.json").is_file():
            raise ValueError("Training did not complete; inspect the run before packaging")
        files = {name: run / name for name in ("submission.py", "weights.pt", "model.json")}
        manifest = None
        for file in files.values():
            if not file.is_file():
                raise FileNotFoundError(file)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in files.items():
            archive.write(source, arcname=name)
        if manifest is not None:
            archive.writestr("model.json", json.dumps(manifest, indent=2))
    validate(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"UPLOAD FILE: {output}\nSHA256: {digest}", flush=True)
    return output


def validate(archive_path):
    """Reject nesting, unexpected executable files, missing weights and broken inference.

    Extraction accepts only the three supported filenames, not arbitrary archive
    paths. The probe runs exported source from a temporary folder, so it cannot
    accidentally import an unshipped model definition from the project.
    """
    archive_path = Path(archive_path).resolve()
    with tempfile.TemporaryDirectory(prefix="track2-validate-") as scratch:
        scratch = Path(scratch)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or not set(names) <= {"submission.py", "weights.pt", "model.json"}:
                raise ValueError("ZIP must contain unique, flat submission.py/weights.pt/model.json entries")
            if "submission.py" not in names or "model.json" not in names:
                raise ValueError("ZIP is missing submission.py or model.json")
            archive.extractall(scratch)
        environment = os.environ.copy()
        # Exclude the project package: a ZIP must not rely on unshipped modules.
        environment["PYTHONPATH"] = str(ROOT / "vendor" / "competition")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "probe_submission.py"),
                        str(scratch)], check=True, cwd=scratch, env=environment)
    return {"zip": str(archive_path), "validated": True}
