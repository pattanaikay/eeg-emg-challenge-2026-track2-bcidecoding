"""Development tools; inference remains entirely in the root submission.py."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "competition"
sys.path.insert(0, str(UPSTREAM))
# Benchopt subprocesses need the same pinned helpers as the parent Python process.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(UPSTREAM), str(ROOT), os.environ.get("PYTHONPATH", "")]
)
