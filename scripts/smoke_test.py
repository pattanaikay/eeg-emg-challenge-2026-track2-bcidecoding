"""Convenient standalone smoke-test entry point for an uploaded repository."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track2.smoke import smoke

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-eegnet", action="store_true")
    smoke(include_eegnet=parser.parse_args().include_eegnet)
