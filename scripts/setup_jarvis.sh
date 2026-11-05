#!/usr/bin/env bash
# Run in Python 3.15; inherit only its matching CUDA torch stack.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert sys.version_info[:2] == (3, 15), "This project requires Python 3.15 and matching CUDA Torch"'
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
export PYYAML_FORCE_LIBYAML=0
python -m pip install -r requirements-training.txt
python run.py doctor
python scripts/smoke_test.py --include-eegnet
printf '\nSetup and offline checks passed. Activate with: source .venv/bin/activate\n'
