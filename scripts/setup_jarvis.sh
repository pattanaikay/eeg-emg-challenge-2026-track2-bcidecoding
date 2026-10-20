#!/usr/bin/env bash
# Run in a Python 3.12+ GPU template; inherit its matching CUDA torch stack.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert sys.version_info >= (3, 12), "PROTEUS stack requires Python 3.12+; select a matching Jarvis template/interpreter"'
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
python run.py doctor
python scripts/smoke_test.py --include-eegnet
printf '\nSetup and offline checks passed. Activate with: source .venv/bin/activate\n'
