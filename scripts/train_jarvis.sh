#!/usr/bin/env bash
# Explicit run folders prevent accidental overwrites or implicit warm starts.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
model="${1:-linear}"
seed="${2:-33}"
case "$model" in linear|eegnet) ;; *) printf 'Model must be linear or eegnet\n' >&2; exit 2;; esac
run="runs/${model}-seed${seed}-$(date -u +%Y%m%dT%H%M%SZ)"
python run.py prepare --data-dir data
python run.py train --model "$model" --seed "$seed" --device cuda --output "$run"
python run.py evaluate --run "$run" --device cuda
python run.py package --run "$run" --output "artifacts/$(basename "$run").zip"
printf '\nCompleted run: %s\n' "$run"
