#!/usr/bin/env bash
# Create a virtual environment, install dev extras, and enable pre-commit hooks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install

echo "PlantDx dev environment ready. Activate with: source .venv/bin/activate"
