#!/usr/bin/env bash
# Build a one-file, double-click installer for the current OS. The app source is bundled inside it.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r installer/requirements.txt
rm -rf build/installer_stage && mkdir -p build/installer_stage/app
cp -r festival_pet dashboard scripts pyproject.toml README.md build/installer_stage/app/
find build/installer_stage -name "__pycache__" -type d -exec rm -rf {} + || true
SEP=":"; [[ "${OS:-}" == "Windows_NT" ]] && SEP=";"
python -m PyInstaller --noconfirm --clean --onefile --windowed \
  --name "FestivalPetInstaller" \
  --add-data "build/installer_stage/app${SEP}app" \
  installer/reachy_installer.py
echo "built: dist/FestivalPetInstaller*"
