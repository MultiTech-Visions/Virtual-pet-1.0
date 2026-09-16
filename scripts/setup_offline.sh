#!/usr/bin/env bash
# One-time setup, run ON THE ROBOT while it still has internet.
# Afterwards the app needs no network at all.
#
#   scp -r . pollen@reachy-mini.local:/home/pollen/festival_pet
#   ssh pollen@reachy-mini.local 'bash /home/pollen/festival_pet/scripts/setup_offline.sh'
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FESTIVAL_PET_PYTHON:-/venvs/apps_venv/bin/python}"  # override only for testing the installer
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/festival_pet"
MODELS="$DATA_DIR/models"

echo "== installing app into the daemon's apps venv"
if [ ! -x "$PY" ]; then
  echo "apps venv not found at $PY (is this the wireless unit?)" >&2; exit 1
fi
"$PY" -m pip install --upgrade --timeout 180 --retries 8 "$APP_DIR"

echo "== downloading face models to $MODELS"
mkdir -p "$MODELS"
curl -fsSL -o "$MODELS/face_detection_yunet_2023mar.onnx" \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -fsSL -o "$MODELS/face_recognition_sface_2021dec.onnx" \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
ls -la "$MODELS"

echo "== downloading the Vosk keyword-spotting model (~40 MB zip)"
if [ ! -d "$MODELS/vosk-model-small-en-us-0.15" ]; then
  curl -fsSL -o /tmp/vosk-small.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  (cd "$MODELS" && unzip -oq /tmp/vosk-small.zip) && rm -f /tmp/vosk-small.zip
fi

echo "== caching the emotions move library for offline use"
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("pollen-robotics/reachy-mini-emotions-library", repo_type="dataset")
print("cached at", p)
PYEOF

echo "== sanity check: models load, moves resolve"
"$PY" - <<'PYEOF'
import os; os.environ["HF_HUB_OFFLINE"] = "1"
import cv2
from festival_pet.main import YUNET_MODEL, SFACE_MODEL, VOSK_MODEL
from festival_pet.hearing import NameSpotter
from reachy_mini.motion.recorded_move import RecordedMoves, DEFAULT_EMOTIONS_DATASET
cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320))
cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
NameSpotter(VOSK_MODEL)
lib = RecordedMoves(DEFAULT_EMOTIONS_DATASET)
for m in ("curious1", "welcoming1", "loving1"):
    lib.get(m)
print("ok:", len(lib.list_moves()), "moves available offline")
PYEOF

echo
echo "Done. Open http://reachy-mini.local:8000 and start 'festival_pet',"
echo "or make it the startup app: see README 'Auto-start at the festival'."
