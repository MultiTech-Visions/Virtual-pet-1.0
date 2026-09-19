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

echo "== face/person models in $MODELS (downloaded only when missing or not the pinned version)"
mkdir -p "$MODELS"
ZOO=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
# "<sha256>  <zoo path>": the exact files this app was tuned against. Bump a hash to move to a new model version.
MODEL_PINS="
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  face_detection_yunet/face_detection_yunet_2023mar.onnx
0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  face_recognition_sface/face_recognition_sface_2021dec.onnx
47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f  person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx
9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f  pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx
"
echo "$MODEL_PINS" | while read -r sha zoo_path; do
  [ -n "$sha" ] || continue
  dest="$MODELS/$(basename "$zoo_path")"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "ok       $(basename "$dest")"
    continue
  fi
  echo "fetching $(basename "$dest")"
  curl -fsSL -o "$dest.part" "$ZOO/$zoo_path"
  got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
  if [ "$got" != "$sha" ]; then
    rm -f "$dest.part"
    echo "$(basename "$dest"): downloaded sha256 $got, expected $sha (upstream changed the file? update MODEL_PINS deliberately)" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
done
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
from festival_pet.main import YUNET_MODEL, SFACE_MODEL, VOSK_MODEL, PERSON_MODEL, POSE_MODEL
from festival_pet.hearing import NameSpotter
from festival_pet.vision import BodyFinder
from festival_pet.pose import PoseReader
from reachy_mini.motion.recorded_move import RecordedMoves, DEFAULT_EMOTIONS_DATASET
cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320))
cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
BodyFinder(PERSON_MODEL)
PoseReader(POSE_MODEL)
NameSpotter(VOSK_MODEL)
lib = RecordedMoves(DEFAULT_EMOTIONS_DATASET)
for m in ("curious1", "welcoming1", "loving1"):
    lib.get(m)
print("ok:", len(lib.list_moves()), "moves available offline")
PYEOF

echo
echo "Done. If http://reachy-mini.local:8000 shows 'Web Dashboard Deprecated' (reachy-mini >= 1.9.0), put it back:"
echo "  sudo bash $APP_DIR/scripts/restore_dashboard.sh"
echo "then open it, start 'festival_pet', or make it the startup app: see README 'Auto-start at the festival'."
