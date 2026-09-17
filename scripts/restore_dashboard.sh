#!/usr/bin/env bash
# Put the web dashboard back on http://reachy-mini.local:8000 (removed in reachy-mini 1.9.0).
# Run ON THE ROBOT, AS ROOT (the daemon venv and its launcher are root-owned):
#
#   sudo bash /home/pollen/festival_pet/scripts/restore_dashboard.sh
#
# Installs the reachy_dashboard package into the daemon's venv, points the daemon's
# launcher at `python -m reachy_dashboard` (same daemon, dashboard mounted back on),
# and restarts the daemon. Idempotent: run it again after every daemon update, since
# an update rewrites launcher.sh.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DAEMON_PY="${DAEMON_PYTHON:-/venvs/mini_daemon/bin/python}"  # override only for testing

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo bash $0" >&2; exit 1
fi
if [ ! -x "$DAEMON_PY" ]; then
  echo "daemon venv not found at $DAEMON_PY (is this the wireless unit?)" >&2; exit 1
fi

echo "== installing reachy_dashboard into the daemon venv"
# Build from a scratch copy: setuptools writes build/ and *.egg-info next to the sources, and as root
# those would land in pollen's checkout and break the next upload's rm -rf.
BUILD_DIR="$(mktemp -d /tmp/reachy_dashboard_build.XXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT
cp -r "$APP_DIR/dashboard/." "$BUILD_DIR/"
"$DAEMON_PY" -m pip install --no-deps --upgrade "$BUILD_DIR"
"$DAEMON_PY" -c "import reachy_dashboard; from importlib.metadata import version; print('reachy_dashboard', reachy_dashboard.STATIC_DIR, '/ reachy-mini', version('reachy-mini'))"

# Located without importing reachy_mini (its import needs the whole robot stack).
LAUNCHER="$("$DAEMON_PY" -c "import importlib.util, pathlib; (loc,) = importlib.util.find_spec('reachy_mini').submodule_search_locations; print(pathlib.Path(loc) / 'daemon/app/services/wireless/launcher.sh')")"
echo "== pointing $LAUNCHER at reachy_dashboard"
if grep -q -- "-m reachy_dashboard " "$LAUNCHER"; then
  echo "already patched"
elif grep -q -- "-m reachy_mini.daemon.app.main " "$LAUNCHER"; then
  sed -i 's/-m reachy_mini\.daemon\.app\.main /-m reachy_dashboard /' "$LAUNCHER"
  echo "patched"
else
  echo "no 'python -u -m reachy_mini.daemon.app.main ...' line found in $LAUNCHER; not touching it" >&2; exit 1
fi
grep -n "python -u" "$LAUNCHER"

if [ "${RESTART_DAEMON:-1}" = "1" ]; then
  echo "== restarting the daemon"
  systemctl restart reachy-mini-daemon
  for _ in $(seq 1 40); do
    sleep 1.5
    if curl -fsS -m 3 http://127.0.0.1:8000/api/daemon/status >/dev/null 2>&1; then
      if curl -fsS -m 3 http://127.0.0.1:8000/ | grep -q "Reachy Mini dashboard"; then
        echo "dashboard is back: http://reachy-mini.local:8000"; exit 0
      fi
      echo "daemon is up but / is not serving the dashboard" >&2; exit 1
    fi
  done
  echo "daemon did not come back within 60 s: journalctl -u reachy-mini-daemon -n 50" >&2; exit 1
fi
