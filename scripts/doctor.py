#!/usr/bin/env python3
"""doctor: why is the pet's page not working?

Answers, in order, the questions you cannot answer from a browser showing a blank page:

    is the app running at all?        (/api/health)
    which build is it running?        installed version vs. this checkout — they are NOT the same thing
    what is /api/mind actually doing? the real exception, not "Internal Server Error"
    what does the app log say?

Run it on the robot (SSH, or the Dev tab):  python3 scripts/doctor.py
Only the standard library.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8042"
REPO = Path(__file__).resolve().parents[1]


def get(path: str, timeout: float = 20.0) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body.decode("utf8", "replace")
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def checkout_version() -> str:
    for line in (REPO / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    return "?"


def installed_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("festival_pet")
    except Exception:  # noqa: BLE001 - PackageNotFoundError, or no metadata at all
        return "not installed"


def main() -> int:
    print(f"checkout at {REPO}")
    here, there = checkout_version(), installed_version()
    print(f"  version in this checkout : {here}")
    print(f"  version the robot RUNS   : {there}")
    if here != there:
        print("  ^^ THESE DO NOT MATCH. The app runs the INSTALLED package, not the files you just pulled.")
        print(f"     Install this checkout and restart the app from the dashboard:")
        print(f"       cd {REPO} && pip install -e .")
    try:
        head = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        print(f"  checkout is at           : {head}")
    except (OSError, subprocess.SubprocessError):
        pass

    print(f"\napp on {BASE}")
    code, health = get("/api/health", timeout=5.0)
    if code != 200:
        print(f"  NOT ANSWERING ({code or 'no connection'}: {health})")
        print("  The app is not running, or it is an old build with no /api/health.")
        print("  Start it from the dashboard, then run this again.")
    else:
        print(f"  alive, build {health['build']}, loop last ran {health['loop_age_s']} s ago"
              + (" (ASLEEP)" if health.get("asleep") else ""))

    print("\nthe page's data (/api/mind)")
    code, mind = get("/api/mind")
    if code == 200:
        print("  ok — the page should be working. If it is not, it is the browser: hard-refresh it.")
    else:
        print(f"  FAILING with {code}")
        if isinstance(mind, dict):
            print(f"  {mind.get('detail')}")
            for line in mind.get("traceback", [])[-14:]:
                print("    " + line)
            print("\n  ^^ send the lines above to Claude; that is the whole answer.")
        else:
            print(f"  {mind}")
            print("  (no traceback in the body: this is an old build — install this checkout, see above)")

    print("\nlast of the app log")
    code, log = get("/api/log?n=25")
    lines = log.get("lines", []) if isinstance(log, dict) else []
    print("\n".join("  " + x for x in lines) if lines else "  (empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
