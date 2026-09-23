#!/usr/bin/env python3
"""petctl: drive and inspect the running pet from a terminal on the robot.

Everything the web page can do, the app already exposes over HTTP on port 8042. This is that API with
a short name on it, so a Claude Code session on the robot can look at what the pet can see and make it
do things without writing a curl line every time.

    petctl mind                        # the whole brain/senses blob (pipe into jq)
    petctl see                         # the short version: what it can see and what it is doing
    petctl watch                       # ...once a second, until you stop it
    petctl do gesture wave             # any control the page has
    petctl do sound fanfare
    petctl do kandi left
    petctl do train_plur all
    petctl camera shot.jpg             # what the camera sees, right now, with the overlays
    petctl trace out.jsonl --last 120  # the black box, last two minutes
    petctl mark "it looked at the wall"
    petctl log -n 80
    petctl controls                    # every control name, with the current value

Only the standard library: it has to run on the robot with nothing installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8042"


def _get(path: str, raw: bool = False):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        data = r.read()
    return data if raw else json.loads(data)


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read() or b"{}").get("detail", str(e))
        raise SystemExit(f"refused: {detail}")


def _value(text: str):
    """Controls take numbers, booleans, null or strings; the page sends JSON, so the CLI should too."""
    if text is None:
        return True
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def summarise(m: dict) -> str:
    """The one-line answer to 'what is it doing and can it see me'."""
    s, mind = m["senses"], m["mind"]
    face, body, arms = s["face"], s["body"], m["arms"]
    who = "no face"
    if face:
        who = f"face t{face['track']} at yaw {face['yaw']:+.0f} pitch {face['pitch']:+.0f} size {face['size']}"
        who += " (stranger)" if face["person"] is None else f" (#{face['person']})"
    elif body:
        who = f"body only at yaw {body['yaw']:+.0f}"
    bits = [
        f"{mind['state']}/{mind['activity']}",
        who,
        "arms " + ("none" if not arms else f"L{arms['left_deg']}° R{arms['right_deg']}° conf {arms['conf']}"),
        "gaze " + ("free" if not mind["gaze"] else f"{mind['gaze'][0]:+.0f},{mind['gaze'][1]:+.0f}"),
        "spots " + (", ".join(f"{x['yaw']:+.0f}°({x['faces']}f,{x['s_ago']}s)" for x in mind["seen_spots"]) or "none"),
    ]
    if mind.get("searching_spot"):
        bits.append(f"checking #{mind['searching_spot']}")
    if m["kandi"]["step"] or m["kandi"]["training"]:
        bits.append(f"plur {m['kandi']['step']}/4" + (f" teaching {m['kandi']['training']['step']}" if m["kandi"]["training"] else ""))
    v = s["vision"]
    if v:
        bits.append(f"vision {v['detect_ms']:.0f}ms" + (f" ERRORS {v['errors']}" if v["errors"] else ""))
    return " | ".join(bits)


def main(argv: list[str] | None = None) -> int:
    global BASE
    ap = argparse.ArgumentParser(prog="petctl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE, help="where the app is (default %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mind", help="the whole mind blob as JSON")
    sub.add_parser("see", help="one line: what it can see and what it is doing")
    w = sub.add_parser("watch", help="`see`, on a loop")
    w.add_argument("--every", type=float, default=1.0)
    w.add_argument("--for", dest="secs", type=float, default=0.0, help="stop after this long (0 = forever)")
    d = sub.add_parser("do", help="any control the page has")
    d.add_argument("control")
    d.add_argument("value", nargs="?")
    c = sub.add_parser("camera", help="save what the camera sees")
    c.add_argument("path", nargs="?", default="camera.jpg")
    t = sub.add_parser("trace", help="save the black box")
    t.add_argument("path", nargs="?", default="trace.jsonl")
    t.add_argument("--last", type=float, default=None, help="only the last N seconds")
    mk = sub.add_parser("mark", help="label this instant in the black box")
    mk.add_argument("text", nargs="?", default="")
    lg = sub.add_parser("log", help="the app log")
    lg.add_argument("-n", type=int, default=60)
    sub.add_parser("controls", help="every control name and its current value")
    args = ap.parse_args(argv)
    BASE = args.base.rstrip("/")

    if args.cmd == "mind":
        print(json.dumps(_get("/api/mind"), indent=1))
    elif args.cmd == "see":
        print(summarise(_get("/api/mind")))
    elif args.cmd == "watch":
        end = time.time() + args.secs if args.secs else None
        while end is None or time.time() < end:
            print(time.strftime("%H:%M:%S"), summarise(_get("/api/mind")), flush=True)
            time.sleep(args.every)
    elif args.cmd == "do":
        _post("/api/control", {"cmd": args.control, "value": _value(args.value)})
        print(f"ok: {args.control} = {_value(args.value)!r}")
    elif args.cmd == "camera":
        _post("/api/control", {"cmd": "preview", "value": True})  # the overlays are only drawn when watched
        time.sleep(0.5)
        with open(args.path, "wb") as f:
            f.write(_get("/api/camera.jpg", raw=True))
        print(f"wrote {args.path}")
    elif args.cmd == "trace":
        path = "/api/trace.jsonl" + (f"?last_s={args.last}" if args.last else "")
        data = _get(path, raw=True)
        with open(args.path, "wb") as f:
            f.write(data)
        print(f"wrote {args.path} ({len(data.splitlines())} lines, {len(data) / 1e6:.1f} MB)")
    elif args.cmd == "mark":
        _post("/api/control", {"cmd": "mark", "value": args.text or True})
        print("marked")
    elif args.cmd == "log":
        print("\n".join(_get(f"/api/log?n={args.n}")["lines"]))
    elif args.cmd == "controls":
        for k, v in sorted(_get("/api/mind")["controls"].items()):
            print(f"{k:24s} {json.dumps(v)[:110]}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach the pet at {BASE}: {e.reason}. Is the app running?")
