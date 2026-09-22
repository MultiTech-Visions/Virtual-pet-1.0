"""Run the page's own code against the robot's own data.

Three page bugs in a row got shipped — a field the API stopped sending, a variable trapped inside a
card's closure, an element id that did not exist — and every one of them only showed up as a blank or
half-frozen page on the robot, because nothing here ever executed the page's JavaScript. Checking that
it *parses* was never going to catch a ReferenceError on line 626.

So: take the real `/api/mind` payload, run `refresh()` in node against a stub DOM, and require that it
draws every card without a single error. No browser, no dependencies beyond node, about a second.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "festival_pet" / "static" / "index.html"

# Enough of a browser for the drawing code: elements that remember what was set on them, a canvas that
# swallows everything, and a fetch that answers with the payloads the page asks for.
HARNESS = """
const PAYLOADS = __PAYLOADS__;
const made = {};
const el = id => (made[id] = made[id] || {
  id, textContent: '', innerHTML: '', value: '', disabled: false, src: '',
  dataset: {}, style: {}, width: 300, height: 150, clientWidth: 300,
  classList: {_s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
              toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); }, contains(c) { return this._s.has(c); }},
  getContext: () => new Proxy({}, {get: () => () => ({})}),
  closest: () => null, addEventListener() {}, appendChild() {},
});
const IDS = __IDS__;
globalThis.document = {
  getElementById: id => (IDS.includes(id) ? el(id) : null),
  querySelectorAll: () => [],
  addEventListener() {}, activeElement: null,
};
globalThis.window = {location: {protocol: 'http:', host: 'x', href: ''}, addEventListener() {}};
globalThis.location = globalThis.window.location;
globalThis.alert = () => {};
globalThis.confirm = () => false;
globalThis.prompt = () => null;
globalThis.setInterval = () => 0;
globalThis.setTimeout = () => 0;
globalThis.clearInterval = () => {};
globalThis.WebSocket = function () { return {close() {}, send() {}, readyState: 0}; };
globalThis.Terminal = function () { return {open() {}, write() {}, onData() {}, loadAddon() {}, cols: 80, rows: 24}; };
globalThis.FitAddon = {FitAddon: function () { return {fit() {}}; }};
globalThis.fetch = async (url) => {
  for (const [path, body] of Object.entries(PAYLOADS)) {
    if (url.startsWith(path)) return {ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body)};
  }
  return {ok: false, status: 404, json: async () => ({detail: 'not stubbed: ' + url})};
};

const errors = [];
globalThis.console = {...console, error: (...a) => errors.push(a.map(String).join(' '))};

__SCRIPT__

(async () => {
  tab = __TAB__;
  await refresh();
  const conn = made['conn'] || {};
  console.log(JSON.stringify({
    errors,
    conn: String(conn.textContent || '') + String(conn.innerHTML || ''),
    drew: Object.keys(made).length,
  }));
})().catch(e => { console.log(JSON.stringify({errors: ['THREW: ' + (e && e.stack || e)], conn: '', drew: 0})); });
"""


def _page_script() -> str:
    text = PAGE.read_text()
    return text.split("<script>")[-1].split("</script>")[0]


def _ids() -> list[str]:
    import re

    return sorted(set(re.findall(r'\bid="([^"]+)"', PAGE.read_text())))


def _run_page(payloads: dict, tab: str) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    js = (HARNESS
          .replace("__PAYLOADS__", json.dumps(payloads))
          .replace("__IDS__", json.dumps(_ids()))
          .replace("__TAB__", json.dumps(tab))
          .replace("__SCRIPT__", _page_script()))
    out = subprocess.run(["node", "--input-type=module", "-"], input=js, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"node failed: {out.stderr[-2000:]}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def _payloads(pet) -> dict:
    from festival_pet.main import install_routes

    app = FastAPI()
    install_routes(app, pet)
    c = TestClient(app)
    return {
        "/api/mind": c.get("/api/mind").json(),
        "/api/log": c.get("/api/log").json(),
        "/api/catalog": c.get("/api/catalog").json(),
        "/api/volume": {"volume": 50},
        "/api/bt": {"powered": True, "devices": []},
        "/api/dev": c.get("/api/dev").json(),
    }


def _busy_pet():
    """A pet with as much switched on as a bench can manage: the quiet paths are where things hide."""
    from test_web import _pet

    from festival_pet.behavior import FaceObs
    from festival_pet.pose import Arms, plur_features

    pet = _pet()
    pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (150.0, 100.0), "l_elbow": (100.0, 120.0),
           "r_elbow": (150.0, 120.0), "l_wrist": (105.0, 112.0), "r_wrist": (146.0, 110.0),
           "l_wrist_ok": True, "r_wrist_ok": True}
    arms = Arms(1000.5, 55.0, 57.0, "out", "out", 0.9, 50.0, pts)
    pet.signs.add_training("peace", plur_features(arms))
    pet.signs.add_training("hug", plur_features(arms))
    pet.set_bracelets([0, 1], 1000.5)
    pet.sing(1000.5)
    pet.control("manual_groove", True)
    for k in range(4):
        pet.control("tap", k == 0)
    pet.plur_next(1000.5)
    pet._last_obs.arms, pet._last_obs.face = arms, FaceObs(1, 5.0, 0.0, 0.05, None, 0.0)
    pet.step(1000.6)
    return pet


@pytest.mark.parametrize("tab", ["mind", "senses", "controls", "play", "people", "dev"])
def test_every_tab_draws_without_a_single_error(tab):
    pet = _busy_pet()
    try:
        result = _run_page(_payloads(pet), tab)
    finally:
        pet.stop()
    assert not result["errors"], f"{tab}: {result['errors']}"
    assert "could not draw" not in result["conn"], f"{tab}: {result['conn']}"
    assert "error" not in result["conn"].lower(), f"{tab}: {result['conn']}"
    assert result["conn"].startswith("live"), f"{tab}: {result['conn']}"


def test_a_fresh_robot_with_nothing_going_on_draws_too():
    """Nobody about, nothing taught, nothing playing: every 'or none' branch on the page."""
    from test_web import _pet

    pet = _pet()
    try:
        result = _run_page(_payloads(pet), "mind")
    finally:
        pet.stop()
    assert not result["errors"] and result["conn"].startswith("live")


def test_the_page_says_so_when_the_robot_cannot_build_its_data():
    """The degraded payload has to be shown, not silently skipped."""
    from test_web import _pet

    pet = _pet()
    try:
        payloads = _payloads(pet)
        payloads["/api/mind"] = {"error": "KeyError: 'something'", "build": {"version": "9.9.9"},
                                 "traceback": ["  File ...", "KeyError: 'something'"]}
        result = _run_page(payloads, "mind")
    finally:
        pet.stop()
    assert not result["errors"]
    assert "cannot build" in result["conn"] and "9.9.9" in result["conn"]
