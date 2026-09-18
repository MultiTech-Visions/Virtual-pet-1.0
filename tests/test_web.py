"""The pet's web API with a fully fake robot (no SDK, no sim)."""

import json
import logging
import threading
import time

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from festival_pet.behavior import Behavior, Observation
from festival_pet.main import LOG_RING, Pet, PetParts, install_routes
from festival_pet.memory import FaceMemory
from festival_pet.motion import MotionComposer


class NullIO:
    K = np.array([[800.0, 0, 640], [0, 800.0, 360], [0, 0, 1]])
    D = np.zeros(5)
    T_head_cam = np.eye(4)

    def head_pose(self): return np.eye(4)
    def imu(self): return None
    def present_antennas(self): return [-0.1745, 0.1745]
    def doa(self): return None
    def audio_chunk(self): return None
    def play(self, buf): pass
    def play_file(self, path): pass
    def set_target(self, head, antennas, body_yaw): pass
    def goto(self, head, antennas, duration): pass
    def sleep_body(self): self.slept = True
    def wake_body(self): self.woke = True
    def motor_mode(self): return "enabled"


class FakeMove:
    duration = 0.5
    sound_path = None

    def evaluate(self, t):
        return np.eye(4), np.zeros(2), 0.0


def _pet():
    mem = FaceMemory("/nonexistent/never-written.json")
    pet = Pet(PetParts(NullIO(), mem, lambda name: FakeMove(), lambda: None, None, Behavior(mem), MotionComposer()))
    pet.start(1000.0)
    pet.step(1000.1)
    return pet


def test_mind_log_catalog_and_controls():
    pet = _pet()
    logging.getLogger("festival_pet.test").addHandler(LOG_RING)
    logging.getLogger("festival_pet.test").warning("hello from the test")
    app = FastAPI()
    install_routes(app, pet)
    c = TestClient(app)

    m = c.get("/api/mind").json()
    assert m["mind"]["state"] in ("IDLE", "WAKING")
    assert m["mind"]["thoughts"] == [] or isinstance(m["mind"]["thoughts"][0]["text"], str)
    assert m["senses"]["face"] is None and "scratch" in m["senses"]
    assert "lonely_in_s" in m["mind"]

    assert any("hello from the test" in line for line in c.get("/api/log").json()["lines"])
    cat = c.get("/api/catalog").json()
    assert "giggle" in cat["sounds"] and "sneeze" in cat["gestures"]

    assert c.post("/api/control", json={"cmd": "sound", "value": "giggle"}).status_code == 200
    assert c.post("/api/control", json={"cmd": "gesture", "value": "tilt"}).status_code == 200
    assert c.post("/api/control", json={"cmd": "move", "value": "dance1"}).status_code == 200
    assert c.post("/api/control", json={"cmd": "sleep"}).status_code == 200
    assert c.get("/api/mind").json()["mind"]["state"] == "SLEEPING"
    assert not getattr(pet.p.io, "slept", False)  # nothing moved on the HTTP thread: the loop does it on its next tick
    pet.step(time.time())
    assert pet.p.io.slept and pet.asleep  # the page's sleep runs the real routine (centre, nest, motors off)
    assert c.post("/api/control", json={"cmd": "wake"}).status_code == 200
    pet.step(time.time())
    assert not pet.asleep
    assert c.post("/api/control", json={"cmd": "mute", "value": True}).json() == {"ok": True}
    assert c.get("/api/mind").json()["controls"]["voice"] == "off"
    assert c.post("/api/control", json={"cmd": "voice", "value": "loud"}).status_code == 400
    assert c.post("/api/control", json={"cmd": "voice", "value": "quiet"}).status_code == 200
    assert c.post("/api/control", json={"cmd": "scratch_onset_ratio", "value": 4.5}).status_code == 200
    assert c.get("/api/mind").json()["controls"]["scratch_onset_ratio"] == 4.5
    assert c.post("/api/control", json={"cmd": "nonsense"}).status_code == 400
    assert c.post("/api/control", json={"cmd": "gesture", "value": "moonwalk"}).status_code == 400
    pet.stop()


def test_people_routes(tmp_path):
    from festival_pet.memory import FaceMemory as FM

    mem = FM(tmp_path / "mem.json", save_interval=0.0)
    a = mem.enroll(np.ones(4), 0.0)
    b = mem.enroll(np.array([1.0, 0, 0, 0]), 1.0)
    mem.set_thumbnail(a, b"\xff\xd8\xff")
    pet = Pet(PetParts(NullIO(), mem, lambda n: FakeMove(), lambda: None, None, Behavior(mem), MotionComposer()))
    pet.start(0.0)
    app = FastAPI()
    install_routes(app, pet)
    c = TestClient(app)
    assert c.get(f"/api/people/{a.person_id}/face.jpg").status_code == 200
    assert c.get(f"/api/people/{b.person_id}/face.jpg").status_code == 404
    assert c.post("/api/people/merge", json={"keep": a.person_id, "other": b.person_id}).json()["kept"] == a.person_id
    assert c.post("/api/people/merge", json={"keep": a.person_id, "other": a.person_id}).status_code == 400
    assert c.delete(f"/api/people/{a.person_id}").status_code == 200
    assert c.delete(f"/api/people/{a.person_id}").status_code == 404
    assert c.get("/api/mind").json()["memory"]["people"] == 0
    pet.stop()


def test_calibration_refuses_when_touch_is_not_louder():
    import time as _t

    pet = _pet()
    a = pet.audio
    a.start_calibration("rub", 0.0)
    a.rub.stats.update({"rub_energy": 1e-3, "flatness": 0.5})
    for k in range(5):
        a._calibration_step(0.5 * k)  # baseline phase
    a._calibration_step(3.1)  # -> active phase
    a.rub.stats.update({"rub_energy": 1.2e-3, "flatness": 0.5})
    for k in range(5):
        a._calibration_step(3.2 + 0.5 * k)
    a._calibration_step(6.3)
    assert a.calibration_result["phase"] == "failed" and a.rub.level_ratio == 12.0

    a.start_calibration("rub", 10.0)
    a.rub.stats.update({"rub_energy": 1e-4, "flatness": 0.2})
    for k in range(5):
        a._calibration_step(10.0 + 0.5 * k)
    a._calibration_step(13.1)
    a.rub.stats.update({"rub_energy": 1e-2, "flatness": 0.6})
    for k in range(5):
        a._calibration_step(13.2 + 0.5 * k)
    a._calibration_step(16.3)
    r = a.calibration_result
    assert r["phase"] == "done" and 5 < a.rub.level_ratio < 20 and abs(a.rub.flatness_min - 0.42) < 1e-6
    pet.stop()


def test_sleep_does_not_wake_itself_from_antenna_droop():
    from festival_pet.behavior import Action

    pet = _pet()
    io = pet.p.io
    io.present = [-0.17, 0.17]
    io.present_antennas = lambda: io.present  # type: ignore[assignment]
    pet.step(1000.2)
    pet._dispatch(Action("sleep", "asked", 5), 1000.2)
    assert pet.asleep and getattr(io, "slept", False)
    io.present = [-3.05, 3.05]  # torque off: antennas fall into the sleep position
    for k in range(130):
        pet.step(1000.3 + k * 0.1)
    assert pet._touch_settle_left < 0  # re-zeroed to the resting position, no false ear tickle woke it
    assert pet.asleep
    pet.p.behavior.state = "SLEEPING"
    pet._dispatch(Action("wake", "name", 5), 1005.0)
    assert not pet.asleep and getattr(io, "woke", False)
    pet.stop()


def test_ears_are_deaf_while_the_pet_itself_makes_noise():
    import numpy as np

    pet = _pet()
    a = pet.audio
    # A loud flat burst that would normally be a head pet...
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(320) * 0.003).astype(np.float32)
    loud = (rng.standard_normal(320) * 0.5).astype(np.float32)
    # Drive the detector directly (thread-free) the way _run does.
    def feed(chunk, now):
        a.beat.push(chunk, now)
        scratched = a.scratch.push(chunk, now)
        rubbed = a.rub.push(chunk, now)
        deaf = now < a.deaf_until or now < a.own_sound_until() + 0.5
        return rubbed and not deaf
    t = 0.0
    for _ in range(150):
        feed(quiet, t); t += 0.02
    a.deaf_until = t + 2.0  # e.g. the sleep sound is playing
    fired = False
    for _ in range(80):
        fired |= feed(loud, t); t += 0.02
    assert not fired
    pet.stop()


def test_settings_persist(tmp_path):
    pet = _pet()
    pet.settings_file = tmp_path / "settings.json"
    pet.control("ears", False)
    pet.control("groove_scale", 1.5)
    pet.control("mimic_flip", False)
    assert (tmp_path / "settings.json").exists()
    pet2 = _pet()
    pet2.settings_file = tmp_path / "settings.json"
    pet2.load_settings()
    assert pet2.audio.enabled is False and pet2.groove_scale == 1.5 and pet2.p.composer.mimic_flip is False
    pet.stop(); pet2.stop()


def test_start_wakes_a_limp_robot_even_when_the_pet_thinks_it_is_awake():
    """The daemon boots asleep (motors disabled); an app launched into that must turn the motors on itself."""
    class LimpIO(NullIO):
        def motor_mode(self): return "disabled"
    mem = FaceMemory("/nonexistent/never-written.json")
    io = LimpIO()
    pet = Pet(PetParts(io, mem, lambda name: FakeMove(), lambda: None, None, Behavior(mem), MotionComposer()))
    assert not pet.asleep
    pet.start(1000.0)
    assert getattr(io, "woke", False)
    pet.stop()

    io2 = NullIO()  # already torqued: no wake move on start
    pet2 = Pet(PetParts(io2, mem, lambda name: FakeMove(), lambda: None, None, Behavior(mem), MotionComposer()))
    pet2.start(1000.0)
    assert not getattr(io2, "woke", False)
    pet2.stop()


def test_tapped_groove_drives_the_composer():
    pet = _pet()
    app = FastAPI()
    install_routes(app, pet)
    c = TestClient(app)
    assert c.post("/api/control", json={"cmd": "manual_groove", "value": True}).status_code == 200
    assert c.post("/api/control", json={"cmd": "groove_body", "value": 1.5}).status_code == 200
    assert c.post("/api/control", json={"cmd": "bpm", "value": 120}).status_code == 200  # (taps in a test all land in the same ms)
    assert c.post("/api/control", json={"cmd": "tap", "value": True}).status_code == 200  # the "1"
    m = c.get("/api/mind").json()
    assert m["controls"]["manual_groove"] and m["controls"]["groove_body"] == 1.5
    assert m["senses"]["tap"]["downbeat_known"]
    pet.step(1001.0)
    assert pet.p.composer.groove is not None and pet.p.composer.groove_phrase is not None
    assert c.post("/api/control", json={"cmd": "bpm", "value": 100}).status_code == 200
    assert c.get("/api/mind").json()["controls"]["bpm"] == 100
    assert c.post("/api/control", json={"cmd": "bpm", "value": 0}).status_code == 200
    assert not pet.tap.active
    assert c.post("/api/control", json={"cmd": "bpm", "value": 999}).status_code == 400
    pet.stop()


def test_keypad_actions_reach_the_pet_and_the_map_persists(tmp_path):
    from festival_pet.keypad import KeyEvent

    pet = _pet()
    pet.settings_file = tmp_path / "settings.json"
    app = FastAPI()
    install_routes(app, pet)
    c = TestClient(app)
    assert c.post("/api/control", json={"cmd": "keymap", "value": "F1:happy:sleep"}).status_code == 200
    assert c.post("/api/control", json={"cmd": "keymap", "value": "F1:banana:none"}).status_code == 400
    m = c.get("/api/mind").json()
    assert m["controls"]["keymap"]["F1"] == {"press": "happy", "hold": "sleep"}
    assert m["senses"]["keypad"]["devices"] == []
    # a tapped beat from the keypad lands in the tap clock with the key's own timestamp
    pet.tap.set_bpm(120)
    pet.keypad.events.put(KeyEvent("C", True, 1000.5, "test"))
    pet.step(1000.6)
    assert pet.tap.downbeat_known and pet.tap.phase(1000.5) == 0.0
    assert any(k == "key" and n == "downbeat" for _, k, n in pet.actions_log)
    assert c.post("/api/control", json={"cmd": "key", "value": "tilt_left"}).status_code == 200
    assert pet.p.composer._gesture.name == "tilt"
    # persisted and restored
    pet2 = _pet()
    pet2.settings_file = pet.settings_file
    pet2.load_settings()
    assert pet2.keymap.keys["F1"] == {"press": "happy", "hold": "sleep"}
    pet.stop(); pet2.stop()


def test_held_mode_asks_to_be_turned():
    pet = _pet()
    pet.control("pickup", True)
    assert pet.p.composer.held
    pet.p.behavior.state = "SEARCHING"  # looks where it last saw someone: way off to its left
    pet.p.behavior._last_seen_yaw, pet.p.behavior._last_seen_pitch = 100.0, 0.0
    pet.p.behavior._state_since = 1001.0
    t, pointed = 1001.0, False
    while t < 1004.0:
        pet.step(t)
        g = pet.p.composer._gesture
        pointed = pointed or (g is not None and g.name == "point" and g.side > 0)
        t += 0.02
    assert any(k == "ask" and n == "turn me left" for _, k, n in pet.actions_log)
    assert any(k == "sound" and n == "huff" for _, k, n in pet.actions_log)
    assert pointed
    pet.stop()


def test_singing_sings_bows_and_saves(tmp_path):
    from festival_pet import songs

    pet = _pet()
    pet.songs_file = tmp_path / "songs.json"
    app = FastAPI(); install_routes(app, pet); c = TestClient(app)
    assert c.post("/api/control", json={"cmd": "save_song"}).status_code == 400  # nothing sung yet
    song = pet.sing(1000.2)  # on the pet's own (fake) clock, so the step loop below is short
    assert pet.last_song is song and pet.tap.bpm == song["bpm"]
    assert c.get("/api/mind").json()["song"]["last"]["name"] == song["name"]
    t, end = 1000.2, pet._singing_until
    assert end - 1000.2 < 40
    while t < end + 0.5:
        pet.step(t); t += 0.02
    assert pet.p.composer._gesture is not None and pet.p.composer._gesture.name == "bow"
    assert c.post("/api/control", json={"cmd": "save_song"}).status_code == 200
    assert json.loads(pet.songs_file.read_text())[0]["name"] == pet.last_song["name"]
    assert c.get("/api/mind").json()["song"]["last"]["saved"]
    pet2 = _pet(); pet2.songs_file = pet.songs_file; pet2.load_songs()
    assert pet2.songs == pet.songs
    assert c.post("/api/control", json={"cmd": "singing", "value": True}).status_code == 200
    assert c.get("/api/mind").json()["song"]["next_in_s"] is not None
    pet.stop(); pet2.stop()


def test_imu_rub_reads_as_petting_and_lowers_the_antennas():
    from festival_pet.senses import ImuRubDetector

    d = ImuRubDetector()
    t = 0.0
    started = []
    for i in range(60):  # 1.2 s of hand jitter on a still robot
        started.append(d.update(0.3, False, t)); t += 0.02
    assert d.rubbing and started.count(True) == 1
    for i in range(60):  # hand gone
        d.update(0.01, False, t); t += 0.02
    assert not d.rubbing
    for i in range(60):  # the robot moving itself is not a rub
        d.update(0.3, True, t); t += 0.02
    assert not d.rubbing
    m = MotionComposer()
    m.petted = True
    for i in range(300):
        _, ants, _ = m.sample(i * 0.02, 0.02)
    assert ants[0] > 0.5 and ants[1] < -0.5  # antennas eased down


def test_the_page_script_parses():
    """A stray redeclaration once left the page stuck on "connecting": every script block must parse."""
    import re
    import shutil
    import subprocess
    from pathlib import Path

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node not installed")
    html = (Path(__file__).resolve().parents[1] / "festival_pet" / "static" / "index.html").read_text()
    for i, script in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S)):
        r = subprocess.run(["node", "--check", "-"], input=script, capture_output=True, text=True)
        assert r.returncode == 0, f"script block {i}: {r.stderr}"


def test_quiet_voice_drops_the_chatter_and_keeps_the_reactions():
    from festival_pet.behavior import Action

    pet = _pet()
    pet.voice = "quiet"
    pet.p.composer.rng.seed(1)
    kept = []
    pet.sound.request = lambda name, prio, now: kept.append((name, prio))  # type: ignore[method-assign]
    for _ in range(40):
        pet._dispatch(Action("sound", "curious", 1), 1000.0)
        pet._dispatch(Action("sound", "giggle", 2), 1000.0)
        pet._dispatch(Action("sound", "hello_new", 3), 1000.0)
    prios = [p for _, p in kept]
    assert prios.count(1) == 0 and prios.count(3) == 40 and 8 < prios.count(2) < 32
    pet.voice = "off"
    pet._dispatch(Action("sound", "hello_new", 5), 1000.0)
    assert len(kept) == 40 + prios.count(2)
    pet.stop()
