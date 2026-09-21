"""The pet's web API with a fully fake robot (no SDK, no sim)."""

import json
import logging
import math
import threading
import time

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from festival_pet.behavior import Behavior, Observation
from festival_pet.main import LOG_RING, Pet, PetParts, install_routes
from festival_pet.memory import FaceMemory
from festival_pet.motion import MotionComposer


def _euler(pose):
    from scipy.spatial.transform import Rotation as R

    return R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)


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


def _pet(t0: float = 1000.0):
    """``t0``: the pet's clock. Controls from the page use the wall clock, so a test that steps the pet after
    a page control (a game start, a flourish timer) starts it at time.time() and steps from there."""
    mem = FaceMemory("/nonexistent/never-written.json")
    pet = Pet(PetParts(NullIO(), mem, lambda name: FakeMove(), lambda: None, None, Behavior(mem), MotionComposer()))
    pet.start(t0)
    pet.step(t0 + 0.1)
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
    assert c.post("/api/control", json={"cmd": "keymap", "value": "petting:0:F1"}).status_code == 200
    assert c.post("/api/control", json={"cmd": "keymap", "value": "petting:0:BANANA"}).status_code == 400
    assert c.post("/api/control", json={"cmd": "keymap", "value": "caring:0:A"}).status_code == 400  # gone
    assert c.post("/api/control", json={"cmd": "keymap", "value": "sleeping:0:A"}).status_code == 400
    m = c.get("/api/mind").json()
    assert m["controls"]["keymap"]["petting"] == ["F1", "F", "G", "H"] and "caring" not in m["controls"]["keymap"]
    assert m["senses"]["keypad"]["devices"] == [] and c.get("/api/catalog").json()["keypad"]["dancing"][1] == "tap"
    # a tapped beat from the keypad lands in the tap clock with the key's own timestamp, and turns manual groove on
    assert not pet.manual_groove
    pet.tap.set_bpm(120)
    pet.keypad.events.put(KeyEvent("C", True, 1000.5, "test"))
    pet.step(1000.6)
    assert pet.tap.downbeat_known and pet.tap.phase(1000.5) == 0.0 and pet.manual_groove
    assert any(k == "key" and n == "downbeat" for _, k, n in pet.actions_log)
    assert c.post("/api/control", json={"cmd": "key", "value": "chin_scratch"}).status_code == 200
    assert pet._cuddle > 0.0
    assert c.post("/api/control", json={"cmd": "key", "value": "snack"}).status_code == 400  # the caring layer is gone
    # persisted and restored; a pre-0.7.2 press/hold map is left alone
    pet2 = _pet()
    pet2.settings_file = pet.settings_file
    pet2.load_settings()
    assert pet2.keymap.layers["petting"] == ["F1", "F", "G", "H"]
    # a saved 0.7.x map with the dropped caring layer: that layer is skipped, the rest still loads
    pet.settings_file.write_text(json.dumps({"keymap": {"petting": ["Z", "F", "G", "H"], "caring": ["I", "J", "K", "L"]}, "groove_scale": 1.4}))
    pet4 = _pet()
    pet4.settings_file = pet.settings_file
    pet4.load_settings()
    assert pet4.keymap.layers["petting"][0] == "Z" and pet4.groove_scale == 1.4 and "caring" not in pet4.keymap.layers
    pet4.stop()
    pet.settings_file.write_text(json.dumps({"keymap": {"A": {"press": "tilt_left", "hold": "none"}}}))
    pet3 = _pet()
    pet3.settings_file = pet.settings_file
    pet3.load_settings()
    assert pet3.keymap.layers["dancing"] == ["A", "B", "C", "D"]
    pet.stop(); pet2.stop(); pet3.stop()


def test_dance_layer_nudges_lean_harder_and_tilt_but_never_turn_away():
    from festival_pet.behavior import FaceObs
    from festival_pet.main import NUDGE_LEAN_MIN, NUDGE_LEAN_STEP

    pet = _pet()
    comp, beh = pet.p.composer, pet.p.behavior
    pet.tap.set_bpm(120)
    face = FaceObs(1, 20.0, -5.0, 0.05, None, 0.0)
    pet.p.latest_sighting = lambda: None
    t = 1001.0
    # someone in front of it: engaged, greeted once (the brain is driven with a face directly: the fake IO has no camera)
    for i in range(60):
        beh.tick(Observation(face=face), t + i * 0.05, 0.05)
    assert beh.state == "ENGAGED"
    t += 3.0
    # one tap: a lean, and manual groove comes on
    pet.key_action("groove_left", t, t)
    assert abs(comp.groove_lean - NUDGE_LEAN_MIN) < 1e-9 and pet.manual_groove
    # keep tapping the same way: the lean grows and it tilts its head over, but it stays looking at them
    pet.key_action("groove_left", t + 0.4, t + 0.4)
    assert abs(comp.groove_lean - (NUDGE_LEAN_MIN + NUDGE_LEAN_STEP)) < 1e-9 and comp._gesture.name != "tilt"
    pet.key_action("groove_left", t + 0.8, t + 0.8)
    assert comp._gesture.name == "tilt" and comp._gesture.side == 1.0
    assert abs(comp.groove_lean - (NUDGE_LEAN_MIN + 2 * NUDGE_LEAN_STEP)) < 1e-9
    pet.step(t + 1.0)
    assert comp._gaze_target is not None and abs(comp._gaze_target[0] - 20.0) < 1e-6  # still on them, not turned 60 degrees away
    assert pet._last_obs.busy is None  # no "turn" for the brain to wait out any more
    # the lean puts a few degrees of body into it, and no more (the rest of the body's angle is it facing them)
    for i in range(1, 60):
        comp.groove_lean = 1.0  # as if the key were held down: the lean decays otherwise
        _, _, leaning = comp.sample(t + 1.0 + i * 0.02, 0.02)
    comp.groove_lean = 0.0
    for i in range(60, 200):
        _, _, plain = comp.sample(t + 1.0 + i * 0.02, 0.02)
    assert 2.0 < leaning - plain < 8.0
    # capped however hard you drum on it
    for k in range(8):
        pet.key_action("groove_right", t + 10.0 + k * 0.2, t + 10.0 + k * 0.2)
    assert comp.groove_lean == -1.0
    # the stop combo: left right left right within a second ends the dancing
    assert pet.manual_groove and pet.tap.active
    for k, a in enumerate(("groove_left", "groove_right", "groove_left")):
        pet.key_action(a, t + 20.0 + k * 0.2, t + 20.0 + k * 0.2)
    assert pet.manual_groove  # three is not the combo
    pet.key_action("groove_right", t + 20.6, t + 20.6)
    assert not pet.manual_groove and not pet.tap.active and comp.groove_lean == 0.0
    assert "done dancing" in beh.thoughts[-1][1] and comp._gesture.name == "shake_off"
    # too slow is just nudging (and the first tap turns the groove back on)
    for k, a in enumerate(("groove_left", "groove_right", "groove_left", "groove_right")):
        pet.key_action(a, t + 30.0 + k * 0.5, t + 30.0 + k * 0.5)
    assert pet.manual_groove
    pet.stop()


def test_petting_keys_build_up_into_one_long_cuddle_instead_of_firing_animations():
    from festival_pet.keypad import KeyEvent
    from festival_pet.main import CUDDLE_FADE_S, CUDDLE_HOLD_S, CUDDLE_STEP

    pet = _pet()
    beh, comp = pet.p.behavior, pet.p.composer
    t = 1001.0
    gestures = lambda: [n for _, k, n in pet.actions_log if k == "gesture"]  # noqa: E731
    pet.keypad.events.put(KeyEvent("E", True, t, "test"))  # head pat
    pet.step(t + 0.05)
    thoughts = lambda: " ".join(x[1] for x in beh.thoughts)  # noqa: E731
    assert abs(pet._cuddle - CUDDLE_STEP) < 1e-9 and "head pats" in thoughts()
    assert "ahh, head pets" in thoughts()  # the one edge, at the start
    # somebody drumming on all four keys like a fidget toy: one hand on it, a build-up, not four animations
    n_before = len(gestures())
    now = t + 0.1
    for i in range(120):
        pet.keypad.events.put(KeyEvent("EFGH"[i % 4], True, now, "test"))
        pet.step(now)
        now += 0.1
    assert pet._cuddle == 1.0 and comp.petted  # a hand is on it the whole time
    # 120 presses: the build-up's four steps plus the brain's own slow lean-and-purr, nothing per press
    assert len(gestures()) - n_before <= 12
    assert "melted" in thoughts()
    assert all(k in thoughts() for k in ("head pats", "chin scratches", "ear rubs", "tummy rubs"))  # it names whatever is being done
    # hands off: the hand comes off after CUDDLE_HOLD_S and the build-up ebbs away
    pet.step(now + CUDDLE_HOLD_S + 0.1)
    assert not pet._last_obs.petting
    for i in range(30):
        pet.step(now + CUDDLE_HOLD_S + 0.2 + i * CUDDLE_FADE_S / 4)
    assert pet._cuddle == 0.0
    # and it can all happen again
    n_before = len(gestures())
    pet.keypad.events.put(KeyEvent("F", True, now + 200, "test"))
    pet.step(now + 200.05)
    assert 0 < pet._cuddle <= CUDDLE_STEP and len(gestures()) > n_before
    assert not pet.manual_groove  # only the dancing layer touches the groove
    pet.stop()


def test_held_mode_asks_to_be_turned():
    pet = _pet()
    pet.control("pickup", True)
    assert pet.p.composer.held
    beh = pet.p.behavior  # looks where it last saw someone: way off to its left
    beh.seen_spots.note(100.0, 0.0, 1000.9)
    beh._enter("SEARCHING", 1001.0)
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
    pet._watching = lambda obs: False  # nobody is looking: one song, no encore, no big bow
    song = pet.sing(1000.2)  # on the pet's own (fake) clock, so the step loop below is short
    assert pet.last_song is song and pet.tap.bpm == songs.body_bpm(song)
    assert c.get("/api/mind").json()["song"]["last"]["name"] == song["name"]
    t, end = 1000.2, pet._singing_until
    assert end - 1000.2 < 45
    while t < end + 0.5:
        pet.step(t); t += 0.02
    assert pet.p.composer._gesture.name in ("wiggle", "tada") and pet._songs_in_set == 0  # pleased with itself, no bow
    assert not pet._next_song_at
    assert c.post("/api/control", json={"cmd": "save_song"}).status_code == 200
    assert json.loads(pet.songs_file.read_text())[0]["name"] == pet.last_song["name"]
    assert c.get("/api/mind").json()["song"]["last"]["saved"]
    pet2 = _pet(); pet2.songs_file = pet.songs_file; pet2.load_songs()
    assert pet2.songs == pet.songs
    # a style can be asked for by name, and only a real one
    assert c.post("/api/control", json={"cmd": "sing", "value": "bass"}).status_code == 200
    assert pet.last_song["style"] == "bass" and c.get("/api/mind").json()["song"]["last"]["style"] == "bass"
    for gone in ("polka", "drumline"):  # the drumline beeps were dropped: asking for one is an error now
        assert c.post("/api/control", json={"cmd": "sing", "value": gone}).status_code == 400
    # ...and a saved drumline song is dropped on load rather than kept as one it can no longer play
    pet.songs_file.write_text(json.dumps([{"style": "drumline", "bpm": 100, "bars": ["quarters"], "hi": 1500, "lo": 1000, "name": "old one"},
                                          {"bpm": 100, "bars": ["quarters"], "hi": 1500, "lo": 1000, "name": "older one"}]))
    pet5 = _pet(); pet5.songs_file = pet.songs_file; pet5.load_songs()
    assert pet5.songs == []
    pet5.stop()
    assert c.post("/api/control", json={"cmd": "singing", "value": True}).status_code == 200
    assert c.get("/api/mind").json()["song"]["next_in_s"] is not None
    pet.stop(); pet2.stop()


def test_watching_it_sing_earns_an_encore_and_a_proper_bow():
    from festival_pet.behavior import FaceObs
    from festival_pet.main import ENCORE_GAP_S, WATCHING_YAW_DEG

    pet = _pet()
    beh, comp = pet.p.behavior, pet.p.composer
    # eye contact is their head pointed at it, not just a face in the frame
    looking = FaceObs(1, 10.0, 0.0, 0.05, None, 0.0, head_yaw_deg=5.0, head_pitch_deg=-3.0)
    away = FaceObs(1, 10.0, 0.0, 0.05, None, 0.0, head_yaw_deg=WATCHING_YAW_DEG + 10, head_pitch_deg=0.0)
    assert pet._watching(Observation(face=looking))
    assert not pet._watching(Observation(face=away)) and not pet._watching(Observation())

    pet._watching = lambda obs: True  # somebody settles in to watch the whole thing
    t = 1001.0
    songs_sung = []
    for _ in range(4):
        if not (t < pet._singing_until or pet._next_song_at):
            break
        t += 0.02
    pet.sing(t)
    songs_sung.append(pet.last_song["name"])
    seen_encore_gap = False
    for _ in range(200000):
        pet.step(t)
        if pet._next_song_at:
            seen_encore_gap = True
        if pet.last_song["name"] not in songs_sung:
            songs_sung.append(pet.last_song["name"])
        # the song's own clock is zeroed the moment it is over, so "still performing" is that, not a
        # comparison against a deadline this loop might step straight over
        if not (pet._singing_until or pet._next_song_at):
            break
        t += 0.02
    assert seen_encore_gap and pet._songs_in_set == 0  # the set ran and then ended
    assert 2 <= len(songs_sung) <= 3  # an encore, and sometimes a third
    assert comp._gesture is not None and comp._gesture.name == "bow"  # the set earned the whole routine
    assert any("thank you" in x[1] for x in beh.thoughts)
    assert any(k == "song" and "encore" in n for _, k, n in pet.actions_log)
    # the gap between songs still counts as performing, so the brain does not go and start something else
    pet.sing(t + 1.0)
    pet._singing_until = t + 1.05  # the first step at or past the end is the one that ends it, whatever the dt
    pet.step(t + 1.02)  # one tick of the song being watched, so the encore is earned
    pet.step(t + 1.06)
    assert pet._next_song_at and pet._last_obs.busy == "sing"
    pet.stop()


def test_manual_groove_on_cuts_a_song_and_a_game_and_keeps_the_brain_out_of_them():
    from festival_pet.behavior import FaceObs

    t = time.time()
    pet = _pet(t - 1.0)
    beh = pet.p.behavior
    pet.control("singing", True)
    pet.sing(t)
    assert pet._singing_until > t and not pet.sound._cut.is_set()
    pet.control("manual_groove", True)  # mid-song: the song stops, we're grooving
    assert pet._singing_until == 0.0 and pet.sound._cut.is_set() and pet.sound.busy_until == 0.0
    pet.step(t + 0.05)
    assert pet._last_obs.grooving and pet._last_obs.busy is None
    # the brain sees a beat: bored as it is, no game or song gets chosen
    face = FaceObs(1, 0.0, 0.0, 0.2, None, 0.0)
    beh.mood.boredom, beh.mood.curiosity = 0.9, 0.9
    pet.vision = ArmsVision()
    for i in range(1, 160):
        pet.vision.arms = _arms("down", "down", t + i * 0.05)
        pet._last_obs.face = face
        pet.step(t + i * 0.05)
    assert beh.activity not in ("mime", "sing", "mirror") and not pet.mime.active and pet._singing_until == 0.0
    assert not beh.can_sing
    # a game started from the page, then the dancing layer pressed: the game is dropped
    pet.control("manual_groove", False)
    pet._last_obs.face = face
    assert pet.start_simon("head", t + 10) == "head" and pet.mime.active
    pet.key_action("groove_left", t + 10.5, t + 10.5)
    assert pet.manual_groove and not pet.mime.active
    pet.stop()


def test_a_wave_gets_a_mirrored_wave_back_and_a_hug_gets_a_nuzzle():
    from festival_pet.pose import Arms, HUG_HOLD_S

    def arms(ts, l_deg, r_deg, r_dx=0.0, hug=False):
        pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (150.0, 100.0), "l_elbow": (95.0, 125.0), "r_elbow": (155.0, 125.0),
               "l_wrist": (100.0, 100.0 if hug else 40.0), "r_wrist": (150.0 + r_dx, 100.0 if hug else 40.0),
               "l_wrist_ok": True, "r_wrist_ok": True}
        from festival_pet.pose import arm_level
        return Arms(ts, l_deg, r_deg, arm_level(l_deg), arm_level(r_deg), 0.9, 50.0, pts)

    t = time.time()
    pet = _pet(t - 1.0)
    comp, beh = pet.p.composer, pet.p.behavior
    pet.vision = ArmsVision()
    # their RIGHT hand waves: three swings; the pose runs live while the arm is up, and it waves back with its LEFT antenna
    for k in range(8):
        now = t + k * 0.15
        pet.vision.arms = arms(now, 20.0, 160.0, r_dx=15.0 if k % 2 else -15.0)
        pet.step(now)
    assert pet.vision.pose_live
    assert any(k == "arms" and n == "wave right" for _, k, n in pet.actions_log)
    assert comp._gesture.name == "wave" and comp._gesture.side == 1.0 and "waving their right hand" in beh.thoughts[-1][1]
    assert not pet.mime.active
    # a hug: both arms out for a while
    t2 = t + 10.0
    now = t2
    while now < t2 + HUG_HOLD_S + 0.5:
        pet.vision.arms = arms(now, 90.0, 90.0, hug=True)
        pet.step(now)
        now += 0.1
    assert any(k == "arms" and n == "hug" for _, k, n in pet.actions_log)
    assert comp._gesture.name == "hug" and "a hug" in beh.thoughts[-1][1]
    assert [n for _, k, n in pet.actions_log if k == "sound"][-1] == "coo"
    assert pet.mind()["arms"]["watching"]
    pet.stop()


def test_the_plur_handshake_trades_a_bracelet_both_ways(tmp_path):
    import numpy as np

    from festival_pet.main import KANDI_GIVE_S, KANDI_OFFER_S, KANDI_SETTLE_S
    from festival_pet.motion import CAREFUL_ANTENNA_RAD, OFFER_RAD
    from festival_pet.pose import PLUR_STEPS
    from test_pose import PLUR_POSES

    t = time.time()
    pet = _pet(t - 1.0)
    beh, comp = pet.p.behavior, pet.p.composer
    pet.vision = ArmsVision()
    mem = pet.p.memory
    mem.path = tmp_path / "mem.json"  # this test enrols someone, so give it somewhere of its own to save
    beh._engaged_person = mem.enroll(np.ones(4), t)
    pet.set_bracelets([0], t)  # it is wearing one on its right ear, ready to trade away
    assert comp.loaded == [True, False]

    now = t
    for step in PLUR_STEPS:  # peace, love, unity, respect, each held for a beat
        end = now + 1.2
        while now < end:
            pet.vision.arms = PLUR_POSES[step](now)
            pet.step(now)
            now += 0.05
        now += 0.3
    assert [n for _, k, n in pet.actions_log if k == "arms" and n.startswith("plur")] == [f"plur {s}" for s in PLUR_STEPS]
    assert pet._last_obs.busy == "kandi"  # nothing else gets started in the middle of this

    # it gives first: the loaded ear tilts down and the antenna lowers until the bracelet runs off
    assert pet._kandi_give_t0 > 0.0 and pet.kandi_side == 0
    give_t0, rolls, antenna = pet._kandi_give_t0, [], []
    while pet._kandi_give_t0:
        pet.step(now)
        _, ants, _ = comp.sample(now, 0.02)
        rolls.append(comp._hold_roll)
        antenna.append(ants[0])
        now += 0.05
    assert max(rolls) > 10.0  # the head really did lean over
    assert max(abs(x) for x in antenna) > 1.5  # ...and the antenna came right down, past the upright gate
    # the ear TOGGLE stays on (you will hang a replacement on it), but the gate is off while the trade runs
    assert pet.kandi_on[0] and not comp.loaded[0]
    assert "gave one away" in [n for _, k, n in pet.actions_log if k == "kandi"]
    gave_at = [t for t, k, n in pet.actions_log if k == "kandi" and n == "gave one away"][0]
    assert gave_at - give_t0 > KANDI_GIVE_S * 0.5  # it lets go near the end, not the moment it starts
    assert any(k == "sound" and n == "giggle" and abs(t - gave_at) < 0.3 for t, k, n in pet.actions_log)  # giggles as it goes
    assert abs((now - give_t0) - KANDI_GIVE_S) < 0.4  # point, tilt, lower, jiggle it off, hold, come back up
    # ...and it shakes the antenna about down there, to bounce the bracelet off the twist near the base
    low = [x for x in antenna[-int(KANDI_GIVE_S / 0.05 * 0.3):]]
    assert max(low) - min(low) > 0.2

    # ...then asks for one back, on the same ear, and freezes
    assert pet._kandi_offer_until > now and pet.kandi_side == 0 and comp.offer_side == 0
    for _ in range(40):
        pet.step(now)
        now += 0.05
    _, ants, _ = comp.sample(now, 0.02)
    assert abs(ants[0] - OFFER_RAD) < 0.02 and ants[1] > 0.5  # right one offered, left one out of the way
    head_a, _, _ = comp.sample(now, 0.02)
    head_b, _, _ = comp.sample(now + 0.5, 0.02)
    assert float(np.abs(head_a - head_b).max()) < 0.01  # dead still while they thread one on

    # the bracelet landing is felt as that antenna being pushed: no flinch, no ear-tickle giggle
    pet.touch.update = lambda *a, **k: True  # type: ignore[assignment]
    pet.touch.last_side = 0
    pet.step(now)
    now += 0.05
    pet.touch.update = lambda *a, **k: False  # type: ignore[assignment]
    assert pet._kandi_got_at > 0.0 and not pet._last_obs.touched
    assert "that's mine now" in beh.thoughts[-1][1]

    # two seconds of settling, then it eases back and is wearing the new one
    while now < pet._kandi_got_at + KANDI_SETTLE_S + 0.2:
        pet.step(now)
        now += 0.05
    assert pet.kandi_on == [True, False] and comp.loaded == [True, False] and comp.offer_side is None
    assert comp.gentle_until > now
    assert beh._engaged_person.kandi == 1 and beh._engaged_person.affection > 0.2
    assert "kandi traded" in [n for _, k, n in pet.actions_log if k == "kandi"]
    # ...and it rides out anything from here, because the gate keeps that antenna upright
    for k in range(200):  # the gate eases shut over a few seconds on a bracelet that has just gone on
        pet.step(now + k * 0.02)
    now += 4.0
    assert comp._gate[0] == 1.0
    worst = 0.0
    for k in range(600):
        comp.request_gesture("bounce", now + k * 0.02, 3)
        comp.groove = ((k * 0.02 / 0.5) % 1.0, 0.0, 1.0)
        _, ants, _ = comp.sample(now + k * 0.02, 0.02)
        worst = max(worst, abs(ants[0]))
    assert worst <= CAREFUL_ANTENNA_RAD + 1e-6
    assert KANDI_OFFER_S >= 10.0  # long enough to dig one out of a bag
    pet.stop()


def test_kandi_falls_back_to_asking_and_can_be_driven_from_the_page():
    from festival_pet.main import KANDI_OFFER_S

    pet = _pet()
    app = FastAPI(); install_routes(app, pet); c = TestClient(app)
    # wearing nothing: there is nothing to give, so the trade is just the asking half
    assert not any(pet.kandi_on)
    assert c.post("/api/control", json={"cmd": "kandi", "value": "left"}).status_code == 200
    assert not pet._kandi_give_t0 and pet.kandi_side == 1 and pet.p.composer.offer_side == 1
    m = c.get("/api/mind").json()["kandi"]
    assert m["offering"] and m["side"] == "left" and m["waiting_s"] > 0 and m["wearing"] == []
    # nobody had one ready: it gives up after the timeout and goes back to normal
    pet.step(time.time() + KANDI_OFFER_S + 0.1)
    assert not pet._kandi_offer_until and pet.p.composer.offer_side is None
    assert "nothing came" in " ".join(n for _, k, n in pet.actions_log if k == "kandi")
    # started by mistake: cancel it
    assert c.post("/api/control", json={"cmd": "kandi", "value": True}).status_code == 200
    assert c.post("/api/control", json={"cmd": "kandi", "value": False}).status_code == 200
    assert not pet._kandi_offer_until and not pet._kandi_give_t0 and pet.signs.plur_step == 0
    # saying which ears are loaded, for when you move a bracelet onto the body
    assert c.post("/api/control", json={"cmd": "bracelet", "value": "both"}).status_code == 200
    assert pet.p.composer.loaded == [True, True] and c.get("/api/mind").json()["kandi"]["wearing"] == ["right", "left"]
    assert c.post("/api/control", json={"cmd": "bracelet", "value": "none"}).status_code == 200
    assert pet.p.composer.loaded == [False, False]
    assert c.post("/api/control", json={"cmd": "bracelet", "value": "elbow"}).status_code == 400
    assert c.post("/api/control", json={"cmd": "kandi", "value": "sideways"}).status_code == 400
    # the shed tilt is tunable, because which way it has to lean is a fact about the real head
    assert c.post("/api/control", json={"cmd": "kandi_roll", "value": -25}).status_code == 200
    assert pet.kandi_roll_deg == -25.0 and c.get("/api/mind").json()["controls"]["kandi_roll"] == -25.0
    assert c.post("/api/control", json={"cmd": "kandi_roll", "value": 80}).status_code == 400
    pet.stop()


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


class ArmsVision:
    """A fake vision that reports arms (and nothing else)."""

    pose, pose_live, refined = True, False, False
    stats = {"detect_ms": 1.0, "embed_ms": 0.0, "body_ms": 0.0, "pose_ms": 0.0, "frames": 3, "faces": 1, "bodies": 0, "last_frame_at": 0.0, "no_frame": 0, "errors": 0, "last_error": ""}
    last_jpeg = None
    body_enabled = True

    def __init__(self):
        self.arms = None

    def latest_arms(self):
        return self.arms

    def status(self, now):
        return {**self.stats, "alive": True, "active": True, "refined": False, "pose": True, "pose_live": self.pose_live, "frame_age_s": 0.1}


def _arms(left, right, ts):
    """A reading with a full set of landmarks, as the real reader always produces."""
    from festival_pet.pose import Arms

    deg = {"down": 10.0, "out": 90.0, "up": 170.0}
    dy = {"down": 60.0, "out": 0.0, "up": -60.0}  # where the wrist sits relative to the shoulder
    pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (150.0, 100.0),
           "l_elbow": (95.0, 125.0), "r_elbow": (155.0, 125.0),
           "l_wrist": (80.0, 100.0 + dy[left]), "r_wrist": (170.0, 100.0 + dy[right]),
           "l_wrist_ok": True, "r_wrist_ok": True}
    return Arms(ts, deg[left], deg[right], left, right, 0.9, 50.0, pts)


def test_arm_simon_says_runs_from_the_page_and_falls_back_to_the_head_game():
    from festival_pet.behavior import FaceObs

    t = time.time()
    pet = _pet(t - 1.0)
    pet.vision = ArmsVision()
    app = FastAPI(); install_routes(app, pet); c = TestClient(app)
    r = c.post("/api/control", json={"cmd": "mime", "value": "arms"})
    assert r.status_code == 400 and "arms" in r.json()["detail"]  # nobody there at all
    # arms in view: the arm game, and the pose runs live
    pet.vision.arms = _arms("down", "down", t)
    pet.step(t)
    assert c.post("/api/control", json={"cmd": "mime", "value": "arms"}).status_code == 200
    assert pet.mime.active and pet.mime.kind == "arms"
    t0 = pet.mime._until - 1.8  # the game's own start time (INTRO_S before its first deadline)
    pet.step(t0 + 0.05)
    assert pet.vision.pose_live
    m = c.get("/api/mind").json()
    assert m["mime"]["kind"] == "arms" and m["arms"]["left"] == "down" and m["senses"]["vision"]["pose_live"]
    # the demo puts the antennas in the flag position
    from festival_pet.mime import ARM_LEVEL_DEG, ARM_MOVES, INTRO_S

    for i in range(1, 8):
        pet.vision.arms = _arms("down", "down", t0 + INTRO_S + i * 0.05)
        pet.step(t0 + INTRO_S + i * 0.05)
    left, right = ARM_MOVES[pet.mime.sequence[0]]
    assert pet.p.composer.arms == (ARM_LEVEL_DEG[left], ARM_LEVEL_DEG[right])
    assert c.post("/api/control", json={"cmd": "mime", "value": False}).status_code == 200
    assert not pet.mime.active and pet.p.composer.arms is None
    pet.step(t0 + 3.0)
    assert not pet.vision.pose_live
    # only a face, too close for arms: asking for the arm game gets the head game
    pet.vision.arms = None
    pet._last_obs.face = FaceObs(1, 0.0, 0.0, 0.2, None, 0.0)
    assert c.post("/api/control", json={"cmd": "mime", "value": "arms"}).status_code == 200
    assert pet.mime.active and pet.mime.kind == "head"
    assert c.post("/api/control", json={"cmd": "mime", "value": "legs"}).status_code == 400
    pet.stop()


def test_dance_along_copies_the_arms_to_the_beat_with_a_flourish():
    t = time.time()
    pet = _pet(t - 1.0)
    pet.vision = ArmsVision()
    app = FastAPI(); install_routes(app, pet); c = TestClient(app)
    assert c.post("/api/control", json={"cmd": "manual_groove", "value": True}).status_code == 200
    assert c.post("/api/control", json={"cmd": "bpm", "value": 120}).status_code == 200
    comp = pet.p.composer
    copied, flourished = [], False
    for i in range(160):  # 8 s: 16 beats at 120 bpm
        now = t + i * 0.05
        pet.vision.arms = _arms("up", "out", now)  # their left up, right out
        pet.step(now)
        if now < pet._flourish_until:
            flourished = True
        elif pet._copying_arms and comp.arms is not None:
            copied.append(comp.arms)
    st = c.get("/api/mind").json()["dance_along"]
    assert st["on"] and st["copying"]
    assert pet._copying_arms and copied and flourished
    assert all(a == (90.0, 170.0) for a in copied)  # mirror: their right (out) is its left, their left (up) its right
    assert any("dancing with you" in th for _, th in pet.p.behavior.thoughts) and any("my turn" in th for _, th in pet.p.behavior.thoughts)
    # switched off: it lets go
    assert c.post("/api/control", json={"cmd": "dance_along", "value": False}).status_code == 200
    pet.step(t)
    assert not pet._copying_arms and comp.arms is None
    assert c.get("/api/mind").json()["controls"]["dance_along"] is False
    pet.stop()


def test_mind_json_never_carries_numpy_scalars():
    """0.6.5 put a numpy bool in the vision status and every poll of the page failed with a 500."""
    import numpy as np

    from festival_pet.vision import Vision

    pet = _pet()
    v = object.__new__(Vision)
    v._thread = None
    v._active = threading.Event()
    v.refined = bool(np.float32(40.0) < 90)  # what the fixed code produces
    v.pose, v.pose_live, v._arms, v._lock = None, False, None, threading.Lock()
    v.stats = {"detect_ms": 1.0, "embed_ms": 0.0, "body_ms": 0.0, "pose_ms": 0.0, "frames": 3, "faces": 1, "bodies": 0, "last_frame_at": 0.0, "no_frame": 0, "errors": 0, "last_error": ""}
    pet.vision = v

    def walk(x, path="mind"):
        if isinstance(x, dict):
            for k, val in x.items():
                walk(val, f"{path}.{k}")
        elif isinstance(x, (list, tuple)):
            for i, val in enumerate(x):
                walk(val, f"{path}[{i}]")
        else:
            assert not isinstance(x, np.generic), f"{path} is {type(x).__name__}"
    walk(pet.mind())
    app = FastAPI(); install_routes(app, pet); c = TestClient(app)
    assert c.get("/api/mind").status_code == 200
    # and the unfixed value would have been caught here
    v.refined = np.float32(40.0) < 90
    import pytest
    with pytest.raises(AssertionError):
        walk(pet.mind())
    pet.stop()


class SpinMove:
    """A library move that ends with the body swung right round, like the circus one."""

    duration = 0.6
    sound_path = None

    def evaluate(self, t):
        return np.eye(4), np.zeros(2), math.radians(180.0 * min(1.0, t / self.duration))


def _stub_interpolation():
    """The blend back out of a library move borrows the SDK's pose interpolation, which is not installed
    off-robot. A straight lerp of the two matrices is close enough for what this test measures."""
    import sys
    import types

    mod = types.ModuleType("reachy_mini.utils.interpolation")
    mod.linear_pose_interpolation = lambda a, b, t: a * (1 - t) + b * t
    for name in ("reachy_mini", "reachy_mini.utils", "reachy_mini.utils.interpolation"):
        sys.modules.setdefault(name, types.ModuleType(name) if name != "reachy_mini.utils.interpolation" else mod)
    sys.modules["reachy_mini.utils.interpolation"] = mod


def test_a_move_that_spins_the_body_unwinds_instead_of_snapping_back():
    """The circus move finishes 180 degrees round. That offset used to vanish between one tick and the
    next, so the robot whipped back to front at whatever speed the motors managed — with kandi on its
    ears, alarming. It has to come back at a walking pace instead."""
    from festival_pet.main import MOVE_BODY_RETURN_DEG_S
    from festival_pet.motion import BODY_YAW_LIMIT

    _stub_interpolation()
    pet = _pet(time.time())  # library moves are timed off the wall clock, so this test runs on it too
    pet.p.library = lambda name: SpinMove()
    now = time.time() + 0.1
    pet.control("move", "circus1")
    bodies, unwinding = [], []
    end = now + SpinMove.duration + 6.0
    while now < end:
        was_moving = pet.move is not None
        pet.step(now)
        bodies.append(pet.last_body)
        if not was_moving and pet.move is None and len(bodies) > 1:
            unwinding.append(bodies[-1] - bodies[-2])  # only the way back: the move's own swing is its business
        now += 0.02
    assert max(abs(b) for b in bodies) >= BODY_YAW_LIMIT - 1e-6  # the move really did turn it right round
    assert abs(bodies[-1] - pet.p.composer.body_yaw) < 8.0  # ...and it came back to where the pet itself wants the body
    fastest = max(abs(d) for d in unwinding) / 0.02
    assert fastest <= MOVE_BODY_RETURN_DEG_S * 1.5  # no snap: it unwinds at about the rate we asked for
    pet.stop()


def test_the_offered_ear_is_turned_to_face_the_person_and_the_other_stays_safe():
    """Offering an ear used to leave it out at the side of the head, pointing past whoever was standing
    there, and with the other ear loaded and gated the two looked nearly the same: you could not tell
    which one was on offer."""
    from festival_pet.motion import OFFER_RAD, OFFER_YAW

    pet = _pet()
    comp = pet.p.composer
    comp.set_gaze((0.0, 0.0))
    pet.set_bracelets([0, 1], 1000.2)  # wearing one on each ear
    pet.touch.update = lambda *a, **k: False  # type: ignore[assignment]  nobody is putting one on yet
    pet.start_kandi(1000.2, 1)  # the LEFT ear is offered
    now = 1000.2
    for _ in range(80):
        pet.step(now)
        now += 0.05
    head, ants, _ = comp.sample(now, 0.02)
    # offering its left: the head turns to its RIGHT of wherever it had been looking, which is what swings
    # that ear round to the front instead of leaving it pointing off past them
    # (not the full OFFER_YAW to the degree: the head can only turn so far on the body, which is fine —
    #  what matters is that it turns that way at all, where before it turned not at all)
    turn = _euler(head)[2] - comp._still_yaw
    assert -OFFER_YAW * 1.3 < turn < -OFFER_YAW * 0.5
    assert abs(ants[1] - -OFFER_RAD) < 0.05  # the left antenna is the post, tipped toward them
    # the other ear is wearing one of its own, so it stays up where that is safe rather than leaning away
    assert abs(ants[0]) < 0.5
    pet.stop()


def test_wearing_kandi_damps_the_head_without_stopping_it_looking_at_you():
    from festival_pet.motion import ANTENNA_NEUTRAL

    def swing(pet, now):
        """How far the head moves about over a few seconds of hard grooving."""
        comp = pet.p.composer
        comp.set_gaze((20.0, 0.0))
        for k in range(300):  # settle the gaze and the gate first
            comp.sample(now + k * 0.02, 0.02)
        yaws, base = [], now + 6.0
        for k in range(300):
            t = base + k * 0.02
            comp.groove = ((k * 0.02 / 0.5) % 1.0, (k * 0.02 / 2.0) % 1.0, 1.0)
            comp.groove_mix.sway = 2.0
            comp._groove_style = 2  # the sway-over-the-bar one: the style that moves the head sideways
            head, _, _ = comp.sample(t, 0.02)
            yaws.append(_euler(head)[2])
        return max(yaws) - min(yaws), sum(yaws) / len(yaws)

    bare = _pet()
    free, aim_free = swing(bare, 1000.2)
    bare.stop()

    worn = _pet()
    worn.control("kandi_damp", 0.8)
    worn.set_bracelets([0], 1000.2)
    damped, aim_worn = swing(worn, 1000.2)
    assert damped < free * 0.5  # a bracelet hanging off an ear is not thrown about any more
    assert abs(aim_worn - aim_free) < 3.0  # ...and it is still looking at the same person
    assert worn.control("kandi_damp", 0.0) == {"ok": True}
    assert worn.p.composer.kandi_damp == 0.0
    for bad in (-0.1, 1.5):
        try:
            worn.control("kandi_damp", bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not be an allowed damping")
    worn.stop()


def test_showing_it_a_plur_pose_teaches_it_what_that_pose_looks_like():
    from festival_pet.main import PLUR_TRAIN_READY_S, PLUR_TRAIN_WATCH_S
    from festival_pet.pose import Arms, plur_pose

    def arms(ts):  # arms at 55 degrees with the hands in toward the middle: none of the built-in rules
        pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (150.0, 100.0), "l_elbow": (100.0, 120.0),
               "r_elbow": (150.0, 120.0), "l_wrist": (105.0, 112.0), "r_wrist": (146.0, 110.0),
               "l_wrist_ok": True, "r_wrist_ok": True}
        return Arms(ts, 55.0, 57.0, "out", "out", 0.9, 50.0, pts)

    pet = _pet()
    app = FastAPI()
    install_routes(app, pet)
    c = TestClient(app)
    assert plur_pose(arms(0.0)) is None

    assert c.post("/api/control", json={"cmd": "train_plur", "value": "peace"}).status_code == 200
    assert c.post("/api/control", json={"cmd": "train_plur", "value": "vibes"}).status_code == 400
    now = 1000.2
    pet.train_plur("peace", now)  # again on the pet's own clock, so the test can step it
    ears = []
    while pet._training is not None and now < 1000.2 + PLUR_TRAIN_READY_S + PLUR_TRAIN_WATCH_S + 2.0:
        pet._train_tick(Observation(arms=arms(now)), now)  # as if the vision thread had a fresh reading
        ears.append(pet.p.composer.ear_hold[1])
        if now < 1000.2 + PLUR_TRAIN_READY_S:
            assert not pet.signs.trained  # still counting them in: it has not started measuring yet
        now += 0.1
    assert pet._training is None
    # the countdown really did run down an antenna, twice: once to get into the pose, once while it read
    ticks = [e for e in ears if e is not None]
    assert len(ticks) > 10 and max(ticks) - min(ticks) > 0.5
    assert set(pet.signs.trained) == {"peace"}
    assert abs(pet.signs.trained["peace"]["hi_deg"] - 57.0) < 1.0
    assert plur_pose(arms(0.0), pet.signs.trained) == "peace"  # it knows that pose now
    assert "learned peace" in " ".join(n for _, k, n in pet.actions_log if k == "plur")

    # it survives a restart, and it can be thrown away again
    settings = pet._settings()
    assert settings["plur_trained"]["peace"]["hi_deg"] == pet.signs.trained["peace"]["hi_deg"]
    pet2 = _pet()
    pet2.control("plur_trained", settings["plur_trained"])
    assert plur_pose(arms(0.0), pet2.signs.trained) == "peace"
    try:
        pet2.control("plur_trained", {"vibes": {}})
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown step should not be quietly accepted")
    assert c.post("/api/control", json={"cmd": "forget_plur", "value": True}).status_code == 200
    assert pet.signs.trained == {}
    pet.stop(); pet2.stop()


def test_learning_a_pose_gives_up_when_it_cannot_see_the_arms():
    from festival_pet.main import PLUR_TRAIN_READY_S, PLUR_TRAIN_WATCH_S

    pet = _pet()
    pet.train_plur("respect", 1000.2)
    assert pet._training is not None
    now = 1000.2
    while pet._training is not None and now < 1000.2 + PLUR_TRAIN_READY_S + PLUR_TRAIN_WATCH_S + 2.0:
        pet._train_tick(Observation(), now)  # nobody in front of it
        now += 0.1
    assert pet._training is None and pet.signs.trained == {}  # it says so rather than learning nonsense
    assert "failed" in " ".join(n for _, k, n in pet.actions_log if k == "plur")
    pet.stop()


def test_it_bobs_to_its_own_song_and_plays_the_phrasing_with_its_antennas():
    """The head used to stand still through a whole song: the brain clears the groove every tick, the
    beat tracker cannot hear our own song (the mics are deaf while we play), and nothing put it back."""
    from festival_pet import songs
    from festival_pet.main import SONG_GROOVE

    pet = _pet()
    now = 1000.2
    song = pet.sing(now)
    grooves, antennas, sections = [], [], []
    end = now + songs.duration(song)
    while now < end - 0.1:
        pet.step(now)
        assert pet.p.composer.groove is not None, "not bobbing to its own song"
        grooves.append(pet.p.composer.groove[2])
        if pet.p.composer.arms is not None:
            antennas.append(tuple(round(x) for x in pet.p.composer.arms))
        sections.append(pet._song_move)
        now += 0.05
    # the bob follows the shape of the song, not one flat level all the way through
    assert max(grooves) > min(grooves) + 0.2
    assert max(grooves) <= SONG_GROOVE + 1e-6
    # ...and the antennas play the phrasing: a build climbs, the drop slams, each section looks different
    assert "build" in sections and "drop" in sections
    assert len(set(antennas)) > 20
    builds = [a for a, m in zip(antennas, sections) if m == "build"]
    assert max(b[0] for b in builds) - min(b[0] for b in builds) > 80.0  # the riser really does climb
    # the song ends and the antennas are given back
    while now < end + 0.5:
        pet.step(now)
        now += 0.05
    assert pet.p.composer.arms is None and pet._song_move == ""
    pet.stop()


def test_teaching_the_handshake_runs_as_one_routine_and_concentrates():
    """Stopping between poses to press a button is exactly when it loses you, so it calls for all four
    itself — and while it does, it stops being a pet: gaze pinned, nothing else started, a countdown
    running down an antenna so you can see it is still waiting for you."""
    from festival_pet.behavior import FaceObs
    from festival_pet.main import PLUR_CLOCK_EAR, PLUR_TRAIN_GAP_S, PLUR_TRAIN_READY_S, PLUR_TRAIN_WATCH_S
    from festival_pet.pose import PLUR_STEPS, Arms

    def arms(ts):
        pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (150.0, 100.0), "l_elbow": (100.0, 120.0),
               "r_elbow": (150.0, 120.0), "l_wrist": (105.0, 112.0), "r_wrist": (146.0, 110.0),
               "l_wrist_ok": True, "r_wrist_ok": True}
        return Arms(ts, 55.0, 57.0, "out", "out", 0.9, 50.0, pts)

    pet = _pet()
    beh = pet.p.behavior
    pet.train_plur("all", 1000.2)
    now, called, clocks = 1000.2, [], []
    per_step = PLUR_TRAIN_READY_S + PLUR_TRAIN_WATCH_S + PLUR_TRAIN_GAP_S
    while pet._training is not None and now < 1000.2 + per_step * len(PLUR_STEPS) + 2.0:
        obs = Observation(arms=arms(now), face=FaceObs(1, 20.0, -3.0, 0.05, None, 0.0))
        pet._train_tick(obs, now)
        called.append(pet._training["step"] if pet._training else "")
        clocks.append(pet.p.composer.ear_hold[PLUR_CLOCK_EAR])
        # it concentrates: looking at the person, and nothing else is allowed to start
        assert beh.gaze == (20.0, -3.0)
        assert beh._next_glance > now and beh._next_jingle > now and beh._look_until == 0.0
        now += 0.1
    # it called for all four itself, in order, without being told again
    order = [s for i, s in enumerate(called) if s and (i == 0 or s != called[i - 1])]
    assert order == list(PLUR_STEPS)
    assert set(pet.signs.trained) == set(PLUR_STEPS)  # and learned every one of them
    assert pet._training is None and pet.p.composer.ear_hold[PLUR_CLOCK_EAR] is None  # the clock is put away
    ticks = [c for c in clocks if c is not None]
    assert len(ticks) > 40 and max(ticks) - min(ticks) > 0.5  # a countdown really ran, all the way through
    assert "learning done: 4 pose(s)" in [n for _, k, n in pet.actions_log if k == "plur"]
    pet.stop()


def test_a_handshake_in_progress_gets_the_same_concentration():
    """'It does the peace thing and then back to chaos' — mid-handshake it used to carry on glancing at
    walls and humming, so there was no way to tell it had seen you or what it was waiting for."""
    from festival_pet.behavior import FaceObs
    from festival_pet.main import PLUR_CLOCK_EAR

    pet = _pet()
    beh = pet.p.behavior
    now = 1000.2
    pet.signs.plur_step, pet.signs.plur_at = 1, now  # peace has landed; it is waiting for love
    for _ in range(30):
        pet.step(now)
        pet._last_obs.face = FaceObs(1, -15.0, 2.0, 0.05, None, 0.0)
        pet._focus(pet._last_obs, now, 8.0, 12.0)
        now += 0.05
    assert pet._last_obs.busy == "kandi"  # the brain starts nothing while a handshake is going
    assert beh.gaze == (-15.0, 2.0) and pet.p.composer._gaze_target == (-15.0, 2.0)
    assert pet.p.composer.ear_hold[PLUR_CLOCK_EAR] is not None  # the window is showing on an antenna
    assert beh._next_jingle > now  # ...and it is not about to start humming at you
    # the handshake lapses: it lets go of the antenna and goes back to being a pet
    pet.signs.plur_step = 0
    pet.step(now)
    assert pet._focus_gaze is None and pet.p.composer.ear_hold[PLUR_CLOCK_EAR] is None
    pet.stop()


def test_two_poses_it_cannot_tell_apart_are_called_out_rather_than_swallowed():
    """Teaching it two poses that measure the same would make the handshake ambiguous for good — and
    silently: it would answer whichever came first every time."""
    from festival_pet.pose import Arms

    def arms(ts):
        pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (150.0, 100.0), "l_elbow": (100.0, 120.0),
               "r_elbow": (150.0, 120.0), "l_wrist": (105.0, 112.0), "r_wrist": (146.0, 110.0),
               "l_wrist_ok": True, "r_wrist_ok": True}
        return Arms(ts, 55.0, 57.0, "out", "out", 0.9, 50.0, pts)

    pet = _pet()
    now = 1000.2
    for step in ("peace", "love"):  # the same arms both times
        pet.train_plur(step, now)
        while pet._training is not None and now < 1000.2 + 60.0:
            pet._train_tick(Observation(arms=arms(now)), now)
            now += 0.1
    assert set(pet.signs.trained) == {"peace", "love"}  # it keeps both: you may be redoing one on purpose
    assert any("looks the same as peace" in n for _, k, n in pet.actions_log if k == "plur")
    pet.stop()
