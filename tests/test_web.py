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
    for style in ("drumline", "bass"):
        assert c.post("/api/control", json={"cmd": "sing", "value": style}).status_code == 200
        assert pet.last_song["style"] == style and c.get("/api/mind").json()["song"]["last"]["style"] == style
    assert c.post("/api/control", json={"cmd": "sing", "value": "polka"}).status_code == 400
    # a repertoire saved before styles existed still loads: those songs are drumline ones
    pet.songs_file.write_text(json.dumps([{"bpm": 100, "bars": ["quarters", "roll_and_stop"], "hi": 1500, "lo": 1000, "name": "old one"}]))
    pet5 = _pet(); pet5.songs_file = pet.songs_file; pet5.load_songs()
    assert pet5.songs[0]["style"] == "drumline" and songs.render(pet5.songs[0]).size > 0
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
        t += 0.02
        if pet._next_song_at:
            seen_encore_gap = True
        if pet.last_song["name"] not in songs_sung:
            songs_sung.append(pet.last_song["name"])
        if not (t < pet._singing_until or pet._next_song_at):
            break
    assert seen_encore_gap and pet._songs_in_set == 0  # the set ran and then ended
    assert 2 <= len(songs_sung) <= 3  # an encore, and sometimes a third
    assert comp._gesture is not None and comp._gesture.name == "bow"  # the set earned the whole routine
    assert any("thank you" in x[1] for x in beh.thoughts)
    assert any(k == "song" and "encore" in n for _, k, n in pet.actions_log)
    # the gap between songs still counts as performing, so the brain does not go and start something else
    pet.sing(t + 1.0)
    pet._singing_until = t + 1.07  # a step that lands in the last 20 ms of the song is the one that ends it
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

    def arms(ts, l_deg, r_deg, r_dx=0.0):
        pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (150.0, 100.0), "l_wrist": (100.0, 40.0), "r_wrist": (150.0 + r_dx, 40.0), "l_wrist_ok": True, "r_wrist_ok": True}
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
        pet.vision.arms = arms(now, 90.0, 90.0)
        pet.step(now)
        now += 0.1
    assert any(k == "arms" and n == "hug" for _, k, n in pet.actions_log)
    assert comp._gesture.name == "hug" and "a hug" in beh.thoughts[-1][1]
    assert [n for _, k, n in pet.actions_log if k == "sound"][-1] == "coo"
    assert pet.mind()["arms"]["watching"]
    pet.stop()


def test_the_plur_handshake_ends_with_it_holding_still_for_the_bracelet(tmp_path):
    import numpy as np

    from festival_pet.main import KANDI_STILL_S
    from festival_pet.pose import PLUR_STEPS
    from test_pose import PLUR_POSES

    t = time.time()
    pet = _pet(t - 1.0)
    beh, comp = pet.p.behavior, pet.p.composer
    pet.vision = ArmsVision()
    mem = pet.p.memory
    mem.path = tmp_path / "mem.json"  # this test enrols someone, so give it somewhere of its own to save
    beh._engaged_person = mem.enroll(np.ones(4), t)
    kandi_before = beh._engaged_person.kandi

    now = t
    for step in PLUR_STEPS:  # peace, love, unity, respect, each held for a beat
        end = now + 1.2
        while now < end:
            pet.vision.arms = PLUR_POSES[step](now)
            pet.step(now)
            now += 0.05
        now += 0.3
    done = [n for _, k, n in pet.actions_log if k == "arms"]
    assert [n for n in done if n.startswith("plur")] == [f"plur {s}" for s in PLUR_STEPS]
    thoughts = " ".join(x[1] for x in beh.thoughts)
    assert "peace" in thoughts and "love" in thoughts and "unity" in thoughts and "respect" in thoughts
    assert pet._last_obs.busy == "kandi"  # nothing else gets started in the middle of this
    # the finale: it stops dead so the bracelet can go on without a fight
    assert comp.still_until > now and pet._kandi_at > now
    for _ in range(40):
        pet.step(now)
        now += 0.05
    head_a, _, _ = comp.sample(now, 0.02)
    head_b, _, _ = comp.sample(now + 0.5, 0.02)
    assert float(np.abs(head_a - head_b).max()) < 0.01  # not breathing, not drifting: holding it
    # and when the time is up it has a look at what it was given, and remembers who gave it
    while now < pet._kandi_at + 0.2:
        pet.step(now)
        now += 0.05
    assert "kandi traded" in [n for _, k, n in pet.actions_log if k == "arms"]
    assert beh._engaged_person.kandi == kandi_before + 1 and beh._engaged_person.affection > 0.2
    assert beh._little_dance_until > now
    assert pet._kandi_at == 0.0 and KANDI_STILL_S > 3.0
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
    from festival_pet.pose import Arms

    deg = {"down": 10.0, "out": 90.0, "up": 170.0}
    return Arms(ts, deg[left], deg[right], left, right, 0.9, 50.0, {})


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
