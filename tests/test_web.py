"""The pet's web API with a fully fake robot (no SDK, no sim)."""

import logging
import threading

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
    assert c.post("/api/control", json={"cmd": "wake"}).status_code == 200
    assert c.post("/api/control", json={"cmd": "mute", "value": True}).json() == {"ok": True}
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
    for k in range(40):
        pet.step(1000.3 + k * 0.1)  # 4 s
    assert pet._touch_resync_at < 0  # re-zeroed to the resting position, no false ear tickle woke it
    assert pet.asleep
    pet.p.behavior.state = "SLEEPING"
    pet._dispatch(Action("wake", "name", 5), 1005.0)
    assert not pet.asleep and getattr(io, "woke", False)
    pet.stop()
